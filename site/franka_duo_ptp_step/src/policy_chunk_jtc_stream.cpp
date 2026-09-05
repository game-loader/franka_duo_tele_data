// Live action-chunk executor for the real Franka Duo Mobile.
//
// Subscribes to the evaluator's Float32MultiArray action chunk
// ([horizon x 20] DP3 Cartesian actions), solves MoveIt KDL IK for both arms
// row by row (seeded from the previous row with consistency limits), resamples
// the joint path from the policy rate to the stream rate, anchors it at the
// measured joint state and publishes one complete JointTrajectory per arm to
// the site relay topics.  jtc_command_relay is the only process that forwards
// to the joint_trajectory_controller command topics.  Each new chunk replaces
// the running trajectory, so the arms never stop between chunks as long as the
// evaluator publishes the next chunk before the previous one ends.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/duration.hpp>
#include <Eigen/Geometry>

#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace {

using JointState = sensor_msgs::msg::JointState;
using JointTrajectory = trajectory_msgs::msg::JointTrajectory;
using Float32MultiArray = std_msgs::msg::Float32MultiArray;

constexpr std::size_t kArmJointCount = 7;
constexpr std::size_t kActionDim = 20;

// Shared midpoint ("newbase") training frame expressed in each arm link0.
constexpr std::array<double, 16> kLeftMidpointFromArmBase{
    0.8809676, 0.40120238, 0.25086382, 0.0,
    -0.44015086, 0.50023556, 0.74567527, 0.05018,
    0.17367569, -0.7673337, 0.6172809, 0.0,
    0.0, 0.0, 0.0, 1.0};

constexpr std::array<double, 16> kRightMidpointFromArmBase{
    0.8809676, -0.40120238, 0.25086382, 0.0,
    0.44015086, 0.50023556, -0.74567527, -0.05018,
    0.17367569, 0.7673337, 0.6172809, 0.0,
    0.0, 0.0, 0.0, 1.0};

Eigen::Matrix3d rotationFromRot6d(const double* values) {
  Eigen::Vector3d first(values[0], values[1], values[2]);
  Eigen::Vector3d second(values[3], values[4], values[5]);
  if (!first.allFinite() || !second.allFinite() || first.norm() < 1e-8) {
    throw std::invalid_argument("rot6d contains a degenerate first row");
  }
  first.normalize();
  second -= first.dot(second) * first;
  if (second.norm() < 1e-8) {
    throw std::invalid_argument("rot6d rows are collinear");
  }
  second.normalize();
  const Eigen::Vector3d third = first.cross(second).normalized();
  Eigen::Matrix3d rotation;
  rotation.row(0) = first.transpose();
  rotation.row(1) = second.transpose();
  rotation.row(2) = third.transpose();
  if (!rotation.allFinite() ||
      (rotation * rotation.transpose() - Eigen::Matrix3d::Identity()).norm() > 1e-6 ||
      std::abs(rotation.determinant() - 1.0) > 1e-6) {
    throw std::invalid_argument("rot6d did not produce a valid rotation");
  }
  return rotation;
}

Eigen::Isometry3d poseFromRow(const double* xyz, const double* rot6d) {
  const Eigen::Vector3d translation(xyz[0], xyz[1], xyz[2]);
  if (!translation.allFinite()) {
    throw std::invalid_argument("target xyz contains non-finite values");
  }
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = rotationFromRot6d(rot6d);
  result.translation() = translation;
  return result;
}

Eigen::Isometry3d transformFromRowMajor(const std::array<double, 16>& values) {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
          values[row * 4 + column];
    }
  }
  return result;
}

std::optional<double> findJointPosition(
    const JointState& message, const std::string& expected_name, std::size_t joint_index) {
  for (std::size_t index = 0; index < message.name.size() && index < message.position.size();
       ++index) {
    if (message.name[index] == expected_name && std::isfinite(message.position[index])) {
      return message.position[index];
    }
  }
  const std::string suffix = "_joint" + std::to_string(joint_index + 1);
  for (std::size_t index = 0; index < message.name.size() && index < message.position.size();
       ++index) {
    const auto& name = message.name[index];
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0 &&
        name.find("finger") == std::string::npos &&
        name.find("gripper") == std::string::npos && std::isfinite(message.position[index])) {
      return message.position[index];
    }
  }
  return std::nullopt;
}

struct JointSample {
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> accelerations;
};

void fillDerivatives(std::vector<JointSample>& samples, double dt) {
  if (samples.empty()) {
    return;
  }
  const auto n = samples.size();
  const auto joints = samples.front().positions.size();
  for (auto& sample : samples) {
    sample.velocities.assign(joints, 0.0);
    sample.accelerations.assign(joints, 0.0);
  }
  if (n == 1U) {
    return;
  }
  const double inv = 1.0 / dt;
  const double inv2 = inv * inv;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < joints; ++j) {
      const auto p = [&](std::size_t k) { return samples[k].positions[j]; };
      if (i == 0U) {
        samples[i].velocities[j] = (p(1) - p(0)) * inv;
      } else if (i + 1U == n) {
        samples[i].velocities[j] = (p(i) - p(i - 1)) * inv;
      } else {
        samples[i].velocities[j] = (p(i + 1) - p(i - 1)) * 0.5 * inv;
      }
      if (n >= 3U) {
        if (i == 0U) {
          samples[i].accelerations[j] = (p(2) - 2.0 * p(1) + p(0)) * inv2;
        } else if (i + 1U == n) {
          samples[i].accelerations[j] = (p(i) - 2.0 * p(i - 1) + p(i - 2)) * inv2;
        } else {
          samples[i].accelerations[j] = (p(i + 1) - 2.0 * p(i) + p(i - 1)) * inv2;
        }
      }
    }
  }
  // The anchor is the measured state and the JTC config rejects a non-zero
  // velocity at the trajectory end, so both boundaries stay stationary.  The
  // evaluator replaces the chunk before the end is reached, which is what
  // keeps the motion continuous.
  std::fill(samples.front().velocities.begin(), samples.front().velocities.end(), 0.0);
  std::fill(samples.front().accelerations.begin(), samples.front().accelerations.end(), 0.0);
  std::fill(samples.back().velocities.begin(), samples.back().velocities.end(), 0.0);
  std::fill(samples.back().accelerations.begin(), samples.back().accelerations.end(), 0.0);
}

std::vector<JointSample> resample(
    const std::vector<std::vector<double>>& path, double source_rate_hz, double target_rate_hz) {
  std::vector<JointSample> samples;
  if (path.empty()) {
    return samples;
  }
  const double source_dt = 1.0 / source_rate_hz;
  const double target_dt = 1.0 / target_rate_hz;
  const double duration = static_cast<double>(path.size() - 1U) * source_dt;
  const std::size_t count = std::max<std::size_t>(
      1U, static_cast<std::size_t>(std::ceil(duration * target_rate_hz)) + 1U);
  samples.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double t = std::min(duration, static_cast<double>(i) * target_dt);
    const double position = t / source_dt;
    const auto lower = std::min<std::size_t>(
        path.size() - 1U, static_cast<std::size_t>(std::floor(position)));
    const auto upper = std::min<std::size_t>(path.size() - 1U, lower + 1U);
    const double alpha = upper == lower ? 0.0 : position - static_cast<double>(lower);
    JointSample sample;
    sample.positions.resize(kArmJointCount);
    for (std::size_t j = 0; j < kArmJointCount; ++j) {
      sample.positions[j] = (1.0 - alpha) * path[lower][j] + alpha * path[upper][j];
    }
    samples.push_back(std::move(sample));
  }
  fillDerivatives(samples, target_dt);
  return samples;
}

builtin_interfaces::msg::Duration durationMessage(double seconds) {
  const auto total = static_cast<std::int64_t>(std::llround(seconds * 1'000'000'000.0));
  builtin_interfaces::msg::Duration result;
  result.sec = static_cast<std::int32_t>(total / 1'000'000'000LL);
  result.nanosec = static_cast<std::uint32_t>(total % 1'000'000'000LL);
  return result;
}

JointTrajectory makeTrajectory(
    const JointSample& anchor,
    const std::vector<JointSample>& samples,
    const std::vector<std::string>& joint_names,
    double sample_period_s,
    double initial_delay_s) {
  JointTrajectory trajectory;
  trajectory.joint_names = joint_names;
  trajectory.points.reserve(samples.size() + 1U);
  trajectory_msgs::msg::JointTrajectoryPoint anchor_point;
  anchor_point.positions = anchor.positions;
  anchor_point.velocities = anchor.velocities;
  anchor_point.accelerations = anchor.accelerations;
  anchor_point.time_from_start = durationMessage(0.0);
  trajectory.points.push_back(std::move(anchor_point));
  for (std::size_t i = 0; i < samples.size(); ++i) {
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = samples[i].positions;
    point.velocities = samples[i].velocities;
    point.accelerations = samples[i].accelerations;
    point.time_from_start =
        durationMessage(initial_delay_s + static_cast<double>(i) * sample_period_s);
    trajectory.points.push_back(std::move(point));
  }
  return trajectory;
}

}  // namespace

class PolicyChunkJTCStream final : public rclcpp::Node {
 public:
  PolicyChunkJTCStream() : Node("policy_chunk_jtc_stream") {
    declare_parameter<std::string>("chunk_topic", "/franka_duo/policy_action_chunk");
    declare_parameter<std::string>(
        "left_joint_topic", "/left/franka_robot_state_broadcaster/measured_joint_states");
    declare_parameter<std::string>(
        "right_joint_topic", "/right/franka_robot_state_broadcaster/measured_joint_states");
    declare_parameter<std::string>(
        "left_trajectory_topic", "/franka_duo/eval/left/joint_trajectory");
    declare_parameter<std::string>(
        "right_trajectory_topic", "/franka_duo/eval/right/joint_trajectory");
    declare_parameter<std::string>(
        "left_gripper_topic", "/left/gripper/gripper_client/target_gripper_width_percent");
    declare_parameter<std::string>(
        "right_gripper_topic", "/right/gripper/gripper_client/target_gripper_width_percent");
    declare_parameter<std::string>("left_group_name", "left_arm");
    declare_parameter<std::string>("right_group_name", "right_arm");
    declare_parameter<std::string>("left_arm_base_link", "left_fr3v2_link0");
    declare_parameter<std::string>("right_arm_base_link", "right_fr3v2_link0");
    declare_parameter<std::string>("left_tip_link", "left_fr3v2_link8");
    declare_parameter<std::string>("right_tip_link", "right_fr3v2_link8");
    // link0: rows are already in each arm's link0 frame (the evaluator applies
    // the manifest link0_from_base transforms before publishing).
    // midpoint: rows are in the shared training frame; the fixed transforms
    // above are applied here.
    declare_parameter<std::string>("action_frame", "link0");
    declare_parameter<double>("tool_offset_z_m", 0.174);
    declare_parameter<double>("ik_timeout_s", 0.02);
    declare_parameter<double>("max_joint_delta_rad", 0.5);
    declare_parameter<double>("input_action_rate_hz", 15.0);
    declare_parameter<double>("stream_rate_hz", 50.0);
    declare_parameter<double>("jtc_initial_delay_s", 0.1);
    declare_parameter<double>("max_joint_velocity_rad_s", 1.5);
    declare_parameter<double>("max_chunk_age_s", 0.5);
    declare_parameter<double>("joint_state_timeout_s", 0.5);
    declare_parameter<double>("wait_timeout_s", 10.0);
    declare_parameter<bool>("execute", false);
    declare_parameter<bool>("confirm", false);
    declare_parameter<bool>("enable_gripper", false);

    chunk_topic_ = get_parameter("chunk_topic").as_string();
    left_joint_topic_ = get_parameter("left_joint_topic").as_string();
    right_joint_topic_ = get_parameter("right_joint_topic").as_string();
    left_trajectory_topic_ = get_parameter("left_trajectory_topic").as_string();
    right_trajectory_topic_ = get_parameter("right_trajectory_topic").as_string();
    left_gripper_topic_ = get_parameter("left_gripper_topic").as_string();
    right_gripper_topic_ = get_parameter("right_gripper_topic").as_string();
    left_group_name_ = get_parameter("left_group_name").as_string();
    right_group_name_ = get_parameter("right_group_name").as_string();
    left_arm_base_link_ = get_parameter("left_arm_base_link").as_string();
    right_arm_base_link_ = get_parameter("right_arm_base_link").as_string();
    left_tip_link_ = get_parameter("left_tip_link").as_string();
    right_tip_link_ = get_parameter("right_tip_link").as_string();
    action_frame_ = get_parameter("action_frame").as_string();
    tool_offset_z_m_ = get_parameter("tool_offset_z_m").as_double();
    ik_timeout_s_ = get_parameter("ik_timeout_s").as_double();
    max_joint_delta_rad_ = get_parameter("max_joint_delta_rad").as_double();
    input_action_rate_hz_ = get_parameter("input_action_rate_hz").as_double();
    stream_rate_hz_ = get_parameter("stream_rate_hz").as_double();
    jtc_initial_delay_s_ = get_parameter("jtc_initial_delay_s").as_double();
    max_joint_velocity_rad_s_ = get_parameter("max_joint_velocity_rad_s").as_double();
    max_chunk_age_s_ = get_parameter("max_chunk_age_s").as_double();
    joint_state_timeout_s_ = get_parameter("joint_state_timeout_s").as_double();
    wait_timeout_s_ = get_parameter("wait_timeout_s").as_double();
    execute_ = get_parameter("execute").as_bool();
    confirm_ = get_parameter("confirm").as_bool();
    enable_gripper_ = get_parameter("enable_gripper").as_bool();

    if (action_frame_ != "link0" && action_frame_ != "midpoint") {
      throw std::invalid_argument("action_frame must be link0 or midpoint");
    }
    const auto positive = [](double value) { return std::isfinite(value) && value > 0.0; };
    if (!std::isfinite(tool_offset_z_m_) || !positive(ik_timeout_s_) ||
        !positive(max_joint_delta_rad_) || !positive(input_action_rate_hz_) ||
        !positive(stream_rate_hz_) || !positive(max_joint_velocity_rad_s_) ||
        !positive(max_chunk_age_s_) || !positive(joint_state_timeout_s_) ||
        !positive(wait_timeout_s_) || !std::isfinite(jtc_initial_delay_s_) ||
        jtc_initial_delay_s_ < 0.0) {
      throw std::invalid_argument("streaming parameters must be finite and positive");
    }

    left_joint_subscription_ = create_subscription<JointState>(
        left_joint_topic_, rclcpp::SensorDataQoS(),
        [this](const JointState::SharedPtr message) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          left_joints_ = message;
          left_joints_time_ = std::chrono::steady_clock::now();
        });
    right_joint_subscription_ = create_subscription<JointState>(
        right_joint_topic_, rclcpp::SensorDataQoS(),
        [this](const JointState::SharedPtr message) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          right_joints_ = message;
          right_joints_time_ = std::chrono::steady_clock::now();
        });
    left_publisher_ =
        create_publisher<JointTrajectory>(left_trajectory_topic_, rclcpp::QoS(10).reliable());
    right_publisher_ =
        create_publisher<JointTrajectory>(right_trajectory_topic_, rclcpp::QoS(10).reliable());
    left_gripper_publisher_ =
        create_publisher<std_msgs::msg::Float32>(left_gripper_topic_, rclcpp::QoS(10).reliable());
    right_gripper_publisher_ =
        create_publisher<std_msgs::msg::Float32>(right_gripper_topic_, rclcpp::QoS(10).reliable());
  }

  void initialize() {
    waitForJointSamples();
    model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(shared_from_this());
    const auto& robot_model = model_loader_->getModel();
    if (robot_model == nullptr) {
      throw std::runtime_error("MoveIt robot model is unavailable");
    }
    left_group_ = robot_model->getJointModelGroup(left_group_name_);
    right_group_ = robot_model->getJointModelGroup(right_group_name_);
    if (left_group_ == nullptr || right_group_ == nullptr) {
      throw std::runtime_error("MoveIt left_arm/right_arm group is unavailable");
    }
    if (left_group_->getVariableCount() != kArmJointCount ||
        right_group_->getVariableCount() != kArmJointCount) {
      throw std::runtime_error("Both MoveIt arm groups must contain seven variables");
    }
    if (left_group_->getSolverInstance() == nullptr ||
        right_group_->getSolverInstance() == nullptr) {
      throw std::runtime_error("MoveIt KDL solver is unavailable for one arm");
    }
    left_joint_names_ = left_group_->getVariableNames();
    right_joint_names_ = right_group_->getVariableNames();
    state_ = std::make_unique<moveit::core::RobotState>(robot_model);
    state_->setToDefaultValues();
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      seedGroup(*state_, left_group_, *left_joints_);
      seedGroup(*state_, right_group_, *right_joints_);
    }
    state_->update();
    left_midpoint_from_arm_base_ = transformFromRowMajor(kLeftMidpointFromArmBase);
    right_midpoint_from_arm_base_ = transformFromRowMajor(kRightMidpointFromArmBase);

    RCLCPP_INFO(
        get_logger(),
        "Ready: chunk_topic=%s action_frame=%s input_rate=%.1fHz stream_rate=%.1fHz "
        "initial_delay=%.3fs execute=%s confirm=%s enable_gripper=%s",
        chunk_topic_.c_str(), action_frame_.c_str(), input_action_rate_hz_, stream_rate_hz_,
        jtc_initial_delay_s_, execute_ ? "true" : "false", confirm_ ? "true" : "false",
        enable_gripper_ ? "true" : "false");
    if (!execute_ || !confirm_) {
      RCLCPP_WARN(
          get_logger(),
          "Dry-run: chunks are solved and logged but no JointTrajectory is published. "
          "Use execute:=true confirm:=true after the trajectory controllers are active.");
    }
    chunk_subscription_ = create_subscription<Float32MultiArray>(
        chunk_topic_, rclcpp::QoS(10).reliable(),
        [this](const Float32MultiArray::SharedPtr message) { onChunk(*message); });
  }

 private:
  void waitForJointSamples() {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(wait_timeout_s_);
    rclcpp::WallRate rate(100.0);
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (left_joints_ != nullptr && right_joints_ != nullptr) {
          return;
        }
      }
      rclcpp::spin_some(shared_from_this());
      rate.sleep();
    }
    throw std::runtime_error(
        "Timed out waiting for " + left_joint_topic_ + " and " + right_joint_topic_);
  }

  static void seedGroup(
      moveit::core::RobotState& state,
      const moveit::core::JointModelGroup* group,
      const JointState& measured) {
    std::map<std::string, double> seed;
    const auto& names = group->getVariableNames();
    for (std::size_t index = 0; index < names.size(); ++index) {
      const auto value = findJointPosition(measured, names[index], index);
      if (!value.has_value()) {
        throw std::runtime_error("Measured joint state is missing " + names[index]);
      }
      seed[names[index]] = value.value();
    }
    state.setVariablePositions(seed);
    state.update();
  }

  static JointSample measuredSample(
      const JointState& measured, const std::vector<std::string>& joint_names) {
    JointSample sample;
    sample.velocities.assign(joint_names.size(), 0.0);
    sample.accelerations.assign(joint_names.size(), 0.0);
    for (std::size_t index = 0; index < joint_names.size(); ++index) {
      const auto value = findJointPosition(measured, joint_names[index], index);
      if (!value.has_value()) {
        throw std::runtime_error("Measured joint state is missing " + joint_names[index]);
      }
      sample.positions.push_back(value.value());
    }
    return sample;
  }

  // Solve one arm for one row; returns false when IK fails or the solution
  // jumps too far from the seed.  On success the state carries the solution
  // as the seed for the next row.
  bool solveArm(
      const moveit::core::JointModelGroup* group,
      const std::string& arm_base_link,
      const std::string& tip_link,
      const Eigen::Isometry3d& midpoint_from_arm_base,
      const double* xyz,
      const double* rot6d,
      std::vector<double>& solution,
      std::string& error) {
    const auto* arm_base = state_->getRobotModel()->getLinkModel(arm_base_link);
    const auto* tip = state_->getRobotModel()->getLinkModel(tip_link);
    if (arm_base == nullptr || tip == nullptr) {
      error = "MoveIt link not found";
      return false;
    }
    std::vector<double> seed;
    state_->copyJointGroupPositions(group, seed);
    Eigen::Isometry3d target_ee_in_arm_base;
    try {
      const Eigen::Isometry3d pose = poseFromRow(xyz, rot6d);
      target_ee_in_arm_base =
          action_frame_ == "midpoint" ? midpoint_from_arm_base.inverse() * pose : pose;
    } catch (const std::exception& exception) {
      error = exception.what();
      return false;
    }
    const Eigen::Isometry3d target_ee =
        state_->getGlobalLinkTransform(arm_base) * target_ee_in_arm_base;
    Eigen::Isometry3d tip_to_tool = Eigen::Isometry3d::Identity();
    tip_to_tool.translation().z() = tool_offset_z_m_;
    const Eigen::Isometry3d target_tip = target_ee * tip_to_tool.inverse();

    const std::vector<double> consistency_limits(kArmJointCount, max_joint_delta_rad_);
    if (!state_->setFromIK(group, target_tip, tip_link, consistency_limits, ik_timeout_s_)) {
      state_->setJointGroupPositions(group, seed);
      state_->update();
      error = "KDL IK failed";
      return false;
    }
    state_->update();
    if (!state_->satisfiesBounds(group)) {
      state_->setJointGroupPositions(group, seed);
      state_->update();
      error = "IK solution is outside MoveIt bounds";
      return false;
    }
    state_->copyJointGroupPositions(group, solution);
    double largest = 0.0;
    for (std::size_t j = 0; j < kArmJointCount; ++j) {
      largest = std::max(largest, std::abs(solution[j] - seed[j]));
    }
    if (largest > max_joint_delta_rad_) {
      state_->setJointGroupPositions(group, seed);
      state_->update();
      error = "IK solution jumps too far from the seed";
      return false;
    }
    return true;
  }

  void onChunk(const Float32MultiArray& message) {
    const auto received = std::chrono::steady_clock::now();
    if (message.data.empty() || message.data.size() % kActionDim != 0U) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Ignoring chunk with %zu values; expected a multiple of %zu",
          message.data.size(), kActionDim);
      return;
    }
    const std::size_t rows = message.data.size() / kActionDim;
    if (message.layout.dim.size() >= 2U &&
        (message.layout.dim[0].size != rows || message.layout.dim[1].size != kActionDim)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Ignoring chunk whose layout [%u x %u] does not match %zu x %zu values",
          message.layout.dim[0].size, message.layout.dim[1].size, rows, kActionDim);
      return;
    }
    for (const auto value : message.data) {
      if (!std::isfinite(value)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Ignoring chunk with non-finite values");
        return;
      }
    }
    // data_offset carries the number of rows the evaluator executes before
    // the next chunk arrives.  The whole chunk is streamed so the arms keep
    // moving if the next chunk is late; the value is only logged here.
    const std::size_t execute_rows =
        message.layout.data_offset > 0U
            ? std::min<std::size_t>(rows, message.layout.data_offset)
            : rows;

    JointState::SharedPtr left_measured;
    JointState::SharedPtr right_measured;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      left_measured = left_joints_;
      right_measured = right_joints_;
      const auto age_left = std::chrono::duration<double>(received - left_joints_time_).count();
      const auto age_right = std::chrono::duration<double>(received - right_joints_time_).count();
      if (age_left > joint_state_timeout_s_ || age_right > joint_state_timeout_s_) {
        RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Measured joint states are stale (left %.3fs, right %.3fs); dropping chunk",
            age_left, age_right);
        return;
      }
    }

    // Seed the first row from the measured state so IK follows the real arm,
    // then chain each row from the previous solution.
    JointSample left_anchor;
    JointSample right_anchor;
    try {
      left_anchor = measuredSample(*left_measured, left_joint_names_);
      right_anchor = measuredSample(*right_measured, right_joint_names_);
    } catch (const std::exception& exception) {
      RCLCPP_ERROR(get_logger(), "%s", exception.what());
      return;
    }
    state_->setJointGroupPositions(left_group_, left_anchor.positions);
    state_->setJointGroupPositions(right_group_, right_anchor.positions);
    state_->update();

    std::vector<std::vector<double>> left_path;
    std::vector<std::vector<double>> right_path;
    left_path.reserve(rows);
    right_path.reserve(rows);
    std::string error;
    std::size_t solved_rows = 0;
    for (std::size_t row = 0; row < rows; ++row) {
      std::array<double, kActionDim> action{};
      for (std::size_t k = 0; k < kActionDim; ++k) {
        action[k] = static_cast<double>(message.data[row * kActionDim + k]);
      }
      std::vector<double> left_solution;
      std::vector<double> right_solution;
      if (!solveArm(
              left_group_, left_arm_base_link_, left_tip_link_, left_midpoint_from_arm_base_,
              action.data(), action.data() + 3, left_solution, error)) {
        RCLCPP_WARN(get_logger(), "left row %zu/%zu: %s", row, rows, error.c_str());
        break;
      }
      if (!solveArm(
              right_group_, right_arm_base_link_, right_tip_link_, right_midpoint_from_arm_base_,
              action.data() + 9, action.data() + 12, right_solution, error)) {
        RCLCPP_WARN(get_logger(), "right row %zu/%zu: %s", row, rows, error.c_str());
        break;
      }
      left_path.push_back(std::move(left_solution));
      right_path.push_back(std::move(right_solution));
      ++solved_rows;
    }
    if (solved_rows == 0U) {
      RCLCPP_ERROR(get_logger(), "No row of the chunk could be solved; nothing published");
      return;
    }

    // Velocity guard on the policy-rate joint path, including the step from
    // the measured anchor to the first row.
    const double source_dt = 1.0 / input_action_rate_hz_;
    const double anchor_dt = std::max(source_dt, jtc_initial_delay_s_);
    double max_velocity = 0.0;
    const auto guard = [&](const std::vector<double>& anchor, const std::vector<std::vector<double>>& path) {
      for (std::size_t j = 0; j < kArmJointCount; ++j) {
        max_velocity = std::max(max_velocity, std::abs(path[0][j] - anchor[j]) / anchor_dt);
      }
      for (std::size_t i = 1; i < path.size(); ++i) {
        for (std::size_t j = 0; j < kArmJointCount; ++j) {
          max_velocity = std::max(max_velocity, std::abs(path[i][j] - path[i - 1][j]) / source_dt);
        }
      }
    };
    guard(left_anchor.positions, left_path);
    guard(right_anchor.positions, right_path);
    if (max_velocity > max_joint_velocity_rad_s_) {
      RCLCPP_ERROR(
          get_logger(),
          "Chunk requires %.3f rad/s, above max_joint_velocity_rad_s=%.3f; dropping chunk",
          max_velocity, max_joint_velocity_rad_s_);
      return;
    }

    const double stream_dt = 1.0 / stream_rate_hz_;
    const auto left_samples = resample(left_path, input_action_rate_hz_, stream_rate_hz_);
    const auto right_samples = resample(right_path, input_action_rate_hz_, stream_rate_hz_);
    const auto left_trajectory =
        makeTrajectory(left_anchor, left_samples, left_joint_names_, stream_dt, jtc_initial_delay_s_);
    const auto right_trajectory = makeTrajectory(
        right_anchor, right_samples, right_joint_names_, stream_dt, jtc_initial_delay_s_);
    const double horizon_s =
        jtc_initial_delay_s_ + static_cast<double>(left_samples.size() - 1U) * stream_dt;

    const double solve_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - received).count();
    if (solve_ms / 1000.0 > max_chunk_age_s_) {
      RCLCPP_ERROR(
          get_logger(), "Chunk processing took %.1f ms, above max_chunk_age_s; dropping chunk",
          solve_ms);
      return;
    }

    const bool publish = execute_ && confirm_;
    if (publish) {
      left_publisher_->publish(left_trajectory);
      right_publisher_->publish(right_trajectory);
    }
    if (publish && enable_gripper_) {
      const float left_open = message.data[18];
      const float right_open = message.data[19];
      if (left_open >= 0.0F && left_open <= 1.0F && right_open >= 0.0F && right_open <= 1.0F) {
        std_msgs::msg::Float32 left_gripper;
        std_msgs::msg::Float32 right_gripper;
        left_gripper.data = left_open;
        right_gripper.data = right_open;
        left_gripper_publisher_->publish(left_gripper);
        right_gripper_publisher_->publish(right_gripper);
      } else {
        RCLCPP_WARN(get_logger(), "Gripper targets outside [0, 1]; gripper not commanded");
      }
    }
    ++chunk_count_;
    RCLCPP_INFO(
        get_logger(),
        "%s chunk %zu: rows=%zu solved=%zu execute_rows=%zu points=%zu horizon=%.3fs "
        "max_velocity=%.3frad/s solve=%.1fms",
        publish ? "Published" : "Dry-run", chunk_count_, rows, solved_rows, execute_rows,
        left_trajectory.points.size(), horizon_s, max_velocity, solve_ms);
  }

  std::string chunk_topic_;
  std::string left_joint_topic_;
  std::string right_joint_topic_;
  std::string left_trajectory_topic_;
  std::string right_trajectory_topic_;
  std::string left_gripper_topic_;
  std::string right_gripper_topic_;
  std::string left_group_name_;
  std::string right_group_name_;
  std::string left_arm_base_link_;
  std::string right_arm_base_link_;
  std::string left_tip_link_;
  std::string right_tip_link_;
  std::string action_frame_;
  double tool_offset_z_m_{0.174};
  double ik_timeout_s_{0.02};
  double max_joint_delta_rad_{0.5};
  double input_action_rate_hz_{15.0};
  double stream_rate_hz_{50.0};
  double jtc_initial_delay_s_{0.1};
  double max_joint_velocity_rad_s_{1.5};
  double max_chunk_age_s_{0.5};
  double joint_state_timeout_s_{0.5};
  double wait_timeout_s_{10.0};
  bool execute_{false};
  bool confirm_{false};
  bool enable_gripper_{false};
  std::size_t chunk_count_{0};

  std::mutex state_mutex_;
  JointState::SharedPtr left_joints_;
  JointState::SharedPtr right_joints_;
  std::chrono::steady_clock::time_point left_joints_time_{};
  std::chrono::steady_clock::time_point right_joints_time_{};

  std::shared_ptr<robot_model_loader::RobotModelLoader> model_loader_;
  const moveit::core::JointModelGroup* left_group_{nullptr};
  const moveit::core::JointModelGroup* right_group_{nullptr};
  std::vector<std::string> left_joint_names_;
  std::vector<std::string> right_joint_names_;
  std::unique_ptr<moveit::core::RobotState> state_;
  Eigen::Isometry3d left_midpoint_from_arm_base_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d right_midpoint_from_arm_base_{Eigen::Isometry3d::Identity()};

  rclcpp::Subscription<JointState>::SharedPtr left_joint_subscription_;
  rclcpp::Subscription<JointState>::SharedPtr right_joint_subscription_;
  rclcpp::Subscription<Float32MultiArray>::SharedPtr chunk_subscription_;
  rclcpp::Publisher<JointTrajectory>::SharedPtr left_publisher_;
  rclcpp::Publisher<JointTrajectory>::SharedPtr right_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr left_gripper_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr right_gripper_publisher_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    auto node = std::make_shared<PolicyChunkJTCStream>();
    node->initialize();
    rclcpp::spin(node);
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("policy_chunk_jtc_stream"), "%s", exception.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
