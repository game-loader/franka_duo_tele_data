// Joint-space action-chunk servo for the Franka Duo Mobile.
//
// Subscribes to 20D Cartesian action chunks tagged with an absolute step
// index, solves MoveIt KDL IK row by row (seeded from the previously planned
// step), stores the joint targets on an absolute-step timeline and tracks
// that timeline at a fixed period with a seven-axis Ruckig position
// tracker per arm.  The tracker state is never reset when chunks change, so
// the commanded joint position, velocity and acceleration stay continuous.
//
// Output is a sensor_msgs/JointState target per arm on a site-owned topic.
// Only gello_target_relay forwards it to the site JointImpedanceController
// input (/{side}/gello/joint_states).  Before the first chunk arrives the
// node follows the measured joints so activating the impedance controller
// holds the current pose.  After the timeline ends, or on any fault, the
// tracker decelerates to rest and holds; the target stream never stops,
// because the site controller shuts the driver down after 0.5 s without
// targets.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include "franka_duo_joint_servo/joint_ruckig_tracker.hpp"
#include "franka_duo_joint_servo/joint_timeline.hpp"

namespace franka_duo_joint_servo {
namespace {

using JointState = sensor_msgs::msg::JointState;
using Float32MultiArray = std_msgs::msg::Float32MultiArray;

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

std::optional<double> findJointPosition(const JointState& message, const std::string& name) {
  for (std::size_t index = 0; index < message.name.size() && index < message.position.size();
       ++index) {
    if (message.name[index] == name && std::isfinite(message.position[index])) {
      return message.position[index];
    }
  }
  return std::nullopt;
}

Joints arrayParameter(rclcpp::Node& node, const std::string& name, const Joints& fallback) {
  const auto values = node.get_parameter(name).as_double_array();
  if (values.empty()) {
    return fallback;
  }
  Joints result{};
  if (values.size() == 1U) {
    result.fill(values.front());
  } else if (values.size() == kJoints) {
    std::copy(values.begin(), values.end(), result.begin());
  } else {
    throw std::invalid_argument(name + " must contain one or seven values");
  }
  return result;
}

}  // namespace

class PolicyChunkJointServo final : public rclcpp::Node {
 public:
  PolicyChunkJointServo() : Node("franka_duo_joint_servo") {
    declare_parameter<std::string>("chunk_topic", "/franka_duo/joint_servo/action_chunk");
    declare_parameter<std::string>("status_topic", "/franka_duo/joint_servo/status");
    declare_parameter<std::string>(
        "left_joint_topic", "/left/franka_robot_state_broadcaster/measured_joint_states");
    declare_parameter<std::string>(
        "right_joint_topic", "/right/franka_robot_state_broadcaster/measured_joint_states");
    declare_parameter<std::string>("left_target_topic", "/franka_duo/joint_servo/left/target");
    declare_parameter<std::string>("right_target_topic", "/franka_duo/joint_servo/right/target");
    declare_parameter<std::string>("left_gripper_topic", "/franka_duo/joint_servo/left/gripper");
    declare_parameter<std::string>("right_gripper_topic", "/franka_duo/joint_servo/right/gripper");
    declare_parameter<std::string>("left_group_name", "left_arm");
    declare_parameter<std::string>("right_group_name", "right_arm");
    declare_parameter<std::string>("left_arm_base_link", "left_fr3v2_link0");
    declare_parameter<std::string>("right_arm_base_link", "right_fr3v2_link0");
    declare_parameter<std::string>("left_tip_link", "left_fr3v2_link8");
    declare_parameter<std::string>("right_tip_link", "right_fr3v2_link8");
    // link0: rows already in each arm's link0 frame.  midpoint: shared
    // training frame; the fixed transforms above are applied here.
    declare_parameter<std::string>("action_frame", "link0");
    declare_parameter<double>("tool_offset_z_m", 0.174);
    declare_parameter<double>("ik_timeout_s", 0.02);
    declare_parameter<double>("max_joint_delta_rad", 0.35);
    declare_parameter<double>("action_rate_hz", 30.0);
    declare_parameter<double>("playback_speed", 0.1);
    declare_parameter<double>("servo_rate_hz", 1000.0);
    declare_parameter<double>("status_rate_hz", 30.0);
    declare_parameter<int>("commit_lead_steps", 3);
    declare_parameter<int>("blend_steps", 4);
    declare_parameter<std::vector<double>>("max_joint_velocity_rad_s", {0.8});
    declare_parameter<std::vector<double>>("max_joint_acceleration_rad_s2", {2.0});
    declare_parameter<std::vector<double>>("max_joint_jerk_rad_s3", {20.0});
    declare_parameter<double>("max_tracking_error_rad", 0.15);
    declare_parameter<double>("joint_state_timeout_s", 0.2);
    declare_parameter<double>("max_chunk_age_s", 0.3);
    declare_parameter<double>("wait_timeout_s", 10.0);
    // Idle mode follows the measured joints only for this long after the
    // first joint sample, then latches the target.  Following measured joints
    // while the impedance controller is active gives zero effective stiffness
    // and the arm drifts; the latch must happen before the controller is
    // activated.
    declare_parameter<double>("idle_follow_timeout_s", 20.0);
    declare_parameter<bool>("enable_gripper", false);

    chunk_topic_ = get_parameter("chunk_topic").as_string();
    status_topic_ = get_parameter("status_topic").as_string();
    left_joint_topic_ = get_parameter("left_joint_topic").as_string();
    right_joint_topic_ = get_parameter("right_joint_topic").as_string();
    left_target_topic_ = get_parameter("left_target_topic").as_string();
    right_target_topic_ = get_parameter("right_target_topic").as_string();
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
    action_rate_hz_ = get_parameter("action_rate_hz").as_double();
    playback_speed_ = get_parameter("playback_speed").as_double();
    servo_rate_hz_ = get_parameter("servo_rate_hz").as_double();
    status_rate_hz_ = get_parameter("status_rate_hz").as_double();
    commit_lead_steps_ = get_parameter("commit_lead_steps").as_int();
    blend_steps_ = get_parameter("blend_steps").as_int();
    max_tracking_error_rad_ = get_parameter("max_tracking_error_rad").as_double();
    joint_state_timeout_s_ = get_parameter("joint_state_timeout_s").as_double();
    max_chunk_age_s_ = get_parameter("max_chunk_age_s").as_double();
    wait_timeout_s_ = get_parameter("wait_timeout_s").as_double();
    idle_follow_timeout_s_ = get_parameter("idle_follow_timeout_s").as_double();
    enable_gripper_ = get_parameter("enable_gripper").as_bool();

    if (action_frame_ != "link0" && action_frame_ != "midpoint") {
      throw std::invalid_argument("action_frame must be link0 or midpoint");
    }
    const auto positive = [](double value) { return std::isfinite(value) && value > 0.0; };
    if (!std::isfinite(tool_offset_z_m_) || !positive(ik_timeout_s_) ||
        !positive(max_joint_delta_rad_) || !positive(action_rate_hz_) ||
        !positive(playback_speed_) || playback_speed_ > 1.0 || !positive(servo_rate_hz_) ||
        !positive(status_rate_hz_) || !positive(max_tracking_error_rad_) ||
        !positive(joint_state_timeout_s_) || !positive(max_chunk_age_s_) ||
        !positive(wait_timeout_s_) || commit_lead_steps_ < 0 || blend_steps_ < 0) {
      throw std::invalid_argument("servo parameters must be finite, positive and speed <= 1");
    }
    Joints fallback{};
    max_velocity_ = arrayParameter(*this, "max_joint_velocity_rad_s", fallback);
    const Joints max_acceleration = arrayParameter(*this, "max_joint_acceleration_rad_s2", fallback);
    const Joints max_jerk = arrayParameter(*this, "max_joint_jerk_rad_s3", fallback);
    servo_period_s_ = 1.0 / servo_rate_hz_;
    left_tracker_.configure(servo_period_s_, max_velocity_, max_acceleration, max_jerk);
    right_tracker_.configure(servo_period_s_, max_velocity_, max_acceleration, max_jerk);
    timeline_ = std::make_unique<JointTimeline>(1.0 / action_rate_hz_);

    left_joint_subscription_ = create_subscription<JointState>(
        left_joint_topic_, rclcpp::SensorDataQoS(),
        [this](const JointState::SharedPtr message) { onJoints(*message, true); });
    right_joint_subscription_ = create_subscription<JointState>(
        right_joint_topic_, rclcpp::SensorDataQoS(),
        [this](const JointState::SharedPtr message) { onJoints(*message, false); });
    left_target_publisher_ =
        create_publisher<JointState>(left_target_topic_, rclcpp::QoS(1).reliable());
    right_target_publisher_ =
        create_publisher<JointState>(right_target_topic_, rclcpp::QoS(1).reliable());
    left_gripper_publisher_ =
        create_publisher<std_msgs::msg::Float32>(left_gripper_topic_, rclcpp::QoS(10).reliable());
    right_gripper_publisher_ =
        create_publisher<std_msgs::msg::Float32>(right_gripper_topic_, rclcpp::QoS(10).reliable());
    status_publisher_ =
        create_publisher<std_msgs::msg::String>(status_topic_, rclcpp::QoS(10).reliable());
  }

  ~PolicyChunkJointServo() override {
    running_ = false;
    if (servo_thread_.joinable()) {
      servo_thread_.join();
    }
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
    if (left_group_->getVariableCount() != kJoints || right_group_->getVariableCount() != kJoints) {
      throw std::runtime_error("Both MoveIt arm groups must contain seven variables");
    }
    if (left_group_->getSolverInstance() == nullptr ||
        right_group_->getSolverInstance() == nullptr) {
      throw std::runtime_error("MoveIt KDL solver is unavailable for one arm");
    }
    left_joint_names_ = left_group_->getVariableNames();
    right_joint_names_ = right_group_->getVariableNames();
    // The site impedance controller copies positions by index, so the
    // published order must be joint1..joint7.
    for (std::size_t j = 0; j < kJoints; ++j) {
      const std::string suffix = "_joint" + std::to_string(j + 1);
      for (const auto* names : {&left_joint_names_, &right_joint_names_}) {
        const auto& name = (*names)[j];
        if (name.size() < suffix.size() ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
          throw std::runtime_error("MoveIt joint order is not joint1..joint7: " + name);
        }
      }
    }
    state_ = std::make_unique<moveit::core::RobotState>(robot_model);
    state_->setToDefaultValues();
    left_midpoint_from_arm_base_ = transformFromRowMajor(kLeftMidpointFromArmBase);
    right_midpoint_from_arm_base_ = transformFromRowMajor(kRightMidpointFromArmBase);

    Joints left_measured{};
    Joints right_measured{};
    {
      std::lock_guard<std::mutex> lock(joints_mutex_);
      left_measured = left_measured_;
      right_measured = right_measured_;
    }
    {
      std::lock_guard<std::mutex> lock(servo_mutex_);
      left_tracker_.reset(left_measured);
      right_tracker_.reset(right_measured);
    }

    RCLCPP_INFO(
        get_logger(),
        "Ready: chunk_topic=%s action_frame=%s action_rate=%.1fHz speed=%.3f servo=%.0fHz "
        "commit_lead=%d blend=%d max_v=%.3f enable_gripper=%s",
        chunk_topic_.c_str(), action_frame_.c_str(), action_rate_hz_, playback_speed_,
        servo_rate_hz_, commit_lead_steps_, blend_steps_, max_velocity_[0],
        enable_gripper_ ? "true" : "false");
    RCLCPP_WARN(
        get_logger(),
        "Idle: following measured joints on %s and %s. Robot motion additionally requires "
        "gello_target_relay enable_robot:=true and an active joint_impedance_controller.",
        left_target_topic_.c_str(), right_target_topic_.c_str());

    running_ = true;
    servo_thread_ = std::thread([this] { servoLoop(); });
    chunk_subscription_ = create_subscription<Float32MultiArray>(
        chunk_topic_, rclcpp::QoS(10).reliable(),
        [this](const Float32MultiArray::SharedPtr message) { onChunk(*message); });
  }

 private:
  void onJoints(const JointState& message, bool left) {
    const auto& names = left ? left_joint_names_ : right_joint_names_;
    Joints values{};
    if (names.empty()) {
      // Before MoveIt names are known, match on the joint suffix.
      for (std::size_t j = 0; j < kJoints; ++j) {
        const std::string suffix = "_joint" + std::to_string(j + 1);
        std::optional<double> found;
        for (std::size_t index = 0; index < message.name.size() && index < message.position.size();
             ++index) {
          const auto& name = message.name[index];
          if (name.size() >= suffix.size() &&
              name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0 &&
              name.find("finger") == std::string::npos && std::isfinite(message.position[index])) {
            found = message.position[index];
            break;
          }
        }
        if (!found.has_value()) {
          return;
        }
        values[j] = *found;
      }
    } else {
      for (std::size_t j = 0; j < kJoints; ++j) {
        const auto found = findJointPosition(message, names[j]);
        if (!found.has_value()) {
          return;
        }
        values[j] = *found;
      }
    }
    std::lock_guard<std::mutex> lock(joints_mutex_);
    if (left) {
      left_measured_ = values;
      left_measured_time_ = std::chrono::steady_clock::now();
      have_left_ = true;
    } else {
      right_measured_ = values;
      right_measured_time_ = std::chrono::steady_clock::now();
      have_right_ = true;
    }
  }

  void waitForJointSamples() {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(wait_timeout_s_);
    rclcpp::WallRate rate(100.0);
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(joints_mutex_);
        if (have_left_ && have_right_) {
          return;
        }
      }
      rclcpp::spin_some(shared_from_this());
      rate.sleep();
    }
    throw std::runtime_error(
        "Timed out waiting for " + left_joint_topic_ + " and " + right_joint_topic_);
  }

  bool solveArm(
      const moveit::core::JointModelGroup* group,
      const std::string& arm_base_link,
      const std::string& tip_link,
      const Eigen::Isometry3d& midpoint_from_arm_base,
      const double* xyz,
      const double* rot6d,
      const Joints& seed,
      Joints& solution,
      std::string& error) {
    const auto* arm_base = state_->getRobotModel()->getLinkModel(arm_base_link);
    const auto* tip = state_->getRobotModel()->getLinkModel(tip_link);
    if (arm_base == nullptr || tip == nullptr) {
      error = "MoveIt link not found";
      return false;
    }
    state_->setJointGroupPositions(group, std::vector<double>(seed.begin(), seed.end()));
    state_->update();
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

    const std::vector<double> consistency_limits(kJoints, max_joint_delta_rad_);
    bool solved = state_->setFromIK(group, target_tip, tip_link, consistency_limits, ik_timeout_s_);
    if (!solved) {
      // One retry with a longer budget before giving up on the chunk.
      state_->setJointGroupPositions(group, std::vector<double>(seed.begin(), seed.end()));
      state_->update();
      solved = state_->setFromIK(group, target_tip, tip_link, consistency_limits, 5.0 * ik_timeout_s_);
    }
    if (!solved) {
      std::ostringstream stream;
      stream.precision(4);
      stream << "KDL IK failed; seed=[";
      for (std::size_t j = 0; j < kJoints; ++j) {
        stream << seed[j] << (j + 1 < kJoints ? "," : "]");
      }
      const auto& t = target_tip.translation();
      stream << " target_tip=(" << t.x() << "," << t.y() << "," << t.z() << ")";
      const auto current = state_->getGlobalLinkTransform(tip).translation();
      stream << " seed_tip=(" << current.x() << "," << current.y() << "," << current.z() << ")";
      error = stream.str();
      return false;
    }
    state_->update();
    if (!state_->satisfiesBounds(group)) {
      error = "IK solution is outside MoveIt bounds";
      return false;
    }
    std::vector<double> values;
    state_->copyJointGroupPositions(group, values);
    double largest = 0.0;
    for (std::size_t j = 0; j < kJoints; ++j) {
      solution[j] = values[j];
      largest = std::max(largest, std::abs(solution[j] - seed[j]));
    }
    if (largest > max_joint_delta_rad_) {
      error = "IK solution jumps too far from the seed";
      return false;
    }
    return true;
  }

  void onChunk(const Float32MultiArray& message) {
    const auto received = std::chrono::steady_clock::now();
    if (fault_.load()) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000, "Servo is in fault hold; chunk ignored");
      return;
    }
    if (message.data.empty() || message.data.size() % kActionDim != 0U) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "Ignoring chunk with %zu values; expected k x %zu",
          message.data.size(), kActionDim);
      return;
    }
    const std::size_t rows = message.data.size() / kActionDim;
    if (message.layout.dim.size() != 2U || message.layout.dim[0].size != rows ||
        message.layout.dim[1].size != kActionDim) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Ignoring chunk whose layout does not declare [%zu x %zu]", rows, kActionDim);
      return;
    }
    for (const auto value : message.data) {
      if (!std::isfinite(value)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Ignoring chunk with non-finite values");
        return;
      }
    }
    // layout.data_offset carries the absolute step index of row zero.
    const auto start_step = static_cast<std::int64_t>(message.layout.data_offset);

    // Commit boundary and IK seed from the running timeline.
    std::int64_t commit_step = start_step;
    Joints left_seed{};
    Joints right_seed{};
    std::size_t blend = 0;
    {
      std::lock_guard<std::mutex> lock(servo_mutex_);
      if (started_) {
        const double now_step = currentStep(received);
        commit_step = std::max<std::int64_t>(
            start_step, static_cast<std::int64_t>(std::ceil(now_step)) + commit_lead_steps_);
        blend = static_cast<std::size_t>(blend_steps_);
        if (start_step > commit_step) {
          RCLCPP_ERROR(
              get_logger(), "Chunk starts at step %ld but the servo is at %.2f; dropping",
              static_cast<long>(start_step), now_step);
          return;
        }
        const auto last = timeline_->last_step();
        const TimelinePoint* previous = timeline_->at(commit_step - 1);
        if (previous == nullptr && last.has_value()) {
          previous = timeline_->at(*last);
        }
        if (previous != nullptr) {
          left_seed = previous->left;
          right_seed = previous->right;
        } else {
          left_seed = left_tracker_.position();
          right_seed = right_tracker_.position();
        }
      } else {
        left_seed = left_tracker_.position();
        right_seed = right_tracker_.position();
      }
    }
    if (commit_step - start_step >= static_cast<std::int64_t>(rows)) {
      RCLCPP_ERROR(
          get_logger(), "Chunk [%ld, %ld) lies entirely before commit step %ld; dropping",
          static_cast<long>(start_step), static_cast<long>(start_step + rows),
          static_cast<long>(commit_step));
      return;
    }

    // Solve IK only for rows at or beyond the commit boundary.
    std::vector<TimelinePoint> points;
    points.reserve(rows);
    std::string error;
    const auto first_row = static_cast<std::size_t>(commit_step - start_step);
    const auto solved_start = commit_step;
    double max_velocity = 0.0;
    const double step_period = timeline_->period() / playback_speed_;
    for (std::size_t row = first_row; row < rows; ++row) {
      std::array<double, kActionDim> action{};
      for (std::size_t k = 0; k < kActionDim; ++k) {
        action[k] = static_cast<double>(message.data[row * kActionDim + k]);
      }
      TimelinePoint point;
      if (!solveArm(
              left_group_, left_arm_base_link_, left_tip_link_, left_midpoint_from_arm_base_,
              action.data(), action.data() + 3, left_seed, point.left, error)) {
        RCLCPP_ERROR(get_logger(), "left row %zu/%zu: %s; dropping chunk", row, rows, error.c_str());
        return;
      }
      if (!solveArm(
              right_group_, right_arm_base_link_, right_tip_link_, right_midpoint_from_arm_base_,
              action.data() + 9, action.data() + 12, right_seed, point.right, error)) {
        RCLCPP_ERROR(get_logger(), "right row %zu/%zu: %s; dropping chunk", row, rows, error.c_str());
        return;
      }
      if (action[18] < 0.0 || action[18] > 1.0 || action[19] < 0.0 || action[19] > 1.0) {
        RCLCPP_ERROR(get_logger(), "row %zu gripper outside [0, 1]; dropping chunk", row);
        return;
      }
      point.left_gripper = action[18];
      point.right_gripper = action[19];
      // Velocity implied by consecutive planned steps at the playback speed.
      // The cross-chunk transition is checked after blending below. Treating
      // its whole offset as one step would reject a valid multi-step bridge.
      if (row > first_row) {
        for (std::size_t j = 0; j < kJoints; ++j) {
          max_velocity = std::max(max_velocity, std::abs(point.left[j] - left_seed[j]) / step_period);
          max_velocity = std::max(max_velocity, std::abs(point.right[j] - right_seed[j]) / step_period);
        }
      }
      left_seed = point.left;
      right_seed = point.right;
      points.push_back(point);
    }
    const double velocity_limit = *std::min_element(max_velocity_.begin(), max_velocity_.end());
    if (max_velocity > 0.9 * velocity_limit) {
      RCLCPP_ERROR(
          get_logger(),
          "Chunk needs %.3f rad/s at speed %.3f, above 0.9 x %.3f; dropping chunk",
          max_velocity, playback_speed_, velocity_limit);
      return;
    }
    const double solve_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - received).count();
    if (solve_s > max_chunk_age_s_) {
      RCLCPP_ERROR(get_logger(), "IK took %.1f ms, above max_chunk_age_s; dropping chunk", solve_s * 1e3);
      return;
    }

    std::size_t skipped = 0;
    double now_step = 0.0;
    {
      std::lock_guard<std::mutex> lock(servo_mutex_);
      if (fault_.load()) {
        return;
      }
      if (!started_) {
        commit_step = start_step;
      } else {
        now_step = currentStep(std::chrono::steady_clock::now());
        const auto min_commit = static_cast<std::int64_t>(std::ceil(now_step)) + commit_lead_steps_;
        // IK can cross a step boundary. Keep each solved row's absolute time
        // and discard the newly expired prefix instead of retiming it.
        commit_step = std::max(commit_step, min_commit);
      }
      if (commit_step - solved_start >= static_cast<std::int64_t>(points.size()) -
          static_cast<std::int64_t>(blend)) {
        RCLCPP_ERROR(get_logger(), "Chunk expired before a complete blend could be committed; dropping");
        return;
      }
      JointTimeline candidate = *timeline_;
      skipped = candidate.replace(solved_start, points, commit_step, blend);
      for (auto step = commit_step; step <= *candidate.last_step(); ++step) {
        const auto* previous = candidate.at(step - 1);
        const auto* point = candidate.at(step);
        if (previous == nullptr || point == nullptr) {
          continue;
        }
        for (std::size_t j = 0; j < kJoints; ++j) {
          max_velocity = std::max(max_velocity, std::abs(point->left[j] - previous->left[j]) / step_period);
          max_velocity = std::max(max_velocity, std::abs(point->right[j] - previous->right[j]) / step_period);
        }
      }
      if (max_velocity > 0.9 * velocity_limit) {
        RCLCPP_ERROR(get_logger(), "Blended chunk needs %.3f rad/s, above 0.9 x %.3f; dropping", max_velocity, velocity_limit);
        return;
      }
      if (!started_) {
        epoch_ = std::chrono::steady_clock::now();
        epoch_step_ = static_cast<double>(start_step);
        started_ = true;
      }
      *timeline_ = std::move(candidate);
      timeline_->trim_before(static_cast<std::int64_t>(std::floor(now_step)) - 2);
      last_chunk_start_step_ = commit_step;
      ++chunk_count_;
    }
    RCLCPP_INFO(
        get_logger(),
        "Chunk %zu: rows=%zu start=%ld commit=%ld planned=%zu skipped_late=%zu "
        "max_v=%.3frad/s ik=%.1fms",
        chunk_count_, rows, static_cast<long>(start_step), static_cast<long>(commit_step),
        points.size() - skipped, skipped + first_row, max_velocity, solve_s * 1e3);
  }

  double currentStep(std::chrono::steady_clock::time_point now) const {
    const double elapsed = std::chrono::duration<double>(now - epoch_).count();
    return epoch_step_ + elapsed * action_rate_hz_ * playback_speed_;
  }

  void servoLoop() {
    const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(servo_period_s_));
    const auto status_every = static_cast<std::size_t>(
        std::max(1.0, std::round(servo_rate_hz_ / status_rate_hz_)));
    auto next = std::chrono::steady_clock::now();
    std::size_t tick = 0;
    std::int64_t last_gripper_step = std::numeric_limits<std::int64_t>::min();
    std::size_t overruns = 0;
    while (running_ && rclcpp::ok()) {
      next += period;
      std::this_thread::sleep_until(next);
      const auto now = std::chrono::steady_clock::now();
      if (now - next > period) {
        ++overruns;
      }
      Joints left_measured{};
      Joints right_measured{};
      double joints_age = 0.0;
      {
        std::lock_guard<std::mutex> lock(joints_mutex_);
        left_measured = left_measured_;
        right_measured = right_measured_;
        joints_age = std::chrono::duration<double>(
                         now - std::min(left_measured_time_, right_measured_time_)).count();
      }

      JointState left_message;
      JointState right_message;
      std::string status;
      bool publish_status = false;
      std::optional<std::pair<double, double>> grippers;
      {
        std::lock_guard<std::mutex> lock(servo_mutex_);
        Joints left_target = left_tracker_.position();
        Joints right_target = right_tracker_.position();
        Joints left_velocity{};
        Joints right_velocity{};
        Joints left_acceleration{};
        Joints right_acceleration{};
        double step = 0.0;
        bool holding = true;
        const bool active = started_ && !fault_.load();
        if (joints_age > joint_state_timeout_s_ && started_ && !fault_.load()) {
          fault_.store(true);
          fault_reason_ = "measured joint states stale";
          RCLCPP_ERROR(get_logger(), "FAULT: %s (%.3f s); holding", fault_reason_.c_str(), joints_age);
        }
        if (active) {
          step = currentStep(now);
          const auto sample = timeline_->sample(step);
          if (sample.has_value()) {
            left_target = sample->left.position;
            right_target = sample->right.position;
            holding = sample->holding;
            const double speed2 = playback_speed_ * playback_speed_;
            for (std::size_t j = 0; j < kJoints; ++j) {
              left_velocity[j] = sample->left.velocity[j] * playback_speed_;
              right_velocity[j] = sample->right.velocity[j] * playback_speed_;
              left_acceleration[j] = sample->left.acceleration[j] * speed2;
              right_acceleration[j] = sample->right.acceleration[j] * speed2;
            }
            const auto gripper_step = static_cast<std::int64_t>(std::floor(step));
            if (gripper_step != last_gripper_step) {
              last_gripper_step = gripper_step;
              grippers = std::make_pair(sample->left_gripper, sample->right_gripper);
            }
          }
          // Tracking guard: the impedance controller must stay close to the
          // commanded target; a large error means a collision, a fault or a
          // controller that is not following this node.
          double error = 0.0;
          for (std::size_t j = 0; j < kJoints; ++j) {
            error = std::max(error, std::abs(left_measured[j] - left_tracker_.position()[j]));
            error = std::max(error, std::abs(right_measured[j] - right_tracker_.position()[j]));
          }
          tracking_error_ = error;
          if (error > max_tracking_error_rad_) {
            fault_.store(true);
            fault_reason_ = "tracking error above limit";
            RCLCPP_ERROR(get_logger(), "FAULT: %s (%.3f rad); holding", fault_reason_.c_str(), error);
          }
        } else if (!started_) {
          // Idle: follow the measured joints for a bounded time so an
          // activated impedance controller holds the arm where it is, then
          // latch.  Following forever under an active controller drifts.
          if (idle_follow_start_.time_since_epoch().count() == 0) {
            idle_follow_start_ = now;
          }
          const double idle_s = std::chrono::duration<double>(now - idle_follow_start_).count();
          if (!idle_latched_ && idle_s > idle_follow_timeout_s_) {
            idle_latched_ = true;
            RCLCPP_WARN(
                get_logger(), "Idle follow timed out after %.1f s; target latched at the measured pose",
                idle_s);
          }
          if (!idle_latched_) {
            left_target = left_measured;
            right_target = right_measured;
          }
        }
        if (fault_.load()) {
          left_target = left_tracker_.position();
          right_target = right_tracker_.position();
          left_velocity.fill(0.0);
          right_velocity.fill(0.0);
          left_acceleration.fill(0.0);
          right_acceleration.fill(0.0);
          grippers.reset();
        }
        if (!left_tracker_.step(left_target, left_velocity, left_acceleration) ||
            !right_tracker_.step(right_target, right_velocity, right_acceleration)) {
          if (!fault_.load()) {
            fault_.store(true);
            fault_reason_ = "Ruckig rejected the target";
            RCLCPP_ERROR(get_logger(), "FAULT: %s; holding", fault_reason_.c_str());
          }
        }
        const auto stamp = this->now();
        left_message.header.stamp = stamp;
        left_message.header.frame_id = left_arm_base_link_;
        left_message.name = left_joint_names_;
        left_message.position.assign(left_tracker_.position().begin(), left_tracker_.position().end());
        left_message.velocity.assign(left_tracker_.velocity().begin(), left_tracker_.velocity().end());
        right_message.header.stamp = stamp;
        right_message.header.frame_id = right_arm_base_link_;
        right_message.name = right_joint_names_;
        right_message.position.assign(right_tracker_.position().begin(), right_tracker_.position().end());
        right_message.velocity.assign(right_tracker_.velocity().begin(), right_tracker_.velocity().end());
        if (tick % status_every == 0U) {
          publish_status = true;
          std::ostringstream stream;
          stream.precision(6);
          stream << "{\"schema\":\"franka_duo_joint_servo_status_v1\","
                 << "\"started\":" << (started_ ? "true" : "false") << ","
                 << "\"fault\":" << (fault_.load() ? "true" : "false") << ","
                 << "\"fault_reason\":\"" << fault_reason_ << "\","
                 << "\"step\":" << step << ","
                 << "\"holding\":" << (holding ? "true" : "false") << ","
                 << "\"idle_latched\":" << (idle_latched_ ? "true" : "false") << ","
                 << "\"last_step\":"
                 << (timeline_->last_step().has_value() ? *timeline_->last_step() : -1) << ","
                 << "\"playback_speed\":" << playback_speed_ << ","
                 << "\"action_rate_hz\":" << action_rate_hz_ << ","
                 << "\"effective_action_rate_hz\":" << action_rate_hz_ * playback_speed_ << ","
                 << "\"commit_lead_steps\":" << commit_lead_steps_ << ","
                 << "\"blend_steps\":" << blend_steps_ << ","
                 << "\"blend_mode\":\"quintic_hold_v1\","
                 << "\"last_chunk_start_step\":" << last_chunk_start_step_ << ","
                 << "\"tracking_error_rad\":" << tracking_error_ << ","
                 << "\"joints_age_s\":" << joints_age << ","
                 << "\"chunks\":" << chunk_count_ << ","
                 << "\"servo_overruns\":" << overruns << "}";
          status = stream.str();
        }
      }
      left_target_publisher_->publish(left_message);
      right_target_publisher_->publish(right_message);
      if (grippers.has_value() && enable_gripper_) {
        std_msgs::msg::Float32 left_gripper;
        std_msgs::msg::Float32 right_gripper;
        left_gripper.data = static_cast<float>(grippers->first);
        right_gripper.data = static_cast<float>(grippers->second);
        left_gripper_publisher_->publish(left_gripper);
        right_gripper_publisher_->publish(right_gripper);
      }
      if (publish_status) {
        std_msgs::msg::String message;
        message.data = status;
        status_publisher_->publish(message);
      }
      ++tick;
    }
  }

  std::string chunk_topic_, status_topic_, left_joint_topic_, right_joint_topic_;
  std::string left_target_topic_, right_target_topic_, left_gripper_topic_, right_gripper_topic_;
  std::string left_group_name_, right_group_name_, left_arm_base_link_, right_arm_base_link_;
  std::string left_tip_link_, right_tip_link_, action_frame_;
  double tool_offset_z_m_{0.174};
  double ik_timeout_s_{0.02};
  double max_joint_delta_rad_{0.35};
  double action_rate_hz_{30.0};
  double playback_speed_{0.1};
  double servo_rate_hz_{1000.0};
  double servo_period_s_{0.001};
  double status_rate_hz_{30.0};
  int commit_lead_steps_{3};
  int blend_steps_{4};
  Joints max_velocity_{};
  double max_tracking_error_rad_{0.15};
  double joint_state_timeout_s_{0.2};
  double max_chunk_age_s_{0.3};
  double wait_timeout_s_{10.0};
  double idle_follow_timeout_s_{20.0};
  bool idle_latched_{false};
  std::chrono::steady_clock::time_point idle_follow_start_{};
  bool enable_gripper_{false};

  std::mutex joints_mutex_;
  Joints left_measured_{};
  Joints right_measured_{};
  std::chrono::steady_clock::time_point left_measured_time_{};
  std::chrono::steady_clock::time_point right_measured_time_{};
  bool have_left_{false};
  bool have_right_{false};

  std::mutex servo_mutex_;
  std::unique_ptr<JointTimeline> timeline_;
  JointRuckigTracker left_tracker_;
  JointRuckigTracker right_tracker_;
  bool started_{false};
  std::atomic<bool> fault_{false};
  std::string fault_reason_;
  double tracking_error_{0.0};
  std::chrono::steady_clock::time_point epoch_{};
  double epoch_step_{0.0};
  std::size_t chunk_count_{0};
  std::int64_t last_chunk_start_step_{-1};

  std::atomic<bool> running_{false};
  std::thread servo_thread_;

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
  rclcpp::Publisher<JointState>::SharedPtr left_target_publisher_;
  rclcpp::Publisher<JointState>::SharedPtr right_target_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr left_gripper_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr right_gripper_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
};

}  // namespace franka_duo_joint_servo

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    auto node = std::make_shared<franka_duo_joint_servo::PolicyChunkJointServo>();
    node->initialize();
    rclcpp::spin(node);
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("franka_duo_joint_servo"), "%s", exception.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
