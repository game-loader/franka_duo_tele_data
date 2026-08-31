#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/duration.hpp>
#include <Eigen/Geometry>

#include <franka_msgs/action/ptp_motion.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace {

using namespace std::chrono_literals;
using PTPMotion = franka_msgs::action::PTPMotion;
using JointState = sensor_msgs::msg::JointState;
using JointTrajectory = trajectory_msgs::msg::JointTrajectory;

constexpr std::size_t kArmJointCount = 7;

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

bool isFinite(double value) {
  return std::isfinite(value);
}

Eigen::Matrix3d rotationFromRot6d(const std::vector<double>& values) {
  if (values.size() != 6) {
    throw std::invalid_argument("rot6d must contain exactly six values");
  }
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
  Eigen::Vector3d third = first.cross(second);
  if (!third.allFinite() || third.norm() < 1e-8) {
    throw std::invalid_argument("rot6d produced a degenerate third row");
  }
  third.normalize();

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

Eigen::Isometry3d poseFromAction(
    const std::vector<double>& xyz,
    const std::vector<double>& rot6d) {
  if (xyz.size() != 3) {
    throw std::invalid_argument("target xyz must contain exactly three values");
  }
  Eigen::Vector3d translation(xyz[0], xyz[1], xyz[2]);
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
    const JointState& message,
    const std::string& expected_name,
    std::size_t joint_index) {
  for (std::size_t index = 0; index < message.name.size() &&
                              index < message.position.size(); ++index) {
    if (message.name[index] == expected_name && isFinite(message.position[index])) {
      return message.position[index];
    }
  }

  const std::string suffix = "_joint" + std::to_string(joint_index + 1);
  for (std::size_t index = 0; index < message.name.size() &&
                              index < message.position.size(); ++index) {
    const auto& name = message.name[index];
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0 &&
        name.find("finger") == std::string::npos &&
        name.find("gripper") == std::string::npos &&
        isFinite(message.position[index])) {
      return message.position[index];
    }
  }
  return std::nullopt;
}

struct ArmSolution {
  std::string label;
  std::string action_name;
  std::vector<double> positions;
  double largest_joint_delta{0.0};
};

struct DualSolution {
  ArmSolution left;
  ArmSolution right;
};

struct JointSample {
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> accelerations;
};

struct DualSample {
  JointSample left;
  JointSample right;
};

using SteadyTimePoint = std::chrono::steady_clock::time_point;

struct FrameResult {
  SteadyTimePoint send_start;
  std::optional<SteadyTimePoint> left_goal_accepted;
  std::optional<SteadyTimePoint> right_goal_accepted;
  std::optional<SteadyTimePoint> left_result_received;
  std::optional<SteadyTimePoint> right_result_received;
  double wall_send_start_unix_s{0.0};
  int left_result_code{-1};
  int right_result_code{-1};
  unsigned int left_target_status{0};
  unsigned int right_target_status{0};
  std::string left_error;
  std::string right_error;
  bool left_ok{false};
  bool right_ok{false};
};

double elapsedSeconds(
    const SteadyTimePoint& start,
    const std::optional<SteadyTimePoint>& end) {
  if (!end.has_value()) {
    return -1.0;
  }
  return std::chrono::duration<double>(end.value() - start).count();
}

std::string csvEscape(const std::string& value) {
  std::string escaped = "\"";
  for (const char character : value) {
    if (character == '"') {
      escaped += "\"\"";
    } else if (character == '\n' || character == '\r') {
      escaped += ' ';
    } else {
      escaped += character;
    }
  }
  escaped += '"';
  return escaped;
}

}  // namespace

class DuoPTPEpisode final : public rclcpp::Node {
 public:
  DuoPTPEpisode() : Node("franka_duo_episode_ptp") {
    declare_parameter<std::string>(
        "left_joint_topic",
        "/left/franka_robot_state_broadcaster/measured_joint_states");
    declare_parameter<std::string>(
        "right_joint_topic",
        "/right/franka_robot_state_broadcaster/measured_joint_states");
    declare_parameter<std::string>("left_action_name", "/left/action_server/ptp_motion");
    declare_parameter<std::string>("right_action_name", "/right/action_server/ptp_motion");
    declare_parameter<std::string>("left_group_name", "left_arm");
    declare_parameter<std::string>("right_group_name", "right_arm");
    declare_parameter<std::string>("left_arm_base_link", "left_fr3v2_link0");
    declare_parameter<std::string>("right_arm_base_link", "right_fr3v2_link0");
    declare_parameter<std::string>("left_tip_link", "left_fr3v2_link8");
    declare_parameter<std::string>("right_tip_link", "right_fr3v2_link8");
    declare_parameter<std::vector<double>>("action_trajectory", {});
    declare_parameter<double>("tool_offset_z_m", 0.174);
    declare_parameter<double>("ik_timeout_s", 0.1);
    declare_parameter<double>("max_joint_delta_rad", 1.0);
    declare_parameter<double>("max_joint_velocity", 0.2);
    declare_parameter<double>("goal_tolerance", 0.01);
    declare_parameter<double>("wait_timeout_s", 10.0);
    declare_parameter<std::string>("log_file", "/tmp/franka_duo_ptp_episode.csv");
    declare_parameter<std::string>("mode", "ptp");
    declare_parameter<std::string>(
        "left_trajectory_topic",
        "/franka_duo/eval/left/joint_trajectory");
    declare_parameter<std::string>(
        "right_trajectory_topic",
        "/franka_duo/eval/right/joint_trajectory");
    declare_parameter<double>("input_action_rate_hz", 15.0);
    declare_parameter<double>("stream_rate_hz", 50.0);
    declare_parameter<int>("chunk_size", 16);
    declare_parameter<double>("chunk_publish_period_s", 0.1);
    declare_parameter<double>("jtc_initial_delay_s", 1.0);
    declare_parameter<std::string>(
        "stream_log_file",
        "/tmp/franka_duo_jtc_chunk_stream.csv");
    declare_parameter<bool>("execute", false);
    declare_parameter<bool>("confirm", false);

    left_joint_topic_ = get_parameter("left_joint_topic").as_string();
    right_joint_topic_ = get_parameter("right_joint_topic").as_string();
    left_action_name_ = get_parameter("left_action_name").as_string();
    right_action_name_ = get_parameter("right_action_name").as_string();
    left_group_name_ = get_parameter("left_group_name").as_string();
    right_group_name_ = get_parameter("right_group_name").as_string();
    left_arm_base_link_ = get_parameter("left_arm_base_link").as_string();
    right_arm_base_link_ = get_parameter("right_arm_base_link").as_string();
    left_tip_link_ = get_parameter("left_tip_link").as_string();
    right_tip_link_ = get_parameter("right_tip_link").as_string();
    action_trajectory_ = get_parameter("action_trajectory").as_double_array();
    tool_offset_z_m_ = get_parameter("tool_offset_z_m").as_double();
    ik_timeout_s_ = get_parameter("ik_timeout_s").as_double();
    max_joint_delta_rad_ = get_parameter("max_joint_delta_rad").as_double();
    max_joint_velocity_ = get_parameter("max_joint_velocity").as_double();
    goal_tolerance_ = get_parameter("goal_tolerance").as_double();
    wait_timeout_s_ = get_parameter("wait_timeout_s").as_double();
    log_file_ = get_parameter("log_file").as_string();
    mode_ = get_parameter("mode").as_string();
    left_trajectory_topic_ = get_parameter("left_trajectory_topic").as_string();
    right_trajectory_topic_ = get_parameter("right_trajectory_topic").as_string();
    input_action_rate_hz_ = get_parameter("input_action_rate_hz").as_double();
    stream_rate_hz_ = get_parameter("stream_rate_hz").as_double();
    chunk_size_ = get_parameter("chunk_size").as_int();
    chunk_publish_period_s_ = get_parameter("chunk_publish_period_s").as_double();
    jtc_initial_delay_s_ = get_parameter("jtc_initial_delay_s").as_double();
    stream_log_file_ = get_parameter("stream_log_file").as_string();
    execute_ = get_parameter("execute").as_bool();
    confirm_ = get_parameter("confirm").as_bool();

    if (action_trajectory_.empty() || action_trajectory_.size() % 18U != 0U) {
      throw std::invalid_argument("action_trajectory must contain one or more 18D actions");
    }
    if (!isFinite(tool_offset_z_m_) || !isFinite(ik_timeout_s_) || ik_timeout_s_ <= 0.0 ||
        !isFinite(max_joint_delta_rad_) || max_joint_delta_rad_ <= 0.0 ||
        !isFinite(max_joint_velocity_) || max_joint_velocity_ <= 0.0 ||
        !isFinite(goal_tolerance_) || goal_tolerance_ <= 0.0 ||
        !isFinite(wait_timeout_s_) || wait_timeout_s_ <= 0.0) {
      throw std::invalid_argument("PTP parameters must be finite and positive");
    }
    if (mode_ != "ptp" && mode_ != "jtc") {
      throw std::invalid_argument("mode must be either ptp or jtc");
    }
    if (!isFinite(input_action_rate_hz_) || input_action_rate_hz_ <= 0.0 ||
        !isFinite(stream_rate_hz_) || stream_rate_hz_ <= 0.0 ||
        chunk_size_ <= 0 ||
        !isFinite(chunk_publish_period_s_) || chunk_publish_period_s_ <= 0.0 ||
        !isFinite(jtc_initial_delay_s_) || jtc_initial_delay_s_ < 0.0) {
      throw std::invalid_argument("JTC streaming parameters must be positive");
    }
    if (mode_ == "jtc" && stream_log_file_.empty()) {
      throw std::invalid_argument("stream_log_file must be set in jtc mode");
    }
    if (execute_ && confirm_ && log_file_.empty()) {
      throw std::invalid_argument("log_file must be set for real execution");
    }

    left_joint_subscription_ = create_subscription<JointState>(
        left_joint_topic_,
        rclcpp::SensorDataQoS(),
        [this](const JointState::SharedPtr message) { left_joints_ = message; });
    right_joint_subscription_ = create_subscription<JointState>(
        right_joint_topic_,
        rclcpp::SensorDataQoS(),
        [this](const JointState::SharedPtr message) { right_joints_ = message; });
  }

  int run() {
    waitForJointSamples();

    robot_model_loader::RobotModelLoader model_loader(shared_from_this());
    const auto& robot_model = model_loader.getModel();
    if (robot_model == nullptr) {
      throw std::runtime_error("MoveIt robot model is unavailable");
    }

    const auto* left_group = robot_model->getJointModelGroup(left_group_name_);
    const auto* right_group = robot_model->getJointModelGroup(right_group_name_);
    if (left_group == nullptr || right_group == nullptr) {
      throw std::runtime_error("MoveIt left_arm/right_arm group is unavailable");
    }
    if (left_group->getVariableCount() != kArmJointCount ||
        right_group->getVariableCount() != kArmJointCount) {
      throw std::runtime_error("Both MoveIt arm groups must contain seven variables");
    }
    left_joint_names_ = left_group->getVariableNames();
    right_joint_names_ = right_group->getVariableNames();
    RCLCPP_INFO(
        get_logger(),
        "MoveIt groups: left=%s solver=%s, right=%s solver=%s",
        left_group_name_.c_str(),
        left_group->getSolverInstance() == nullptr ? "missing" : "available",
        right_group_name_.c_str(),
        right_group->getSolverInstance() == nullptr ? "missing" : "available");
    if (left_group->getSolverInstance() == nullptr || right_group->getSolverInstance() == nullptr) {
      throw std::runtime_error("MoveIt KDL solver is unavailable for one arm");
    }
    RCLCPP_INFO(
        get_logger(),
        "IK frames: left base=%s tip=%s; right base=%s tip=%s",
        left_group->getSolverInstance()->getBaseFrame().c_str(),
        left_group->getSolverInstance()->getTipFrame().c_str(),
        right_group->getSolverInstance()->getBaseFrame().c_str(),
        right_group->getSolverInstance()->getTipFrame().c_str());

    moveit::core::RobotState state(robot_model);
    state.setToDefaultValues();
    seedGroup(state, left_group, *left_joints_);
    seedGroup(state, right_group, *right_joints_);
    state.update();

    const auto left_transform = transformFromRowMajor(kLeftMidpointFromArmBase);
    const auto right_transform = transformFromRowMajor(kRightMidpointFromArmBase);
    const std::size_t action_count = action_trajectory_.size() / 18U;
    std::vector<DualSolution> solutions;
    solutions.reserve(action_count);
    for (std::size_t action_index = 0; action_index < action_count; ++action_index) {
      const auto begin = action_trajectory_.begin() + static_cast<std::ptrdiff_t>(action_index * 18U);
      const std::vector<double> action(begin, begin + 18);
      const std::vector<double> left_xyz(action.begin(), action.begin() + 3);
      const std::vector<double> left_rot6d(action.begin() + 3, action.begin() + 9);
      const std::vector<double> right_xyz(action.begin() + 9, action.begin() + 12);
      const std::vector<double> right_rot6d(action.begin() + 12, action.end());
      auto left_solution = solveArm(
          state,
          left_group,
          left_group_name_,
          left_arm_base_link_,
          left_tip_link_,
          left_action_name_,
          left_transform,
          left_xyz,
          left_rot6d);
      auto right_solution = solveArm(
          state,
          right_group,
          right_group_name_,
          right_arm_base_link_,
          right_tip_link_,
          right_action_name_,
          right_transform,
          right_xyz,
          right_rot6d);
      RCLCPP_INFO(
          get_logger(),
          "Prepared action %zu/%zu: left_delta=%.4f right_delta=%.4f",
          action_index + 1,
          action_count,
          left_solution.largest_joint_delta,
          right_solution.largest_joint_delta);
      solutions.push_back(DualSolution{std::move(left_solution), std::move(right_solution)});
    }

    if (mode_ == "jtc") {
      return streamJointTrajectory(solutions);
    }
    if (!execute_ || !confirm_) {
      RCLCPP_WARN(
          get_logger(),
          "Dry-run only. Use execute:=true confirm:=true to send the PTP trajectory");
      return 0;
    }
    return sendTrajectory(solutions);
  }

 private:
  void waitForJointSamples() {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(wait_timeout_s_);
    rclcpp::WallRate rate(100.0);
    while (rclcpp::ok() &&
           (left_joints_ == nullptr || right_joints_ == nullptr) &&
           std::chrono::steady_clock::now() < deadline) {
      rclcpp::spin_some(shared_from_this());
      rate.sleep();
    }
    if (left_joints_ == nullptr) {
      throw std::runtime_error("Timed out waiting for " + left_joint_topic_);
    }
    if (right_joints_ == nullptr) {
      throw std::runtime_error("Timed out waiting for " + right_joint_topic_);
    }
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

  ArmSolution solveArm(
      moveit::core::RobotState& state,
      const moveit::core::JointModelGroup* group,
      const std::string& group_name,
      const std::string& arm_base_link,
      const std::string& tip_link,
      const std::string& action_name,
      const Eigen::Isometry3d& midpoint_from_arm_base,
      const std::vector<double>& target_xyz,
      const std::vector<double>& target_rot6d) const {
    const auto* arm_base = state.getRobotModel()->getLinkModel(arm_base_link);
    const auto* tip = state.getRobotModel()->getLinkModel(tip_link);
    if (arm_base == nullptr || tip == nullptr) {
      throw std::runtime_error("MoveIt link not found for " + group_name);
    }

    std::vector<double> seed_positions;
    state.copyJointGroupPositions(group, seed_positions);
    const Eigen::Isometry3d target_ee_in_midpoint =
        poseFromAction(target_xyz, target_rot6d);
    const Eigen::Isometry3d target_ee_in_arm_base =
        midpoint_from_arm_base.inverse() * target_ee_in_midpoint;
    const Eigen::Isometry3d target_ee =
        state.getGlobalLinkTransform(arm_base) * target_ee_in_arm_base;

    Eigen::Isometry3d tip_to_tool = Eigen::Isometry3d::Identity();
    tip_to_tool.translation().z() = tool_offset_z_m_;
    const Eigen::Isometry3d target_tip = target_ee * tip_to_tool.inverse();
    const Eigen::Isometry3d current_tip = state.getGlobalLinkTransform(tip);

    RCLCPP_INFO(
        get_logger(),
        "%s target arm-base=(%.6f, %.6f, %.6f) model-tip=(%.6f, %.6f, %.6f) "
        "current-tip=(%.6f, %.6f, %.6f)",
        group_name.c_str(),
        target_ee_in_arm_base.translation().x(),
        target_ee_in_arm_base.translation().y(),
        target_ee_in_arm_base.translation().z(),
        target_tip.translation().x(),
        target_tip.translation().y(),
        target_tip.translation().z(),
        current_tip.translation().x(),
        current_tip.translation().y(),
        current_tip.translation().z());

    const std::vector<double> consistency_limits(kArmJointCount, max_joint_delta_rad_);
    if (!state.setFromIK(
            group,
            target_tip,
            tip_link,
            consistency_limits,
            ik_timeout_s_)) {
      throw std::runtime_error("KDL IK failed for " + group_name);
    }
    state.update();
    if (!state.satisfiesBounds(group)) {
      throw std::runtime_error("IK solution is outside MoveIt bounds for " + group_name);
    }

    std::vector<double> solution;
    state.copyJointGroupPositions(group, solution);
    if (solution.size() != kArmJointCount || seed_positions.size() != kArmJointCount) {
      throw std::runtime_error("IK solution does not contain seven joints for " + group_name);
    }
    double largest_joint_delta = 0.0;
    for (std::size_t index = 0; index < solution.size(); ++index) {
      largest_joint_delta =
          std::max(largest_joint_delta, std::abs(solution[index] - seed_positions[index]));
    }
    if (largest_joint_delta > max_joint_delta_rad_) {
      throw std::runtime_error("IK solution is too far from measured seed for " + group_name);
    }

    RCLCPP_INFO(
        get_logger(),
        "%s IK succeeded; action=%s tool_offset_z=%.4f max_joint_delta=%.4f",
        group_name.c_str(),
        action_name.c_str(),
        tool_offset_z_m_,
        largest_joint_delta);
    RCLCPP_INFO(
        get_logger(),
        "%s q=[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]",
        group_name.c_str(),
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
        solution[6]);

    return ArmSolution{group_name, action_name, std::move(solution), largest_joint_delta};
  }

  static void fillDerivatives(std::vector<JointSample>& samples, double dt) {
    if (samples.empty()) {
      return;
    }
    const auto sample_count = samples.size();
    const auto joint_count = samples.front().positions.size();
    for (auto& sample : samples) {
      sample.velocities.assign(joint_count, 0.0);
      sample.accelerations.assign(joint_count, 0.0);
    }
    if (sample_count == 1U) {
      return;
    }

    const double inverse_dt = 1.0 / dt;
    const double inverse_dt_squared = inverse_dt * inverse_dt;
    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
      for (std::size_t joint_index = 0; joint_index < joint_count; ++joint_index) {
        if (sample_index == 0U) {
          samples[sample_index].velocities[joint_index] =
              (samples[1].positions[joint_index] - samples[0].positions[joint_index]) *
              inverse_dt;
        } else if (sample_index + 1U == sample_count) {
          samples[sample_index].velocities[joint_index] =
              (samples[sample_index].positions[joint_index] -
               samples[sample_index - 1U].positions[joint_index]) *
              inverse_dt;
        } else {
          samples[sample_index].velocities[joint_index] =
              (samples[sample_index + 1U].positions[joint_index] -
               samples[sample_index - 1U].positions[joint_index]) *
              0.5 * inverse_dt;
        }

        if (sample_count >= 3U) {
          if (sample_index == 0U) {
            samples[sample_index].accelerations[joint_index] =
                (samples[2].positions[joint_index] -
                 2.0 * samples[1].positions[joint_index] +
                 samples[0].positions[joint_index]) *
                inverse_dt_squared;
          } else if (sample_index + 1U == sample_count) {
            samples[sample_index].accelerations[joint_index] =
                (samples[sample_index].positions[joint_index] -
                 2.0 * samples[sample_index - 1U].positions[joint_index] +
                 samples[sample_index - 2U].positions[joint_index]) *
                inverse_dt_squared;
          } else {
            samples[sample_index].accelerations[joint_index] =
                (samples[sample_index + 1U].positions[joint_index] -
                 2.0 * samples[sample_index].positions[joint_index] +
                 samples[sample_index - 1U].positions[joint_index]) *
                inverse_dt_squared;
          }
        }
      }
    }

    // Each chunk is connected to the measured-state anchor and to the next
    // chunk at rest. Keep both explicit boundaries stationary so JTC's
    // quintic interpolation cannot inherit a finite-difference edge slope.
    std::fill(samples.front().velocities.begin(), samples.front().velocities.end(), 0.0);
    std::fill(
        samples.front().accelerations.begin(),
        samples.front().accelerations.end(),
        0.0);
    std::fill(samples.back().velocities.begin(), samples.back().velocities.end(), 0.0);
    std::fill(
        samples.back().accelerations.begin(),
        samples.back().accelerations.end(),
        0.0);
  }

  std::vector<DualSample> resampleForJTC(
      const std::vector<DualSolution>& solutions) const {
    if (solutions.empty()) {
      return {};
    }
    const double source_dt = 1.0 / input_action_rate_hz_;
    const double target_dt = 1.0 / stream_rate_hz_;
    const double source_duration =
        static_cast<double>(solutions.size() - 1U) * source_dt;
    const std::size_t sample_count =
        std::max<std::size_t>(
            1U,
            static_cast<std::size_t>(std::ceil(source_duration * stream_rate_hz_)) + 1U);

    std::vector<DualSample> samples;
    samples.reserve(sample_count);
    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
      const double sample_time = std::min(
          source_duration,
          static_cast<double>(sample_index) * target_dt);
      const double source_position = sample_time / source_dt;
      const auto lower_index = std::min<std::size_t>(
          solutions.size() - 1U,
          static_cast<std::size_t>(std::floor(source_position)));
      const auto upper_index = std::min<std::size_t>(
          solutions.size() - 1U,
          lower_index + 1U);
      const double alpha = upper_index == lower_index
                               ? 0.0
                               : source_position - static_cast<double>(lower_index);

      DualSample sample;
      sample.left.positions.resize(kArmJointCount);
      sample.right.positions.resize(kArmJointCount);
      for (std::size_t joint_index = 0; joint_index < kArmJointCount; ++joint_index) {
        sample.left.positions[joint_index] =
            (1.0 - alpha) * solutions[lower_index].left.positions[joint_index] +
            alpha * solutions[upper_index].left.positions[joint_index];
        sample.right.positions[joint_index] =
            (1.0 - alpha) * solutions[lower_index].right.positions[joint_index] +
            alpha * solutions[upper_index].right.positions[joint_index];
      }
      samples.push_back(std::move(sample));
    }

    std::vector<JointSample> left_samples;
    std::vector<JointSample> right_samples;
    left_samples.reserve(samples.size());
    right_samples.reserve(samples.size());
    for (const auto& sample : samples) {
      left_samples.push_back(sample.left);
      right_samples.push_back(sample.right);
    }
    fillDerivatives(left_samples, target_dt);
    fillDerivatives(right_samples, target_dt);
    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
      samples[sample_index].left = std::move(left_samples[sample_index]);
      samples[sample_index].right = std::move(right_samples[sample_index]);
    }
    return samples;
  }

  static builtin_interfaces::msg::Duration durationMessage(double seconds) {
    const auto total_nanoseconds = static_cast<std::int64_t>(
        std::llround(seconds * 1'000'000'000.0));
    builtin_interfaces::msg::Duration result;
    result.sec = static_cast<std::int32_t>(total_nanoseconds / 1'000'000'000LL);
    result.nanosec = static_cast<std::uint32_t>(
        total_nanoseconds % 1'000'000'000LL);
    return result;
  }

  static JointTrajectory makeTrajectory(
      const std::vector<DualSample>& samples,
      const std::vector<std::string>& joint_names,
      bool left,
      double sample_period_s,
      double initial_delay_s) {
    JointTrajectory trajectory;
    trajectory.joint_names = joint_names;
    trajectory.points.reserve(samples.size());
    for (std::size_t point_index = 0; point_index < samples.size(); ++point_index) {
      const auto& sample = samples[point_index];
      const auto& joint_sample = left ? sample.left : sample.right;
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions = joint_sample.positions;
      point.velocities = joint_sample.velocities;
      point.accelerations = joint_sample.accelerations;
      const double time_from_start =
          point_index == 0U
              ? 0.0
              : initial_delay_s +
                    static_cast<double>(point_index - 1U) * sample_period_s;
      point.time_from_start = durationMessage(time_from_start);
      trajectory.points.push_back(std::move(point));
    }
    return trajectory;
  }

  static JointSample measuredSample(
      const JointState& measured,
      const std::vector<std::string>& joint_names) {
    JointSample sample;
    sample.positions.reserve(joint_names.size());
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

  int streamJointTrajectory(const std::vector<DualSolution>& solutions) {
    const double sample_period_s = 1.0 / stream_rate_hz_;
    const std::size_t source_chunk_size = static_cast<std::size_t>(chunk_size_);
    const std::size_t chunk_count =
        (solutions.size() + source_chunk_size - 1U) / source_chunk_size;
    RCLCPP_INFO(
        get_logger(),
        "JTC stream prepared: source_actions=%zu source_rate=%.3fHz "
        "source_chunk_size=%d target_rate=%.3fHz chunks=%zu initial_delay=%.3fs",
        solutions.size(),
        input_action_rate_hz_,
        chunk_size_,
        stream_rate_hz_,
        chunk_count,
        jtc_initial_delay_s_);

    if (!execute_ || !confirm_) {
      RCLCPP_WARN(
          get_logger(),
          "JTC dry-run only. Use mode:=jtc execute:=true confirm:=true "
          "after loading both joint trajectory controllers");
      return 0;
    }

    auto left_publisher =
        create_publisher<JointTrajectory>(left_trajectory_topic_, rclcpp::QoS(10));
    auto right_publisher =
        create_publisher<JointTrajectory>(right_trajectory_topic_, rclcpp::QoS(10));
    const auto publisher_deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration<double>(wait_timeout_s_);
    while (rclcpp::ok() &&
           (left_publisher->get_subscription_count() == 0U ||
            right_publisher->get_subscription_count() == 0U) &&
           std::chrono::steady_clock::now() < publisher_deadline) {
      rclcpp::spin_some(shared_from_this());
      std::this_thread::sleep_for(10ms);
    }
    if (left_publisher->get_subscription_count() == 0U ||
        right_publisher->get_subscription_count() == 0U) {
      throw std::runtime_error(
          "JTC trajectory topic has no subscriber on one or both arms");
    }

    std::ofstream log(stream_log_file_, std::ios::out | std::ios::trunc);
    if (!log.is_open()) {
      throw std::runtime_error("Could not open JTC stream log " + stream_log_file_);
    }
    log << "chunk_index,action_start,action_count,point_count,wall_publish_unix_s,"
           "publish_elapsed_s,trajectory_horizon_s,wait_elapsed_s,"
           "max_abs_velocity,max_abs_acceleration\n";
    log << std::setprecision(17);

    const auto stream_start = std::chrono::steady_clock::now();
    for (std::size_t chunk_index = 0U;
         rclcpp::ok() && chunk_index < chunk_count;
         ++chunk_index) {
      const std::size_t start_index = chunk_index * source_chunk_size;
      const std::size_t end_index =
          std::min(solutions.size(), start_index + source_chunk_size);
      const std::vector<DualSolution> source_chunk(
          solutions.begin() + static_cast<std::ptrdiff_t>(start_index),
          solutions.begin() + static_cast<std::ptrdiff_t>(end_index));
      const auto action_samples = resampleForJTC(source_chunk);
      if (action_samples.empty()) {
        throw std::runtime_error("JTC resampling produced no samples");
      }

      // Anchor every replacement at the current measured state.  The action
      // chunk remains one complete trajectory, but its first target is reached
      // through the bridge interval instead of appearing as a position step.
      rclcpp::spin_some(shared_from_this());
      if (left_joints_ == nullptr || right_joints_ == nullptr) {
        throw std::runtime_error("Measured joint state disappeared before JTC publish");
      }
      DualSample measured_sample;
      measured_sample.left = measuredSample(*left_joints_, left_joint_names_);
      measured_sample.right = measuredSample(*right_joints_, right_joint_names_);
      std::vector<DualSample> samples;
      samples.reserve(action_samples.size() + 1U);
      samples.push_back(std::move(measured_sample));
      samples.insert(samples.end(), action_samples.begin(), action_samples.end());

      double max_velocity = 0.0;
      double max_acceleration = 0.0;
      for (const auto& sample : samples) {
        for (const auto value : sample.left.velocities) {
          max_velocity = std::max(max_velocity, std::abs(value));
        }
        for (const auto value : sample.right.velocities) {
          max_velocity = std::max(max_velocity, std::abs(value));
        }
        for (const auto value : sample.left.accelerations) {
          max_acceleration = std::max(max_acceleration, std::abs(value));
        }
        for (const auto value : sample.right.accelerations) {
          max_acceleration = std::max(max_acceleration, std::abs(value));
        }
      }

      const double trajectory_horizon_s =
          jtc_initial_delay_s_ +
          static_cast<double>(action_samples.size() - 1U) * sample_period_s;
      const auto publish_time = std::chrono::steady_clock::now();
      auto left_trajectory = makeTrajectory(
          samples,
          left_joint_names_,
          true,
          sample_period_s,
          jtc_initial_delay_s_);
      auto right_trajectory = makeTrajectory(
          samples,
          right_joint_names_,
          false,
          sample_period_s,
          jtc_initial_delay_s_);
      left_publisher->publish(left_trajectory);
      right_publisher->publish(right_trajectory);

      const double publish_elapsed =
          std::chrono::duration<double>(publish_time - stream_start).count();
      const auto point_count = left_trajectory.points.size();
      const double wall_publish_unix_s =
          std::chrono::duration<double>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
      log << chunk_index << ','
          << start_index << ','
          << source_chunk.size() << ','
          << point_count << ','
          << wall_publish_unix_s << ','
          << publish_elapsed << ','
          << trajectory_horizon_s << ',';
      log.flush();
      const auto wait_start = std::chrono::steady_clock::now();
      std::this_thread::sleep_for(
          std::chrono::duration<double>(trajectory_horizon_s + chunk_publish_period_s_));
      const double wait_elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start)
              .count();
      log << wait_elapsed << ','
          << max_velocity << ','
          << max_acceleration << '\n';
      log.flush();
      RCLCPP_INFO(
          get_logger(),
          "Published JTC trajectory %zu/%zu: action_start=%zu action_count=%zu "
          "action_points=%zu points=%zu horizon=%.3fs max_abs_velocity=%.4frad/s "
          "max_abs_acceleration=%.4frad/s^2",
          chunk_index + 1U,
          chunk_count,
          start_index,
          source_chunk.size(),
          action_samples.size(),
          point_count,
          trajectory_horizon_s,
          max_velocity,
          max_acceleration);
    }

    RCLCPP_INFO(
        get_logger(),
        "JTC stream completed: trajectories=%zu log=%s",
        chunk_count,
        stream_log_file_.c_str());
    return 0;
  }

  FrameResult sendBothPTP(const ArmSolution& left, const ArmSolution& right) {
    using Client = rclcpp_action::Client<PTPMotion>;
    FrameResult frame;
    frame.send_start = std::chrono::steady_clock::now();
    frame.wall_send_start_unix_s =
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    auto left_client = rclcpp_action::create_client<PTPMotion>(
        shared_from_this(), left.action_name);
    auto right_client = rclcpp_action::create_client<PTPMotion>(
        shared_from_this(), right.action_name);
    if (!left_client->wait_for_action_server(
            std::chrono::duration<double>(wait_timeout_s_)) ||
        !right_client->wait_for_action_server(
            std::chrono::duration<double>(wait_timeout_s_))) {
      frame.left_error = "one or both PTP action servers are unavailable";
      frame.right_error = frame.left_error;
      return frame;
    }

    PTPMotion::Goal left_goal;
    left_goal.goal_joint_configuration = left.positions;
    left_goal.maximum_joint_velocities.assign(kArmJointCount, max_joint_velocity_);
    left_goal.goal_tolerance = goal_tolerance_;
    PTPMotion::Goal right_goal;
    right_goal.goal_joint_configuration = right.positions;
    right_goal.maximum_joint_velocities.assign(kArmJointCount, max_joint_velocity_);
    right_goal.goal_tolerance = goal_tolerance_;

    const auto left_goal_future = left_client->async_send_goal(left_goal);
    const auto right_goal_future = right_client->async_send_goal(right_goal);
    const auto goal_deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(wait_timeout_s_);
    while (rclcpp::ok() &&
           (!frame.left_goal_accepted.has_value() ||
            !frame.right_goal_accepted.has_value()) &&
           std::chrono::steady_clock::now() < goal_deadline) {
      rclcpp::spin_some(shared_from_this());
      const auto now = std::chrono::steady_clock::now();
      if (!frame.left_goal_accepted.has_value() &&
          left_goal_future.wait_for(0ms) == std::future_status::ready) {
        frame.left_goal_accepted = now;
      }
      if (!frame.right_goal_accepted.has_value() &&
          right_goal_future.wait_for(0ms) == std::future_status::ready) {
        frame.right_goal_accepted = now;
      }
      if (!frame.left_goal_accepted.has_value() ||
          !frame.right_goal_accepted.has_value()) {
        std::this_thread::sleep_for(1ms);
      }
    }
    if (!frame.left_goal_accepted.has_value()) {
      frame.left_error = "timed out waiting for left PTP goal response";
    }
    if (!frame.right_goal_accepted.has_value()) {
      frame.right_error = "timed out waiting for right PTP goal response";
    }
    if (!frame.left_goal_accepted.has_value() || !frame.right_goal_accepted.has_value()) {
      if (frame.left_goal_accepted.has_value()) {
        const auto left_handle = left_goal_future.get();
        if (left_handle != nullptr) {
          left_client->async_cancel_goal(left_handle);
        }
      }
      if (frame.right_goal_accepted.has_value()) {
        const auto right_handle = right_goal_future.get();
        if (right_handle != nullptr) {
          right_client->async_cancel_goal(right_handle);
        }
      }
      return frame;
    }

    const auto left_handle = left_goal_future.get();
    const auto right_handle = right_goal_future.get();
    if (left_handle == nullptr || right_handle == nullptr) {
      if (left_handle != nullptr) {
        left_client->async_cancel_goal(left_handle);
      }
      if (right_handle != nullptr) {
        right_client->async_cancel_goal(right_handle);
      }
      if (left_handle == nullptr) {
        frame.left_error = "left PTP goal rejected";
      }
      if (right_handle == nullptr) {
        frame.right_error = "right PTP goal rejected";
      }
      return frame;
    }

    RCLCPP_WARN(get_logger(), "Sending left and right PTP targets concurrently");
    const auto left_result_future = left_client->async_get_result(left_handle);
    const auto right_result_future = right_client->async_get_result(right_handle);
    const auto result_deadline = std::chrono::steady_clock::now() +
                                 std::chrono::duration<double>(wait_timeout_s_ * 10.0);
    while (rclcpp::ok() &&
           (!frame.left_result_received.has_value() ||
            !frame.right_result_received.has_value()) &&
           std::chrono::steady_clock::now() < result_deadline) {
      rclcpp::spin_some(shared_from_this());
      const auto now = std::chrono::steady_clock::now();
      if (!frame.left_result_received.has_value() &&
          left_result_future.wait_for(0ms) == std::future_status::ready) {
        frame.left_result_received = now;
      }
      if (!frame.right_result_received.has_value() &&
          right_result_future.wait_for(0ms) == std::future_status::ready) {
        frame.right_result_received = now;
      }
      if (!frame.left_result_received.has_value() ||
          !frame.right_result_received.has_value()) {
        std::this_thread::sleep_for(1ms);
      }
    }
    if (!frame.left_result_received.has_value()) {
      frame.left_error = "timed out waiting for left PTP result";
      left_client->async_cancel_goal(left_handle);
    }
    if (!frame.right_result_received.has_value()) {
      frame.right_error = "timed out waiting for right PTP result";
      right_client->async_cancel_goal(right_handle);
    }
    if (!frame.left_result_received.has_value() || !frame.right_result_received.has_value()) {
      return frame;
    }

    const auto left_result = left_result_future.get();
    const auto right_result = right_result_future.get();
    frame.left_result_code = static_cast<int>(left_result.code);
    frame.right_result_code = static_cast<int>(right_result.code);
    frame.left_target_status =
        left_result.result == nullptr ? 0U : left_result.result->target_status.status;
    frame.right_target_status =
        right_result.result == nullptr ? 0U : right_result.result->target_status.status;
    frame.left_ok =
        left_result.code == rclcpp_action::ResultCode::SUCCEEDED &&
        left_result.result != nullptr;
    frame.right_ok =
        right_result.code == rclcpp_action::ResultCode::SUCCEEDED &&
        right_result.result != nullptr;
    if (left_result.result != nullptr) {
      frame.left_error = left_result.result->error_message;
    }
    if (right_result.result != nullptr) {
      frame.right_error = right_result.result->error_message;
    }
    RCLCPP_INFO(
        get_logger(),
        "PTP results: left_code=%d left_status=%u right_code=%d right_status=%u",
        static_cast<int>(left_result.code),
        frame.left_target_status,
        static_cast<int>(right_result.code),
        frame.right_target_status);
    if (!frame.left_ok || !frame.right_ok) {
      if (!frame.left_ok && left_result.result != nullptr) {
        RCLCPP_ERROR(
            get_logger(), "Left PTP error: %s", left_result.result->error_message.c_str());
      }
      if (!frame.right_ok && right_result.result != nullptr) {
        RCLCPP_ERROR(
            get_logger(), "Right PTP error: %s", right_result.result->error_message.c_str());
      }
      return frame;
    }
    return frame;
  }

  int sendTrajectory(const std::vector<DualSolution>& solutions) {
    std::ofstream log(log_file_, std::ios::out | std::ios::trunc);
    if (!log.is_open()) {
      throw std::runtime_error("Could not open PTP execution log " + log_file_);
    }
    log << "action_index,wall_send_start_unix_s,send_elapsed_s,send_interval_s,"
           "left_goal_accepted_s,right_goal_accepted_s,left_result_s,right_result_s,"
           "frame_duration_s,left_result_code,left_target_status,right_result_code,"
           "right_target_status,left_joint_delta_rad,right_joint_delta_rad,left_ok,right_ok,"
           "left_error,right_error\n";
    log << std::setprecision(17);

    const auto trajectory_start = std::chrono::steady_clock::now();
    std::optional<SteadyTimePoint> previous_send_start;
    std::vector<double> send_intervals;
    std::vector<double> frame_durations;
    send_intervals.reserve(solutions.size() > 0 ? solutions.size() - 1 : 0);
    frame_durations.reserve(solutions.size());

    for (std::size_t action_index = 0; action_index < solutions.size(); ++action_index) {
      RCLCPP_WARN(
          get_logger(),
          "Executing PTP trajectory action %zu/%zu",
          action_index + 1,
          solutions.size());
      const auto frame = sendBothPTP(
          solutions[action_index].left,
          solutions[action_index].right);
      const double send_elapsed =
          std::chrono::duration<double>(frame.send_start - trajectory_start).count();
      const double send_interval =
          previous_send_start.has_value()
              ? std::chrono::duration<double>(
                    frame.send_start - previous_send_start.value())
                    .count()
              : -1.0;
      const double left_goal_accepted_s =
          elapsedSeconds(frame.send_start, frame.left_goal_accepted);
      const double right_goal_accepted_s =
          elapsedSeconds(frame.send_start, frame.right_goal_accepted);
      const double left_result_s =
          elapsedSeconds(frame.send_start, frame.left_result_received);
      const double right_result_s =
          elapsedSeconds(frame.send_start, frame.right_result_received);
      const double frame_duration_s =
          std::max(left_result_s, right_result_s);
      log << action_index << ','
          << frame.wall_send_start_unix_s << ','
          << send_elapsed << ','
          << send_interval << ','
          << left_goal_accepted_s << ','
          << right_goal_accepted_s << ','
          << left_result_s << ','
          << right_result_s << ','
          << frame_duration_s << ','
          << frame.left_result_code << ','
          << frame.left_target_status << ','
          << frame.right_result_code << ','
          << frame.right_target_status << ','
          << solutions[action_index].left.largest_joint_delta << ','
          << solutions[action_index].right.largest_joint_delta << ','
          << (frame.left_ok ? 1 : 0) << ','
          << (frame.right_ok ? 1 : 0) << ','
          << csvEscape(frame.left_error) << ','
          << csvEscape(frame.right_error) << '\n';
      log.flush();

      if (send_interval >= 0.0) {
        send_intervals.push_back(send_interval);
      }
      if (frame_duration_s >= 0.0) {
        frame_durations.push_back(frame_duration_s);
      }
      previous_send_start = frame.send_start;

      if (!frame.left_ok || !frame.right_ok) {
        RCLCPP_ERROR(get_logger(), "Stopping trajectory after action %zu", action_index + 1);
        return 1;
      }
    }

    if (!send_intervals.empty()) {
      auto sorted_intervals = send_intervals;
      std::sort(sorted_intervals.begin(), sorted_intervals.end());
      const double interval_sum =
          std::accumulate(send_intervals.begin(), send_intervals.end(), 0.0);
      const double average_interval = interval_sum / send_intervals.size();
      const double median_interval =
          sorted_intervals[sorted_intervals.size() / 2];
      RCLCPP_INFO(
          get_logger(),
          "PTP timing: frames=%zu total_send_span=%.3fs average_interval=%.3fs "
          "average_rate=%.3fHz median_interval=%.3fs min_interval=%.3fs max_interval=%.3fs",
          solutions.size(),
          std::chrono::duration<double>(
              previous_send_start.value() - trajectory_start)
              .count(),
          average_interval,
          1.0 / average_interval,
          median_interval,
          sorted_intervals.front(),
          sorted_intervals.back());
    }
    if (!frame_durations.empty()) {
      const double completion_sum =
          std::accumulate(frame_durations.begin(), frame_durations.end(), 0.0);
      RCLCPP_INFO(
          get_logger(),
          "PTP completion: average_frame_duration=%.3fs min=%.3fs max=%.3fs",
          completion_sum / frame_durations.size(),
          *std::min_element(frame_durations.begin(), frame_durations.end()),
          *std::max_element(frame_durations.begin(), frame_durations.end()));
    }
    RCLCPP_INFO(
        get_logger(),
        "Completed all %zu dual-arm PTP actions; log=%s",
        solutions.size(),
        log_file_.c_str());
    return 0;
  }

  std::string left_joint_topic_;
  std::string right_joint_topic_;
  std::string left_action_name_;
  std::string right_action_name_;
  std::string left_group_name_;
  std::string right_group_name_;
  std::string left_arm_base_link_;
  std::string right_arm_base_link_;
  std::string left_tip_link_;
  std::string right_tip_link_;
  std::vector<std::string> left_joint_names_;
  std::vector<std::string> right_joint_names_;
  std::vector<double> action_trajectory_;
  double tool_offset_z_m_{0.174};
  double ik_timeout_s_{0.1};
  double max_joint_delta_rad_{1.0};
  double max_joint_velocity_{0.2};
  double goal_tolerance_{0.01};
  double wait_timeout_s_{10.0};
  std::string log_file_;
  std::string mode_;
  std::string left_trajectory_topic_;
  std::string right_trajectory_topic_;
  double input_action_rate_hz_{15.0};
  double stream_rate_hz_{50.0};
  int chunk_size_{16};
  double chunk_publish_period_s_{0.1};
  double jtc_initial_delay_s_{1.0};
  std::string stream_log_file_;
  bool execute_{false};
  bool confirm_{false};

  JointState::SharedPtr left_joints_;
  JointState::SharedPtr right_joints_;
  rclcpp::Subscription<JointState>::SharedPtr left_joint_subscription_;
  rclcpp::Subscription<JointState>::SharedPtr right_joint_subscription_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int result = 1;
  try {
    result = std::make_shared<DuoPTPEpisode>()->run();
  } catch (const std::exception& exception) {
    RCLCPP_ERROR(rclcpp::get_logger("franka_duo_episode_ptp"), "%s", exception.what());
  }
  rclcpp::shutdown();
  return result;
}
