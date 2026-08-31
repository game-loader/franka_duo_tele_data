#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <franka_msgs/action/ptp_motion.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace {

using namespace std::chrono_literals;
using PTPMotion = franka_msgs::action::PTPMotion;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using JointState = sensor_msgs::msg::JointState;

constexpr std::size_t kArmJointCount = 7;

bool isFinite(double value) {
  return std::isfinite(value);
}

Eigen::Isometry3d toEigen(const geometry_msgs::msg::Pose& message) {
  const Eigen::Quaterniond orientation(
      message.orientation.w,
      message.orientation.x,
      message.orientation.y,
      message.orientation.z);
  if (!orientation.coeffs().allFinite() || orientation.norm() < 1e-8) {
    throw std::runtime_error("current pose has an invalid orientation");
  }

  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = orientation.normalized().toRotationMatrix();
  result.translation() << message.position.x, message.position.y, message.position.z;
  if (!result.matrix().allFinite()) {
    throw std::runtime_error("current pose has non-finite position");
  }
  return result;
}

std::optional<double> findJointPosition(
    const JointState& message,
    const std::string& expected_name,
    std::size_t joint_index) {
  for (std::size_t index = 0; index < message.name.size() && index < message.position.size(); ++index) {
    if (message.name[index] == expected_name && isFinite(message.position[index])) {
      return message.position[index];
    }
  }

  const std::string suffix = "_joint" + std::to_string(joint_index + 1);
  for (std::size_t index = 0; index < message.name.size() && index < message.position.size(); ++index) {
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

}  // namespace

class LeftPTPStep final : public rclcpp::Node {
 public:
  LeftPTPStep() : Node("franka_duo_left_ptp_step") {
    declare_parameter<std::string>(
        "pose_topic",
        "/left/franka_robot_state_broadcaster/current_pose");
    declare_parameter<std::string>(
        "joint_topic",
        "/left/franka_robot_state_broadcaster/measured_joint_states");
    declare_parameter<std::string>("action_name", "/left/action_server/ptp_motion");
    declare_parameter<std::string>("group_name", "left_arm");
    declare_parameter<std::string>("arm_base_link", "left_fr3v2_link0");
    declare_parameter<double>("offset_m", 0.01);
    declare_parameter<double>("tool_offset_z_m", 0.174);
    declare_parameter<double>("ik_timeout_s", 0.005);
    declare_parameter<double>("max_joint_delta_rad", 0.35);
    declare_parameter<double>("max_joint_velocity", 0.2);
    declare_parameter<double>("goal_tolerance", 0.01);
    declare_parameter<double>("wait_timeout_s", 10.0);
    declare_parameter<bool>("execute", false);
    declare_parameter<bool>("confirm", false);

    pose_topic_ = get_parameter("pose_topic").as_string();
    joint_topic_ = get_parameter("joint_topic").as_string();
    action_name_ = get_parameter("action_name").as_string();
    group_name_ = get_parameter("group_name").as_string();
    arm_base_link_ = get_parameter("arm_base_link").as_string();
    offset_m_ = get_parameter("offset_m").as_double();
    tool_offset_z_m_ = get_parameter("tool_offset_z_m").as_double();
    ik_timeout_s_ = get_parameter("ik_timeout_s").as_double();
    max_joint_delta_rad_ = get_parameter("max_joint_delta_rad").as_double();
    max_joint_velocity_ = get_parameter("max_joint_velocity").as_double();
    goal_tolerance_ = get_parameter("goal_tolerance").as_double();
    wait_timeout_s_ = get_parameter("wait_timeout_s").as_double();
    execute_ = get_parameter("execute").as_bool();
    confirm_ = get_parameter("confirm").as_bool();

    if (!isFinite(offset_m_) || !isFinite(tool_offset_z_m_) ||
        !isFinite(ik_timeout_s_) || ik_timeout_s_ <= 0.0 ||
        !isFinite(max_joint_delta_rad_) || max_joint_delta_rad_ <= 0.0 ||
        !isFinite(max_joint_velocity_) || max_joint_velocity_ <= 0.0 ||
        !isFinite(goal_tolerance_) || goal_tolerance_ <= 0.0 ||
        !isFinite(wait_timeout_s_) || wait_timeout_s_ <= 0.0) {
      throw std::invalid_argument("PTP step parameters must be finite and positive");
    }

    pose_subscription_ = create_subscription<PoseStamped>(
        pose_topic_,
        rclcpp::SensorDataQoS(),
        [this](const PoseStamped::SharedPtr message) { current_pose_ = message; });
    joint_subscription_ = create_subscription<JointState>(
        joint_topic_,
        rclcpp::SensorDataQoS(),
        [this](const JointState::SharedPtr message) { current_joints_ = message; });
  }

  int run() {
    waitForSamples();

    robot_model_loader::RobotModelLoader model_loader(shared_from_this());
    const auto& robot_model = model_loader.getModel();
    if (robot_model == nullptr) {
      throw std::runtime_error(
          "MoveIt robot model is unavailable; launch this node with the official Duo description");
    }

    const auto* group = robot_model->getJointModelGroup(group_name_);
    if (group == nullptr) {
      throw std::runtime_error("MoveIt joint group not found: " + group_name_);
    }
    if (group->getVariableCount() != kArmJointCount) {
      throw std::runtime_error("Expected a 7-variable arm group");
    }
    RCLCPP_INFO(
        get_logger(),
        "MoveIt group=%s is_chain=%s solver=%s",
        group_name_.c_str(),
        group->isChain() ? "true" : "false",
        group->getSolverInstance() == nullptr ? "missing" : "available");
    if (group->getSolverInstance() != nullptr) {
      RCLCPP_INFO(
          get_logger(),
          "MoveIt IK solver base=%s tip=%s",
          group->getSolverInstance()->getBaseFrame().c_str(),
          group->getSolverInstance()->getTipFrame().c_str());
    }

    moveit::core::RobotState state(robot_model);
    state.setToDefaultValues();
    const auto joint_names = group->getVariableNames();
    if (joint_names.size() != kArmJointCount) {
      throw std::runtime_error("MoveIt arm group does not contain exactly 7 variables");
    }

    std::map<std::string, double> seed;
    std::vector<double> seed_positions;
    seed_positions.reserve(joint_names.size());
    for (std::size_t index = 0; index < joint_names.size(); ++index) {
      const auto value = findJointPosition(*current_joints_, joint_names[index], index);
      if (!value.has_value()) {
        throw std::runtime_error("Measured joint state is missing " + joint_names[index]);
      }
      seed[joint_names[index]] = value.value();
      seed_positions.push_back(value.value());
    }
    state.setVariablePositions(seed);
    state.update();

    const auto* arm_base = robot_model->getLinkModel(arm_base_link_);
    if (arm_base == nullptr) {
      throw std::runtime_error("MoveIt arm base link not found: " + arm_base_link_);
    }

    // current_pose is the mounted tool pose, while MoveIt's arm group ends at
    // link8. The broadcaster reports the pose in the Franka-local base frame,
    // while the Duo MoveIt model has a larger root frame and mounting offset.
    Eigen::Isometry3d target_tool_pose_in_arm_base = toEigen(current_pose_->pose);
    target_tool_pose_in_arm_base.translation().x() += offset_m_;
    const Eigen::Isometry3d model_from_arm_base =
        state.getGlobalLinkTransform(arm_base);
    const Eigen::Isometry3d target_tool_pose =
        model_from_arm_base * target_tool_pose_in_arm_base;

    // The real tool/TCP is 0.174 m along the local +Z direction from link8.
    Eigen::Isometry3d tip_to_tool = Eigen::Isometry3d::Identity();
    tip_to_tool.translation().z() = tool_offset_z_m_;
    const Eigen::Isometry3d target_pose = target_tool_pose * tip_to_tool.inverse();
    const Eigen::Isometry3d current_tip_pose =
        state.getGlobalLinkTransform("left_fr3v2_link8");
    RCLCPP_INFO(
        get_logger(),
        "IK target link8=(%.6f, %.6f, %.6f), FK current link8=(%.6f, %.6f, %.6f)",
        target_pose.translation().x(),
        target_pose.translation().y(),
        target_pose.translation().z(),
        current_tip_pose.translation().x(),
        current_tip_pose.translation().y(),
        current_tip_pose.translation().z());

    const std::vector<double> consistency_limits(
        kArmJointCount,
        max_joint_delta_rad_);
    if (!state.setFromIK(
            group,
            target_pose,
            "left_fr3v2_link8",
            consistency_limits,
            ik_timeout_s_)) {
      throw std::runtime_error("KDL IK failed for the requested +X 0.01 m offset");
    }
    state.update();
    if (!state.satisfiesBounds(group)) {
      throw std::runtime_error("IK solution is outside the MoveIt joint bounds");
    }

    std::vector<double> solution;
    state.copyJointGroupPositions(group, solution);
    if (solution.size() != kArmJointCount) {
      throw std::runtime_error("IK solution does not contain exactly 7 joint values");
    }
    double largest_joint_delta = 0.0;
    for (std::size_t index = 0; index < solution.size(); ++index) {
      largest_joint_delta =
          std::max(largest_joint_delta, std::abs(solution[index] - seed_positions[index]));
    }
    if (largest_joint_delta > max_joint_delta_rad_) {
      throw std::runtime_error("IK solution is too far from the measured joint seed");
    }

    RCLCPP_INFO(
        get_logger(),
        "IK succeeded; local_pose_frame=%s arm_base=%s tip=left_fr3v2_link8 "
        "tool_offset=(%.4f, %.4f, %.4f) m "
        "offset_x=%.4f m max_joint_delta=%.4f rad",
        current_pose_->header.frame_id.c_str(),
        arm_base_link_.c_str(),
        0.0,
        0.0,
        tool_offset_z_m_,
        offset_m_,
        largest_joint_delta);
    RCLCPP_INFO(
        get_logger(),
        "q=[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]",
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
        solution[6]);

    if (!execute_ || !confirm_) {
      RCLCPP_WARN(
          get_logger(),
          "Dry-run only. Use execute:=true confirm:=true to send this target to PTP");
      return 0;
    }

    return sendPTP(solution);
  }

 private:
  void waitForSamples() {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(wait_timeout_s_);
    rclcpp::WallRate rate(100.0);
    while (rclcpp::ok() &&
           (current_pose_ == nullptr || current_joints_ == nullptr) &&
           std::chrono::steady_clock::now() < deadline) {
      rclcpp::spin_some(shared_from_this());
      rate.sleep();
    }
    if (current_pose_ == nullptr) {
      throw std::runtime_error("Timed out waiting for " + pose_topic_);
    }
    if (current_joints_ == nullptr) {
      throw std::runtime_error("Timed out waiting for " + joint_topic_);
    }
  }

  int sendPTP(const std::vector<double>& solution) {
    auto client = rclcpp_action::create_client<PTPMotion>(shared_from_this(), action_name_);
    if (!client->wait_for_action_server(std::chrono::duration<double>(wait_timeout_s_))) {
      throw std::runtime_error("PTP action server is unavailable: " + action_name_);
    }

    PTPMotion::Goal goal;
    goal.goal_joint_configuration = solution;
    goal.maximum_joint_velocities.assign(kArmJointCount, max_joint_velocity_);
    goal.goal_tolerance = goal_tolerance_;

    const auto goal_future = client->async_send_goal(goal);
    if (rclcpp::spin_until_future_complete(
            shared_from_this(),
            goal_future,
            std::chrono::duration<double>(wait_timeout_s_)) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      throw std::runtime_error("Timed out while sending the PTP goal");
    }
    const auto goal_handle = goal_future.get();
    if (goal_handle == nullptr) {
      throw std::runtime_error("PTP goal was rejected");
    }

    RCLCPP_WARN(get_logger(), "Sending left-arm PTP target");
    const auto result_future = client->async_get_result(goal_handle);
    if (rclcpp::spin_until_future_complete(
            shared_from_this(),
            result_future,
            std::chrono::duration<double>(wait_timeout_s_ * 10.0)) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      throw std::runtime_error("Timed out waiting for the PTP result");
    }

    const auto wrapped_result = result_future.get();
    if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED ||
        wrapped_result.result == nullptr) {
      RCLCPP_ERROR(get_logger(), "PTP action did not succeed");
      return 1;
    }
    RCLCPP_INFO(
        get_logger(),
        "PTP completed with target_status=%u",
        wrapped_result.result->target_status.status);
    return 0;
  }

  std::string pose_topic_;
  std::string joint_topic_;
  std::string action_name_;
  std::string group_name_;
  std::string arm_base_link_;
  double offset_m_{0.01};
  double tool_offset_z_m_{0.174};
  double ik_timeout_s_{0.005};
  double max_joint_delta_rad_{0.35};
  double max_joint_velocity_{0.2};
  double goal_tolerance_{0.01};
  double wait_timeout_s_{10.0};
  bool execute_{false};
  bool confirm_{false};

  PoseStamped::SharedPtr current_pose_;
  JointState::SharedPtr current_joints_;
  rclcpp::Subscription<PoseStamped>::SharedPtr pose_subscription_;
  rclcpp::Subscription<JointState>::SharedPtr joint_subscription_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int result = 1;
  try {
    result = std::make_shared<LeftPTPStep>()->run();
  } catch (const std::exception& exception) {
    RCLCPP_ERROR(rclcpp::get_logger("franka_duo_left_ptp_step"), "%s", exception.what());
  }
  rclcpp::shutdown();
  return result;
}
