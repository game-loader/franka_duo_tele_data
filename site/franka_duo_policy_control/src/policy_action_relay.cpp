#include "franka_duo_policy_control/action_math.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

namespace franka_duo_policy_control {
namespace {

using Float32MultiArray = std_msgs::msg::Float32MultiArray;
using PoseStamped = geometry_msgs::msg::PoseStamped;

}  // namespace

class PolicyActionRelay final : public rclcpp::Node {
 public:
  PolicyActionRelay() : Node("franka_duo_policy_action_relay") {
    declare_parameter<std::string>("input_topic", "/franka_duo/policy_action");
    declare_parameter<std::string>("action_frame", "midpoint");
    declare_parameter<std::string>("left_pose_topic", "/franka_duo/policy/left/target_pose");
    declare_parameter<std::string>("right_pose_topic", "/franka_duo/policy/right/target_pose");
    declare_parameter<std::string>(
        "left_gripper_topic", "/left/gripper/gripper_client/target_gripper_width_percent");
    declare_parameter<std::string>(
        "right_gripper_topic", "/right/gripper/gripper_client/target_gripper_width_percent");
    declare_parameter<std::string>("left_arm_frame", "left_fr3v2_link0");
    declare_parameter<std::string>("right_arm_frame", "right_fr3v2_link0");
    declare_parameter<bool>("enable_robot", false);
    declare_parameter<bool>("enable_gripper", false);

    input_topic_ = get_parameter("input_topic").as_string();
    action_frame_ = get_parameter("action_frame").as_string();
    if (action_frame_ != "midpoint" && action_frame_ != "link0") {
      throw std::invalid_argument("action_frame must be midpoint or link0");
    }
    left_pose_topic_ = get_parameter("left_pose_topic").as_string();
    right_pose_topic_ = get_parameter("right_pose_topic").as_string();
    left_gripper_topic_ = get_parameter("left_gripper_topic").as_string();
    right_gripper_topic_ = get_parameter("right_gripper_topic").as_string();
    left_arm_frame_ = get_parameter("left_arm_frame").as_string();
    right_arm_frame_ = get_parameter("right_arm_frame").as_string();
    enable_robot_ = get_parameter("enable_robot").as_bool();
    enable_gripper_ = get_parameter("enable_gripper").as_bool();
    left_midpoint_from_arm_base_ = trainingMidpointFromLeftArmBase();
    right_midpoint_from_arm_base_ = trainingMidpointFromRightArmBase();

    action_subscription_ = create_subscription<Float32MultiArray>(
        input_topic_,
        rclcpp::QoS(1).reliable(),
        std::bind(&PolicyActionRelay::onAction, this, std::placeholders::_1));
    left_pose_publisher_ = create_publisher<PoseStamped>(left_pose_topic_, rclcpp::QoS(10).reliable());
    right_pose_publisher_ =
        create_publisher<PoseStamped>(right_pose_topic_, rclcpp::QoS(10).reliable());
    left_gripper_publisher_ =
        create_publisher<std_msgs::msg::Float32>(left_gripper_topic_, rclcpp::QoS(10).reliable());
    right_gripper_publisher_ =
        create_publisher<std_msgs::msg::Float32>(right_gripper_topic_, rclcpp::QoS(10).reliable());

    if (enable_robot_) {
      RCLCPP_WARN(
          get_logger(),
          "Robot output is ENABLED; controller allow_motion must still be explicitly enabled");
    } else {
      RCLCPP_INFO(get_logger(), "Robot pose output is disabled by enable_robot=false");
    }
    if (enable_gripper_) {
      RCLCPP_WARN(get_logger(), "Gripper output is ENABLED");
    } else {
      RCLCPP_INFO(get_logger(), "Gripper output is disabled by enable_gripper=false");
    }
  }

 private:
  void onAction(const Float32MultiArray::SharedPtr message) {
    if (message == nullptr || message->data.size() != 20U) {
      RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Ignoring policy action: expected exactly 20 float values");
      return;
    }

    try {
      std::array<double, 6> left_rot6d{};
      std::array<double, 6> right_rot6d{};
      for (std::size_t index = 0; index < 6; ++index) {
        left_rot6d[index] = static_cast<double>(message->data[3U + index]);
        right_rot6d[index] = static_cast<double>(message->data[12U + index]);
      }
      const Matrix3d left_rotation = rot6dRowsToMatrix(left_rot6d);
      const Matrix3d right_rotation = rot6dRowsToMatrix(right_rot6d);
      const Eigen::Vector3d left_midpoint_position(
          message->data[0], message->data[1], message->data[2]);
      const Eigen::Vector3d right_midpoint_position(
          message->data[9], message->data[10], message->data[11]);
      if (!left_midpoint_position.allFinite() || !right_midpoint_position.allFinite()) {
        throw std::invalid_argument("policy action contains non-finite position");
      }
      const float left_gripper_target = gripperOpenFractionToSiteTarget(message->data[18]);
      const float right_gripper_target = gripperOpenFractionToSiteTarget(message->data[19]);

      const Matrix4d left_midpoint_from_ee =
          makeTransform(left_rotation, left_midpoint_position);
      const Matrix4d right_midpoint_from_ee =
          makeTransform(right_rotation, right_midpoint_position);
      const Matrix4d left_arm_from_ee =
          action_frame_ == "link0" ? left_midpoint_from_ee :
          (left_midpoint_from_arm_base_.inverse() * left_midpoint_from_ee).eval();
      const Matrix4d right_arm_from_ee =
          action_frame_ == "link0" ? right_midpoint_from_ee :
          (right_midpoint_from_arm_base_.inverse() * right_midpoint_from_ee).eval();
      if (!isHomogeneous(left_arm_from_ee) || !isHomogeneous(right_arm_from_ee) ||
          !isRotation(left_arm_from_ee.block<3, 3>(0, 0)) ||
          !isRotation(right_arm_from_ee.block<3, 3>(0, 0))) {
        throw std::invalid_argument("arm-base transform produced an invalid pose");
      }

      if (enable_robot_) {
        const auto stamp = now();
        const auto left_message = poseMessage(left_arm_from_ee, left_arm_frame_, stamp);
        const auto right_message = poseMessage(right_arm_from_ee, right_arm_frame_, stamp);
        left_pose_publisher_->publish(left_message);
        right_pose_publisher_->publish(right_message);
      }

      if (enable_gripper_) {
        std_msgs::msg::Float32 left_gripper;
        std_msgs::msg::Float32 right_gripper;
        left_gripper.data = left_gripper_target;
        right_gripper.data = right_gripper_target;
        left_gripper_publisher_->publish(left_gripper);
        right_gripper_publisher_->publish(right_gripper);
      }

      if (!enable_robot_ && !enable_gripper_) {
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 5000, "Decoded policy action; all outputs disabled");
      }
    } catch (const std::exception& exception) {
      RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Ignoring invalid policy action: %s",
          exception.what());
    }
  }

  static PoseStamped poseMessage(
      const Matrix4d& transform,
      const std::string& frame_id,
      const rclcpp::Time& stamp) {
    const Eigen::Quaterniond orientation =
        quaternionFromRotation(transform.block<3, 3>(0, 0));
    const Eigen::Vector3d position = transform.block<3, 1>(0, 3);
    PoseStamped message;
    message.header.stamp = stamp;
    message.header.frame_id = frame_id;
    message.pose.position.x = position.x();
    message.pose.position.y = position.y();
    message.pose.position.z = position.z();
    message.pose.orientation.x = orientation.x();
    message.pose.orientation.y = orientation.y();
    message.pose.orientation.z = orientation.z();
    message.pose.orientation.w = orientation.w();
    return message;
  }

  std::string input_topic_;
  std::string action_frame_;
  std::string left_pose_topic_;
  std::string right_pose_topic_;
  std::string left_gripper_topic_;
  std::string right_gripper_topic_;
  std::string left_arm_frame_;
  std::string right_arm_frame_;
  bool enable_robot_{false};
  bool enable_gripper_{false};
  Matrix4d left_midpoint_from_arm_base_{Matrix4d::Identity()};
  Matrix4d right_midpoint_from_arm_base_{Matrix4d::Identity()};

  rclcpp::Subscription<Float32MultiArray>::SharedPtr action_subscription_;
  rclcpp::Publisher<PoseStamped>::SharedPtr left_pose_publisher_;
  rclcpp::Publisher<PoseStamped>::SharedPtr right_pose_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr left_gripper_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr right_gripper_publisher_;
};

}  // namespace franka_duo_policy_control

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<franka_duo_policy_control::PolicyActionRelay>());
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("franka_duo_policy_action_relay"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
