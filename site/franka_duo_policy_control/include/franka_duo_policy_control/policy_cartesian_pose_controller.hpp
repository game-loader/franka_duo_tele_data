#pragma once

#include <memory>
#include <cstdint>
#include <string>

#include <Eigen/Dense>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>

#include "franka_semantic_components/franka_cartesian_pose_interface.hpp"
#include "franka_duo_policy_control/velocity_servo.hpp"

namespace franka_duo_policy_control {

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class PolicyCartesianPoseController final : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::return_type update(
      const rclcpp::Time& time,
      const rclcpp::Duration& period) override;

 private:
  struct TargetPose {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
    std::int64_t received_ns{0};
  };

  static Eigen::Vector3d orientationError(
      const Eigen::Quaterniond& current,
      const Eigen::Quaterniond& target);

  void equilibriumPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr message);

  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface>
      franka_cartesian_pose_;

  std::string arm_prefix_;
  std::string target_topic_;
  std::string expected_frame_id_;
  bool allow_motion_{false};
  double target_timeout_s_{0.25};
  double linear_kp_{16.0};
  double linear_kd_{8.0};
  double linear_max_velocity_{0.15};
  double linear_max_acceleration_{0.5};
  double linear_max_jerk_{5.0};
  double angular_kp_{16.0};
  double angular_kd_{8.0};
  double angular_max_velocity_{1.5};
  double angular_max_acceleration_{5.0};
  double angular_max_jerk_{50.0};

  Eigen::Vector3d position_d_{Eigen::Vector3d::Zero()};
  VelocityServo linear_servo_;
  Eigen::Quaterniond orientation_d_{Eigen::Quaterniond::Identity()};
  VelocityServo angular_servo_;

  realtime_tools::RealtimeBuffer<TargetPose> target_pose_buffer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_subscription_;

  bool interfaces_assigned_{false};
};

}  // namespace franka_duo_policy_control
