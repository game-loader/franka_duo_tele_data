// Based on the official Franka ROS 2 Cartesian pose example controller.
// The policy target is buffered outside the real-time loop and consumed at 1 kHz.

#include "franka_duo_policy_control/policy_cartesian_pose_controller.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <tuple>

#include <pluginlib/class_list_macros.hpp>

namespace franka_duo_policy_control {

controller_interface::InterfaceConfiguration
PolicyCartesianPoseController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration configuration;
  configuration.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  if (franka_cartesian_pose_ != nullptr) {
    configuration.names = franka_cartesian_pose_->get_command_interface_names();
  }
  return configuration;
}

controller_interface::InterfaceConfiguration
PolicyCartesianPoseController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration configuration;
  configuration.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  if (franka_cartesian_pose_ != nullptr) {
    configuration.names = franka_cartesian_pose_->get_state_interface_names();
  }
  return configuration;
}

CallbackReturn PolicyCartesianPoseController::on_init() {
  try {
    auto_declare<std::string>("arm_prefix", "");
    auto_declare<std::string>("target_topic", "");
    auto_declare<std::string>("expected_frame_id", "");
    auto_declare<bool>("allow_motion", false);
    auto_declare<double>("linear_kp", 16.0);
    auto_declare<double>("linear_kd", 8.0);
    auto_declare<double>("linear_max_velocity", 0.15);
    auto_declare<double>("linear_max_acceleration", 0.5);
    auto_declare<double>("linear_max_jerk", 5.0);
    auto_declare<double>("angular_kp", 16.0);
    auto_declare<double>("angular_kd", 8.0);
    auto_declare<double>("angular_max_velocity", 1.5);
    auto_declare<double>("angular_max_acceleration", 5.0);
    auto_declare<double>("angular_max_jerk", 50.0);

    arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
    target_topic_ = get_node()->get_parameter("target_topic").as_string();
    expected_frame_id_ = get_node()->get_parameter("expected_frame_id").as_string();
    allow_motion_ = get_node()->get_parameter("allow_motion").as_bool();
    linear_kp_ = get_node()->get_parameter("linear_kp").as_double();
    linear_kd_ = get_node()->get_parameter("linear_kd").as_double();
    linear_max_velocity_ = get_node()->get_parameter("linear_max_velocity").as_double();
    linear_max_acceleration_ =
        get_node()->get_parameter("linear_max_acceleration").as_double();
    linear_max_jerk_ = get_node()->get_parameter("linear_max_jerk").as_double();
    angular_kp_ = get_node()->get_parameter("angular_kp").as_double();
    angular_kd_ = get_node()->get_parameter("angular_kd").as_double();
    angular_max_velocity_ = get_node()->get_parameter("angular_max_velocity").as_double();
    angular_max_acceleration_ =
        get_node()->get_parameter("angular_max_acceleration").as_double();
    angular_max_jerk_ = get_node()->get_parameter("angular_max_jerk").as_double();

    if (!arm_prefix_.empty() && arm_prefix_.back() != '_') {
      arm_prefix_ += "_";
    }
    if (target_topic_.empty()) {
      throw std::invalid_argument("target_topic must be set");
    }
    const double gains[] = {
        linear_kp_, linear_kd_, linear_max_velocity_, linear_max_acceleration_, linear_max_jerk_,
        angular_kp_, angular_kd_, angular_max_velocity_, angular_max_acceleration_,
        angular_max_jerk_};
    for (const double gain : gains) {
      if (!std::isfinite(gain) || gain <= 0.0) {
        throw std::invalid_argument("Cartesian servo parameters must be finite and positive");
      }
    }

    franka_cartesian_pose_ =
        std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
            arm_prefix_, false);
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(get_node()->get_logger(), "Controller initialization failed: %s", exception.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn PolicyCartesianPoseController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  target_pose_buffer_.initRT(TargetPose{});
  target_subscription_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      target_topic_,
      rclcpp::QoS(10).reliable(),
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
        equilibriumPoseCallback(message);
      });
  RCLCPP_INFO(
      get_node()->get_logger(),
      "Configured %s Cartesian pose policy controller: target=%s allow_motion=%s",
      arm_prefix_.c_str(),
      target_topic_.c_str(),
      allow_motion_ ? "true" : "false");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PolicyCartesianPoseController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (franka_cartesian_pose_ == nullptr) {
    return CallbackReturn::ERROR;
  }

  try {
    if (!franka_cartesian_pose_->assign_loaned_command_interfaces(command_interfaces_) ||
        !franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_)) {
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Controller activation failed: Franka Cartesian pose interfaces were not assigned");
      franka_cartesian_pose_->release_interfaces();
      return CallbackReturn::ERROR;
    }
    interfaces_assigned_ = true;

    Eigen::Quaterniond orientation_init;
    Eigen::Vector3d position_init;
    std::tie(orientation_init, position_init) =
        franka_cartesian_pose_->getCurrentOrientationAndTranslation();
    if (!position_init.allFinite() || !orientation_init.coeffs().allFinite() ||
        orientation_init.norm() < 1e-8) {
      throw std::runtime_error("current Cartesian pose is invalid");
    }
    orientation_init.normalize();
    position_d_ = position_init;
    linear_velocity_d_.setZero();
    linear_acceleration_d_.setZero();
    orientation_d_ = orientation_init;
    angular_velocity_d_.setZero();
    angular_acceleration_d_.setZero();

    TargetPose initial_target;
    initial_target.position = position_init;
    initial_target.orientation = orientation_init;
    target_pose_buffer_.initRT(initial_target);

    if (!franka_cartesian_pose_->setCommand(orientation_d_, position_d_)) {
      throw std::runtime_error("failed to initialize Cartesian pose command");
    }
  } catch (const std::exception& exception) {
    RCLCPP_ERROR(get_node()->get_logger(), "Controller activation failed: %s", exception.what());
    franka_cartesian_pose_->release_interfaces();
    interfaces_assigned_ = false;
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn PolicyCartesianPoseController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (franka_cartesian_pose_ != nullptr) {
    franka_cartesian_pose_->release_interfaces();
  }
  interfaces_assigned_ = false;
  return CallbackReturn::SUCCESS;
}

void PolicyCartesianPoseController::equilibriumPoseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr message) {
  if (message == nullptr) {
    return;
  }
  if (!expected_frame_id_.empty() && message->header.frame_id != expected_frame_id_) {
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        2000,
        "Ignoring %s target with frame_id=%s; expected %s",
        arm_prefix_.c_str(),
        message->header.frame_id.c_str(),
        expected_frame_id_.c_str());
    return;
  }

  TargetPose target;
  target.position << message->pose.position.x, message->pose.position.y, message->pose.position.z;
  target.orientation = Eigen::Quaterniond(
      message->pose.orientation.w,
      message->pose.orientation.x,
      message->pose.orientation.y,
      message->pose.orientation.z);
  if (!target.position.allFinite() || !target.orientation.coeffs().allFinite() ||
      target.orientation.norm() < 1e-8) {
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        2000,
        "Ignoring %s target with invalid Cartesian pose",
        arm_prefix_.c_str());
    return;
  }
  target.orientation.normalize();
  target_pose_buffer_.writeFromNonRT(target);
}

controller_interface::return_type PolicyCartesianPoseController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  if (!interfaces_assigned_ || franka_cartesian_pose_ == nullptr) {
    return controller_interface::return_type::ERROR;
  }

  const TargetPose target = *target_pose_buffer_.readFromRT();
  if (allow_motion_) {
    double dt = period.seconds();
    if (!std::isfinite(dt) || dt < 0.0005 || dt > 0.002) {
      dt = 0.001;
    }

    updateServoState(
        target.position,
        dt,
        linear_kp_,
        linear_kd_,
        linear_max_velocity_,
        linear_max_acceleration_,
        linear_max_jerk_,
        position_d_,
        linear_velocity_d_,
        linear_acceleration_d_);

    const Eigen::Vector3d orientation_target_error =
        orientationError(orientation_d_, target.orientation);
    const Eigen::Vector3d desired_angular_acceleration =
        clampNorm(
            angular_kp_ * orientation_target_error - angular_kd_ * angular_velocity_d_,
            angular_max_acceleration_);
    const Eigen::Vector3d angular_acceleration_delta =
        clampNorm(
            desired_angular_acceleration - angular_acceleration_d_,
            angular_max_jerk_ * dt);
    angular_acceleration_d_ =
        clampNorm(angular_acceleration_d_ + angular_acceleration_delta, angular_max_acceleration_);
    angular_velocity_d_ =
        clampNorm(angular_velocity_d_ + angular_acceleration_d_ * dt, angular_max_velocity_);

    const Eigen::Vector3d angular_step =
        angular_velocity_d_ * dt + 0.5 * angular_acceleration_d_ * dt * dt;
    const double angular_step_norm = angular_step.norm();
    if (std::isfinite(angular_step_norm) && angular_step_norm > 1e-12) {
      orientation_d_ =
          (Eigen::Quaterniond(Eigen::AngleAxisd(
               angular_step_norm, angular_step / angular_step_norm)) *
           orientation_d_)
              .normalized();
    }
  }

  if (!franka_cartesian_pose_->setCommand(orientation_d_, position_d_)) {
    RCLCPP_ERROR_THROTTLE(
        get_node()->get_logger(),
        *get_node()->get_clock(),
        2000,
        "Failed to set Cartesian pose command for %s",
        arm_prefix_.c_str());
    return controller_interface::return_type::ERROR;
  }
  return controller_interface::return_type::OK;
}

Eigen::Vector3d PolicyCartesianPoseController::clampNorm(
    const Eigen::Vector3d& value,
    double limit) {
  const double norm = value.norm();
  if (!std::isfinite(norm) || norm <= limit || norm < 1e-12) {
    return value;
  }
  return value * (limit / norm);
}

Eigen::Vector3d PolicyCartesianPoseController::orientationError(
    const Eigen::Quaterniond& current,
    const Eigen::Quaterniond& target) {
  Eigen::Quaterniond error = target * current.conjugate();
  error.normalize();
  if (error.w() < 0.0) {
    error.coeffs() = -error.coeffs();
  }
  const double scalar = std::clamp(error.w(), -1.0, 1.0);
  const double angle = 2.0 * std::acos(scalar);
  const double sine_half_angle = std::sin(0.5 * angle);
  if (!std::isfinite(angle) || angle < 1e-12 || std::abs(sine_half_angle) < 1e-10) {
    return Eigen::Vector3d::Zero();
  }
  return error.vec() * (angle / sine_half_angle);
}

void PolicyCartesianPoseController::updateServoState(
    const Eigen::Vector3d& target,
    double dt,
    double kp,
    double kd,
    double max_velocity,
    double max_acceleration,
    double max_jerk,
    Eigen::Vector3d& position,
    Eigen::Vector3d& velocity,
    Eigen::Vector3d& acceleration) {
  const Eigen::Vector3d error = target - position;
  const Eigen::Vector3d desired_acceleration =
      clampNorm(kp * error - kd * velocity, max_acceleration);
  const Eigen::Vector3d acceleration_delta =
      clampNorm(desired_acceleration - acceleration, max_jerk * dt);
  acceleration = clampNorm(acceleration + acceleration_delta, max_acceleration);
  velocity = clampNorm(velocity + acceleration * dt, max_velocity);
  position += velocity * dt + 0.5 * acceleration * dt * dt;
}

}  // namespace franka_duo_policy_control

PLUGINLIB_EXPORT_CLASS(
    franka_duo_policy_control::PolicyCartesianPoseController,
    controller_interface::ControllerInterface)
