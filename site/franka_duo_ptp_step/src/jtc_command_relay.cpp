#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

class JTCCommandRelay final : public rclcpp::Node {
 public:
  JTCCommandRelay() : Node("jtc_command_relay") {
    declare_parameter<std::string>("input_topic", "");
    declare_parameter<std::string>("output_topic", "");
    declare_parameter<std::vector<std::string>>("expected_joint_names", {});
    declare_parameter<bool>("enable_robot", false);

    input_topic_ = get_parameter("input_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    expected_joint_names_ = get_parameter("expected_joint_names").as_string_array();
    enable_robot_ = get_parameter("enable_robot").as_bool();
    if (input_topic_.empty() || output_topic_.empty() || expected_joint_names_.empty()) {
      throw std::invalid_argument(
          "input_topic, output_topic, and expected_joint_names are required");
    }

    publisher_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
        output_topic_, rclcpp::QoS(10).reliable());
    subscription_ = create_subscription<trajectory_msgs::msg::JointTrajectory>(
        input_topic_,
        rclcpp::QoS(10).reliable(),
        [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr message) {
          forward(*message);
        });
    RCLCPP_INFO(
        get_logger(),
        "Relay %s -> %s; enable_robot=%s",
        input_topic_.c_str(),
        output_topic_.c_str(),
        enable_robot_ ? "true" : "false");
  }

 private:
  static bool finite(double value) {
    return std::isfinite(value);
  }

  bool valid(const trajectory_msgs::msg::JointTrajectory& message) const {
    if (message.joint_names != expected_joint_names_ || message.points.empty()) {
      return false;
    }
    int64_t previous_nanoseconds = -1;
    for (const auto& point : message.points) {
      if (point.positions.size() != expected_joint_names_.size() ||
          (!point.velocities.empty() &&
           point.velocities.size() != expected_joint_names_.size()) ||
          (!point.accelerations.empty() &&
           point.accelerations.size() != expected_joint_names_.size())) {
        return false;
      }
      const int64_t nanoseconds =
          static_cast<int64_t>(point.time_from_start.sec) * 1'000'000'000LL +
          static_cast<int64_t>(point.time_from_start.nanosec);
      if (nanoseconds < 0 || (previous_nanoseconds >= 0 && nanoseconds < previous_nanoseconds)) {
        return false;
      }
      previous_nanoseconds = nanoseconds;
      for (const auto value : point.positions) {
        if (!finite(value)) {
          return false;
        }
      }
      for (const auto value : point.velocities) {
        if (!finite(value)) {
          return false;
        }
      }
      for (const auto value : point.accelerations) {
        if (!finite(value)) {
          return false;
        }
      }
    }
    return true;
  }

  void forward(const trajectory_msgs::msg::JointTrajectory& message) {
    if (!valid(message)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          5000,
          "Dropped invalid JointTrajectory from %s",
          input_topic_.c_str());
      return;
    }
    if (!enable_robot_) {
      RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          5000,
          "Robot output disabled; dropping trajectory from %s",
          input_topic_.c_str());
      return;
    }
    publisher_->publish(message);
  }

  std::string input_topic_;
  std::string output_topic_;
  std::vector<std::string> expected_joint_names_;
  bool enable_robot_{false};
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr publisher_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr subscription_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<JTCCommandRelay>());
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("jtc_command_relay"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
