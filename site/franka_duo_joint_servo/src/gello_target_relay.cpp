// Site-owned gate between the joint servo and the site JointImpedanceController.
//
// Forwards sensor_msgs/JointState targets to the controller input topic
// (relative "gello/joint_states" inside the arm namespace) only when
// enable_robot is true, and Float32 gripper targets to the site gripper
// client only when enable_gripper is true.  Targets must carry the expected
// seven joint names in order, finite values and a fresh stamp.

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32.hpp>

class GelloTargetRelay final : public rclcpp::Node {
 public:
  GelloTargetRelay() : Node("gello_target_relay") {
    declare_parameter<std::string>("input_topic", "");
    declare_parameter<std::string>("output_topic", "");
    declare_parameter<std::string>("gripper_input_topic", "");
    declare_parameter<std::string>("gripper_output_topic", "");
    declare_parameter<std::vector<std::string>>("expected_joint_names", {});
    declare_parameter<double>("max_target_age_s", 0.1);
    declare_parameter<bool>("enable_robot", false);
    declare_parameter<bool>("enable_gripper", false);
    // The site JointImpedanceController shuts the whole driver down when the
    // next target arrives more than 0.5 s after the previous one.  When the
    // servo stops publishing, keep re-sending the last forwarded target with
    // a fresh stamp so the arm holds its pose instead of losing the driver.
    declare_parameter<bool>("hold_on_input_loss", true);

    input_topic_ = get_parameter("input_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    gripper_input_topic_ = get_parameter("gripper_input_topic").as_string();
    gripper_output_topic_ = get_parameter("gripper_output_topic").as_string();
    expected_joint_names_ = get_parameter("expected_joint_names").as_string_array();
    max_target_age_s_ = get_parameter("max_target_age_s").as_double();
    enable_robot_ = get_parameter("enable_robot").as_bool();
    enable_gripper_ = get_parameter("enable_gripper").as_bool();
    hold_on_input_loss_ = get_parameter("hold_on_input_loss").as_bool();
    if (input_topic_.empty() || output_topic_.empty() || expected_joint_names_.size() != 7U) {
      throw std::invalid_argument(
          "input_topic, output_topic and seven expected_joint_names are required");
    }
    if (input_topic_ == output_topic_) {
      throw std::invalid_argument("input_topic and output_topic must differ");
    }
    if (!std::isfinite(max_target_age_s_) || max_target_age_s_ <= 0.0 || max_target_age_s_ >= 0.5) {
      throw std::invalid_argument("max_target_age_s must be in (0, 0.5)");
    }

    publisher_ = create_publisher<sensor_msgs::msg::JointState>(output_topic_, rclcpp::QoS(1).reliable());
    subscription_ = create_subscription<sensor_msgs::msg::JointState>(
        input_topic_, rclcpp::QoS(1).reliable(),
        [this](const sensor_msgs::msg::JointState::SharedPtr message) { forward(*message); });
    if (!gripper_input_topic_.empty() && !gripper_output_topic_.empty()) {
      gripper_publisher_ =
          create_publisher<std_msgs::msg::Float32>(gripper_output_topic_, rclcpp::QoS(10).reliable());
      gripper_subscription_ = create_subscription<std_msgs::msg::Float32>(
          gripper_input_topic_, rclcpp::QoS(10).reliable(),
          [this](const std_msgs::msg::Float32::SharedPtr message) { forwardGripper(*message); });
    }
    RCLCPP_INFO(
        get_logger(), "Relay %s -> %s; enable_robot=%s enable_gripper=%s", input_topic_.c_str(),
        output_topic_.c_str(), enable_robot_ ? "true" : "false", enable_gripper_ ? "true" : "false");
    if (enable_robot_) {
      RCLCPP_WARN(get_logger(), "Robot output ENABLED: targets reach the joint impedance controller");
    }
    if (hold_on_input_loss_) {
      hold_timer_ = create_wall_timer(std::chrono::milliseconds(5), [this] { holdTick(); });
    }
  }

 private:
  void holdTick() {
    if (!enable_robot_ || !have_last_) {
      return;
    }
    const double age = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - last_forward_time_).count();
    if (age < 0.02) {
      return;
    }
    sensor_msgs::msg::JointState held = last_forwarded_;
    held.header.stamp = now();
    publisher_->publish(held);
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "No target from %s for %.2f s; holding the last forwarded joint target", input_topic_.c_str(), age);
  }

  bool valid(const sensor_msgs::msg::JointState& message) const {
    if (message.name != expected_joint_names_ || message.position.size() != 7U) {
      return false;
    }
    for (const auto value : message.position) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
    const double age = (now() - rclcpp::Time(message.header.stamp)).seconds();
    return age >= -max_target_age_s_ && age <= max_target_age_s_;
  }

  void forward(const sensor_msgs::msg::JointState& message) {
    if (!valid(message)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "Dropped invalid or stale joint target from %s",
          input_topic_.c_str());
      return;
    }
    if (!enable_robot_) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "Robot output disabled; dropping targets from %s",
          input_topic_.c_str());
      return;
    }
    publisher_->publish(message);
    last_forwarded_ = message;
    last_forward_time_ = std::chrono::steady_clock::now();
    have_last_ = true;
  }

  void forwardGripper(const std_msgs::msg::Float32& message) {
    if (!std::isfinite(message.data) || message.data < 0.0F || message.data > 1.0F) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Dropped gripper target outside [0, 1]");
      return;
    }
    if (!enable_gripper_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Gripper output disabled; dropping target");
      return;
    }
    gripper_publisher_->publish(message);
  }

  std::string input_topic_, output_topic_, gripper_input_topic_, gripper_output_topic_;
  std::vector<std::string> expected_joint_names_;
  double max_target_age_s_{0.1};
  bool enable_robot_{false};
  bool enable_gripper_{false};
  bool hold_on_input_loss_{true};
  bool have_last_{false};
  sensor_msgs::msg::JointState last_forwarded_;
  std::chrono::steady_clock::time_point last_forward_time_{};
  rclcpp::TimerBase::SharedPtr hold_timer_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gripper_publisher_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr gripper_subscription_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<GelloTargetRelay>());
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("gello_target_relay"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
