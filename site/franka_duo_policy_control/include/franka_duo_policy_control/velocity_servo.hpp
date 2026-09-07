#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>
#include <ruckig/ruckig.hpp>

namespace franka_duo_policy_control {

// Ruckig integrates the commanded velocity without resetting derivatives when
// a new target arrives. Limits are Euclidean norms, not per-axis maxima.
class VelocityServo {
 public:
  static constexpr double kPeriod = 0.001;

  void configure(double velocity, double acceleration, double jerk) {
    for (double limit : {velocity, acceleration, jerk}) {
      if (!std::isfinite(limit) || limit <= 0.0) {
        throw std::invalid_argument("servo limits must be finite and positive");
      }
    }
    const double axis_velocity = velocity / std::sqrt(3.0);
    const double axis_jerk = jerk / std::sqrt(3.0);
    const double axis_acceleration =
        std::min(acceleration / std::sqrt(3.0), std::sqrt(axis_velocity * axis_jerk));
    // Velocity mode does not enforce max_velocity. Reserve the largest speed
    // increase while nonzero acceleration is ramped back to zero at max jerk.
    target_limit_ = axis_velocity - axis_acceleration * axis_acceleration / (2.0 * axis_jerk);
    input_.control_interface = ruckig::ControlInterface::Velocity;
    input_.synchronization = ruckig::Synchronization::None;
    input_.max_velocity.fill(axis_velocity);
    input_.max_acceleration.fill(axis_acceleration);
    input_.max_jerk.fill(axis_jerk);
    reset();
  }

  void reset() {
    input_.current_position.fill(0.0);
    input_.current_velocity.fill(0.0);
    input_.current_acceleration.fill(0.0);
    input_.target_position.fill(0.0);
    input_.target_velocity.fill(0.0);
    input_.target_acceleration.fill(0.0);
    trajectory_.reset();
  }

  bool step(const Eigen::Vector3d& requested_velocity, Eigen::Vector3d& displacement) {
    if (!requested_velocity.allFinite()) {
      return false;
    }
    const auto previous_position = input_.current_position;
    for (std::size_t i = 0; i < 3; ++i) {
      input_.target_velocity[i] = std::clamp(requested_velocity[i], -target_limit_, target_limit_);
    }
    const auto result = trajectory_.update(input_, output_);
    if (result != ruckig::Result::Working && result != ruckig::Result::Finished) {
      return false;
    }
    for (std::size_t i = 0; i < 3; ++i) {
      displacement[i] = output_.new_position[i] - previous_position[i];
    }
    output_.pass_to_input(input_);
    return displacement.allFinite();
  }

  Eigen::Vector3d velocity() const {
    return Eigen::Map<const Eigen::Vector3d>(input_.current_velocity.data());
  }

  Eigen::Vector3d acceleration() const {
    return Eigen::Map<const Eigen::Vector3d>(input_.current_acceleration.data());
  }

 private:
  ruckig::Ruckig<3> trajectory_{kPeriod};
  ruckig::InputParameter<3> input_;
  ruckig::OutputParameter<3> output_;
  double target_limit_{0.0};
};

}  // namespace franka_duo_policy_control
