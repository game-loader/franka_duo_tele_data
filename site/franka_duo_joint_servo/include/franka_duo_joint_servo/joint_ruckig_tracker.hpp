#pragma once

// Seven-axis jerk-limited tracker.  Ruckig's position interface follows a
// moving target (position and velocity) every cycle; the internal state is
// never reset when the target changes, so commanded position, velocity and
// acceleration stay continuous across chunk boundaries and target holds.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include <ruckig/ruckig.hpp>

#include "franka_duo_joint_servo/joint_timeline.hpp"

namespace franka_duo_joint_servo {

class JointRuckigTracker {
 public:
  void configure(
      double period_s,
      const Joints& max_velocity,
      const Joints& max_acceleration,
      const Joints& max_jerk) {
    if (!std::isfinite(period_s) || period_s <= 0.0) {
      throw std::invalid_argument("tracker period must be finite and positive");
    }
    for (std::size_t j = 0; j < kJoints; ++j) {
      for (const double limit : {max_velocity[j], max_acceleration[j], max_jerk[j]}) {
        if (!std::isfinite(limit) || limit <= 0.0) {
          throw std::invalid_argument("tracker limits must be finite and positive");
        }
      }
    }
    period_ = period_s;
    ruckig_ = std::make_unique<ruckig::Ruckig<kJoints>>(period_s);
    input_ = ruckig::InputParameter<kJoints>();
    output_ = ruckig::OutputParameter<kJoints>();
    input_.control_interface = ruckig::ControlInterface::Position;
    input_.synchronization = ruckig::Synchronization::None;
    input_.max_velocity = max_velocity;
    input_.max_acceleration = max_acceleration;
    input_.max_jerk = max_jerk;
    configured_ = true;
    reset(Joints{});
  }

  bool configured() const { return configured_; }
  double period() const { return period_; }

  void reset(const Joints& position) {
    input_.current_position = position;
    input_.current_velocity.fill(0.0);
    input_.current_acceleration.fill(0.0);
    input_.target_position = position;
    input_.target_velocity.fill(0.0);
    input_.target_acceleration.fill(0.0);
  }

  // Advance one period toward the target.  Returns false and leaves the
  // command unchanged when Ruckig rejects the input.
  bool step(
      const Joints& target_position,
      const Joints& target_velocity,
      const Joints& target_acceleration = Joints{}) {
    if (!configured_) {
      return false;
    }
    for (std::size_t j = 0; j < kJoints; ++j) {
      if (!std::isfinite(target_position[j]) || !std::isfinite(target_velocity[j]) ||
          !std::isfinite(target_acceleration[j])) {
        return false;
      }
      // Ruckig rejects target derivatives above the limits; keep a margin.
      // Passing the timeline's own velocity and acceleration lets the
      // tracker ride the spline instead of braking toward a static target.
      const double velocity_bound = 0.98 * input_.max_velocity[j];
      const double acceleration_bound = 0.98 * input_.max_acceleration[j];
      input_.target_position[j] = target_position[j];
      input_.target_velocity[j] = std::clamp(target_velocity[j], -velocity_bound, velocity_bound);
      input_.target_acceleration[j] =
          std::clamp(target_acceleration[j], -acceleration_bound, acceleration_bound);
    }
    const auto result = ruckig_->update(input_, output_);
    if (result != ruckig::Result::Working && result != ruckig::Result::Finished) {
      return false;
    }
    output_.pass_to_input(input_);
    return true;
  }

  const Joints& position() const { return input_.current_position; }
  const Joints& velocity() const { return input_.current_velocity; }
  const Joints& acceleration() const { return input_.current_acceleration; }
  const Joints& max_velocity() const { return input_.max_velocity; }

 private:
  bool configured_{false};
  double period_{0.001};
  std::unique_ptr<ruckig::Ruckig<kJoints>> ruckig_;
  ruckig::InputParameter<kJoints> input_;
  ruckig::OutputParameter<kJoints> output_;
};

}  // namespace franka_duo_joint_servo
