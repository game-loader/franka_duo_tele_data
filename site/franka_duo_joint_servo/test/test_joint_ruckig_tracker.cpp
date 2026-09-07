#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include "franka_duo_joint_servo/joint_ruckig_tracker.hpp"

using franka_duo_joint_servo::JointRuckigTracker;
using franka_duo_joint_servo::Joints;

namespace {

Joints filled(double value) {
  Joints joints;
  joints.fill(value);
  return joints;
}

}  // namespace

TEST(JointRuckigTracker, RetargetingKeepsDerivativesBounded) {
  JointRuckigTracker tracker;
  const double dt = 0.001;
  tracker.configure(dt, filled(0.5), filled(1.0), filled(10.0));
  tracker.reset(filled(0.3));
  std::mt19937 random(7);
  std::uniform_real_distribution<double> target(-1.0, 1.0);
  Joints goal = filled(0.3);
  Joints last_velocity = filled(0.0);
  Joints last_acceleration = filled(0.0);
  Joints last_position = filled(0.3);
  for (int step = 0; step < 20000; ++step) {
    if (step % 33 == 0) {
      for (auto& value : goal) {
        value = target(random);
      }
    }
    ASSERT_TRUE(tracker.step(goal, filled(0.0)));
    for (std::size_t j = 0; j < 7; ++j) {
      const double velocity = (tracker.position()[j] - last_position[j]) / dt;
      const double acceleration = (velocity - last_velocity[j]) / dt;
      const double jerk = (acceleration - last_acceleration[j]) / dt;
      EXPECT_LE(std::abs(velocity), 0.5 + 1e-6);
      EXPECT_LE(std::abs(acceleration), 1.0 + 1e-4);
      EXPECT_LE(std::abs(jerk), 10.0 + 1e-2);
      last_position[j] = tracker.position()[j];
      last_velocity[j] = velocity;
      last_acceleration[j] = acceleration;
    }
  }
  // Holding the target brings the tracker to rest.
  for (int step = 0; step < 4000; ++step) {
    ASSERT_TRUE(tracker.step(goal, filled(0.0)));
  }
  for (std::size_t j = 0; j < 7; ++j) {
    EXPECT_NEAR(tracker.position()[j], goal[j], 1e-6);
    EXPECT_LT(std::abs(tracker.velocity()[j]), 1e-6);
  }
}

TEST(JointRuckigTracker, FollowsMovingTargetWithVelocityFeedforward) {
  JointRuckigTracker tracker;
  const double dt = 0.001;
  tracker.configure(dt, filled(2.0), filled(5.0), filled(50.0));
  tracker.reset(filled(0.0));
  double error = 0.0;
  for (int step = 0; step < 3000; ++step) {
    const double t = (step + 1) * dt;
    const double q = 0.2 * std::sin(2.0 * t);
    const double dq = 0.4 * std::cos(2.0 * t);
    const double ddq = -0.8 * std::sin(2.0 * t);
    ASSERT_TRUE(tracker.step(filled(q), filled(dq), filled(ddq)));
    if (step > 1000) {
      error = std::max(error, std::abs(tracker.position()[0] - q));
    }
  }
  EXPECT_LT(error, 5e-3);
}

TEST(JointRuckigTracker, RejectsInvalidInput) {
  JointRuckigTracker tracker;
  EXPECT_FALSE(tracker.step(filled(0.0), filled(0.0)));
  EXPECT_THROW(tracker.configure(0.001, filled(0.0), filled(1.0), filled(1.0)), std::invalid_argument);
  tracker.configure(0.001, filled(1.0), filled(1.0), filled(1.0));
  EXPECT_FALSE(tracker.step(filled(std::nan("")), filled(0.0)));
  EXPECT_TRUE(tracker.step(filled(0.0), filled(5.0)));
}
