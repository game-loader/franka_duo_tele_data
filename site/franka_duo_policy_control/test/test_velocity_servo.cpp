#include <limits>
#include <random>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "franka_duo_policy_control/velocity_servo.hpp"

using franka_duo_policy_control::VelocityServo;

TEST(VelocityServo, RetargetingPreservesCommandDerivativesAndBounds) {
  VelocityServo servo;
  servo.configure(0.05, 0.10, 0.5);
  std::mt19937 random(42);
  std::uniform_real_distribution<double> request(-2.0, 2.0);
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_acceleration = Eigen::Vector3d::Zero();
  for (int step = 0; step < 30000; ++step) {
    if (step % 33 == 0) {
      target = Eigen::Vector3d(request(random), request(random), request(random));
    }
    Eigen::Vector3d displacement;
    ASSERT_TRUE(servo.step(target, displacement));
    const Eigen::Vector3d velocity = displacement / VelocityServo::kPeriod;
    const Eigen::Vector3d acceleration = (velocity - last_velocity) / VelocityServo::kPeriod;
    const Eigen::Vector3d jerk = (acceleration - last_acceleration) / VelocityServo::kPeriod;
    EXPECT_LE(velocity.norm(), 0.05 + 1e-8);
    EXPECT_LE(acceleration.norm(), 0.10 + 1e-7);
    EXPECT_LE(jerk.norm(), 0.5 + 1e-5);
    last_velocity = velocity;
    last_acceleration = acceleration;
  }
  for (int step = 0; step < 2000; ++step) {
    Eigen::Vector3d displacement;
    ASSERT_TRUE(servo.step(Eigen::Vector3d::Zero(), displacement));
    const Eigen::Vector3d velocity = displacement / VelocityServo::kPeriod;
    const Eigen::Vector3d acceleration = (velocity - last_velocity) / VelocityServo::kPeriod;
    EXPECT_LE(acceleration.norm(), 0.10 + 1e-7);
    EXPECT_LE(((acceleration - last_acceleration) / VelocityServo::kPeriod).norm(), 0.5 + 1e-5);
    last_velocity = velocity;
    last_acceleration = acceleration;
  }
  EXPECT_LT(servo.velocity().norm(), 1e-10);
  EXPECT_LT(servo.acceleration().norm(), 1e-10);
}

TEST(VelocityServo, QuaternionCommandsRemainContinuousWhenRotationAxisChanges) {
  VelocityServo servo;
  servo.configure(0.3, 0.5, 2.0);
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d last_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_acceleration = Eigen::Vector3d::Zero();
  for (int step = 0; step < 10000; ++step) {
    Eigen::Vector3d target = Eigen::Vector3d::Zero();
    target[(step / 100) % 3] = (step / 500) % 2 == 0 ? 2.0 : -2.0;
    Eigen::Vector3d displacement;
    ASSERT_TRUE(servo.step(target, displacement));
    const auto previous = orientation;
    if (displacement.norm() > 1e-15) {
      orientation = (Eigen::Quaterniond(Eigen::AngleAxisd(displacement.norm(), displacement.normalized())) *
                     orientation).normalized();
    }
    Eigen::AngleAxisd delta(orientation * previous.conjugate());
    const Eigen::Vector3d velocity = delta.axis() * delta.angle() / VelocityServo::kPeriod;
    const Eigen::Vector3d acceleration = (velocity - last_velocity) / VelocityServo::kPeriod;
    EXPECT_LE(velocity.norm(), 0.3 + 1e-8);
    EXPECT_LE(acceleration.norm(), 0.5 + 1e-7);
    EXPECT_LE(((acceleration - last_acceleration) / VelocityServo::kPeriod).norm(), 2.0 + 1e-4);
    last_velocity = velocity;
    last_acceleration = acceleration;
  }
}

TEST(VelocityServo, InvalidInputsAndStationaryReset) {
  VelocityServo servo;
  EXPECT_THROW(servo.configure(0, 1, 1), std::invalid_argument);
  servo.configure(0.05, 0.1, 0.5);
  Eigen::Vector3d displacement;
  EXPECT_FALSE(servo.step(Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()), displacement));
  ASSERT_TRUE(servo.step(Eigen::Vector3d::Zero(), displacement));
  EXPECT_EQ(displacement.norm(), 0.0);
}
