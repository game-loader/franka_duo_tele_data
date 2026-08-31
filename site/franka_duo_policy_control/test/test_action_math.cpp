#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "franka_duo_policy_control/action_math.hpp"

namespace franka_duo_policy_control {
namespace {

TEST(ActionMath, ConvertsRowRot6dToProperRotation) {
  const auto rotation = rot6dRowsToMatrix({0.0, 2.0, 0.0, -3.0, 0.0, 0.0});

  EXPECT_TRUE(isRotation(rotation));
  EXPECT_NEAR(rotation(0, 1), 1.0, 1e-9);
  EXPECT_NEAR(rotation(1, 0), -1.0, 1e-9);
  EXPECT_NEAR(rotation(2, 2), 1.0, 1e-9);
}

TEST(ActionMath, UsesTrainingArmBaseTransforms) {
  const auto left = trainingMidpointFromLeftArmBase();
  const auto right = trainingMidpointFromRightArmBase();

  EXPECT_TRUE(isHomogeneous(left));
  EXPECT_TRUE(isHomogeneous(right));
  EXPECT_TRUE(isRotation(left.block<3, 3>(0, 0)));
  EXPECT_TRUE(isRotation(right.block<3, 3>(0, 0)));
  EXPECT_NEAR(left(1, 3), 0.05018, 1e-9);
  EXPECT_NEAR(right(1, 3), -0.05018, 1e-9);
  EXPECT_NEAR(left(0, 0), right(0, 0), 1e-7);
  EXPECT_NEAR(left(0, 1), -right(0, 1), 1e-7);
}

TEST(ActionMath, ConvertsRowMajorMatrixToFrankasColumnMajorLayout) {
  const auto matrix = matrixFromRowMajor(
      {1.0, 2.0, 3.0, 4.0,
       5.0, 6.0, 7.0, 8.0,
       9.0, 10.0, 11.0, 12.0,
       13.0, 14.0, 15.0, 16.0});
  const std::array<double, 16> expected{
      1.0, 5.0, 9.0, 13.0,
      2.0, 6.0, 10.0, 14.0,
      3.0, 7.0, 11.0, 15.0,
      4.0, 8.0, 12.0, 16.0};

  EXPECT_EQ(matrixToColumnMajor(matrix), expected);
}

TEST(ActionMath, AppliesFixedTransformInTheTrainingDirection) {
  Matrix4d midpoint_pose = Matrix4d::Identity();
  midpoint_pose(0, 3) = 0.7;
  midpoint_pose(1, 3) = -0.1;
  midpoint_pose(2, 3) = 0.2;

  const auto arm_pose = trainingMidpointFromLeftArmBase().inverse() * midpoint_pose;
  const auto reconstructed = trainingMidpointFromLeftArmBase() * arm_pose;

  EXPECT_TRUE(isHomogeneous(arm_pose));
  EXPECT_TRUE(isHomogeneous(reconstructed));
  EXPECT_NEAR((reconstructed - midpoint_pose).norm(), 0.0, 1e-9);
}

TEST(ActionMath, PassesThroughSiteGripperOpenFraction) {
  EXPECT_FLOAT_EQ(gripperOpenFractionToSiteTarget(0.0F), 0.0F);
  EXPECT_FLOAT_EQ(gripperOpenFractionToSiteTarget(0.8F), 0.8F);
  EXPECT_FLOAT_EQ(gripperOpenFractionToSiteTarget(1.0F), 1.0F);
}

TEST(ActionMath, RejectsInvalidGripperOpenFraction) {
  EXPECT_THROW(gripperOpenFractionToSiteTarget(-0.01F), std::invalid_argument);
  EXPECT_THROW(gripperOpenFractionToSiteTarget(1.01F), std::invalid_argument);
  EXPECT_THROW(
      gripperOpenFractionToSiteTarget(std::numeric_limits<float>::quiet_NaN()),
      std::invalid_argument);
}

}  // namespace
}  // namespace franka_duo_policy_control
