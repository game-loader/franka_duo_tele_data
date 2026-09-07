#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "franka_duo_joint_servo/joint_timeline.hpp"

using franka_duo_joint_servo::JointTimeline;
using franka_duo_joint_servo::TimelinePoint;

namespace {

TimelinePoint point(double value) {
  TimelinePoint p;
  p.left.fill(value);
  p.right.fill(-value);
  return p;
}

std::vector<TimelinePoint> ramp(int count, double start, double slope) {
  std::vector<TimelinePoint> rows;
  for (int i = 0; i < count; ++i) {
    rows.push_back(point(start + slope * i));
  }
  return rows;
}

}  // namespace

TEST(JointTimeline, ReplaceSkipsRowsBeforeCommitAndKeepsTimeAxis) {
  JointTimeline timeline(1.0 / 30.0);
  EXPECT_EQ(timeline.replace(0, ramp(8, 0.0, 0.1), 0, 0), 0U);
  ASSERT_TRUE(timeline.last_step().has_value());
  EXPECT_EQ(*timeline.last_step(), 7);
  // Late chunk starting at step 2 while the commit boundary is step 5.
  EXPECT_EQ(timeline.replace(2, ramp(8, 1.0, 0.0), 5, 0), 3U);
  EXPECT_DOUBLE_EQ(timeline.at(4)->left[0], 0.4);
  EXPECT_DOUBLE_EQ(timeline.at(5)->left[0], 1.0);
  EXPECT_EQ(*timeline.last_step(), 9);
  // A chunk entirely before the commit boundary changes nothing.
  EXPECT_EQ(timeline.replace(0, ramp(3, 9.0, 0.0), 5, 0), 3U);
  EXPECT_DOUBLE_EQ(timeline.at(5)->left[0], 1.0);
}

TEST(JointTimeline, BlendPreservesCommitPointAndUsesQuinticWeight) {
  JointTimeline timeline(1.0 / 30.0);
  timeline.replace(0, ramp(10, 0.0, 0.0), 0, 0);
  timeline.replace(0, ramp(10, 1.0, 0.0), 4, 3);
  EXPECT_DOUBLE_EQ(timeline.at(3)->left[0], 0.0);
  EXPECT_NEAR(timeline.at(4)->left[0], 0.0, 1e-12);
  EXPECT_NEAR(timeline.at(5)->left[0], 17.0 / 81.0, 1e-12);
  EXPECT_NEAR(timeline.at(6)->left[0], 64.0 / 81.0, 1e-12);
  EXPECT_DOUBLE_EQ(timeline.at(7)->left[0], 1.0);
  EXPECT_NEAR(timeline.at(5)->right[0], -17.0 / 81.0, 1e-12);
}

TEST(JointTimeline, ReplacementPreservesTheEntireSegmentBeforeCommit) {
  JointTimeline timeline(0.1);
  timeline.replace(0, ramp(12, 0.0, 0.01), 0, 0);
  const auto before = timeline.sample(3.7);
  const auto knot = timeline.sample(4.0);
  timeline.replace(4, ramp(12, 0.12, 0.02), 4, 4);
  const auto after = timeline.sample(3.7);
  EXPECT_NEAR(after->left.position[0], before->left.position[0], 1e-12);
  EXPECT_NEAR(after->left.velocity[0], before->left.velocity[0], 1e-12);
  EXPECT_NEAR(timeline.sample(4.0)->left.velocity[0], knot->left.velocity[0], 1e-12);
}

TEST(JointTimeline, LateChunkBlendsFromHeldEndpointWithoutGapJump) {
  JointTimeline timeline(0.1);
  timeline.replace(0, ramp(3, 0.0, 0.01), 0, 0);
  timeline.replace(10, ramp(8, 0.1, 0.01), 10, 4);
  EXPECT_DOUBLE_EQ(timeline.at(10)->left[0], 0.02);
  EXPECT_NEAR(timeline.sample(9.5)->left.position[0], 0.02, 1e-12);
  EXPECT_NEAR(timeline.sample(10.0)->left.velocity[0], 0.0, 1e-12);
  EXPECT_GT(timeline.sample(10.5)->left.position[0], 0.02);
  EXPECT_DOUBLE_EQ(timeline.at(14)->left[0], 0.14);
}

TEST(JointTimeline, SampleIsSmoothAndHoldsEndpoints) {
  JointTimeline timeline(0.1);
  timeline.replace(0, ramp(6, 0.0, 1.0), 0, 0);
  const auto before = timeline.sample(-1.0);
  ASSERT_TRUE(before.has_value());
  EXPECT_TRUE(before->holding);
  EXPECT_DOUBLE_EQ(before->left.position[0], 0.0);
  EXPECT_DOUBLE_EQ(before->left.velocity[0], 0.0);
  const auto after = timeline.sample(9.0);
  EXPECT_TRUE(after->holding);
  EXPECT_DOUBLE_EQ(after->left.position[0], 5.0);
  // Interior of a linear ramp: exact position and velocity slope/period.
  const auto mid = timeline.sample(2.5);
  EXPECT_FALSE(mid->holding);
  EXPECT_NEAR(mid->left.position[0], 2.5, 1e-12);
  EXPECT_NEAR(mid->left.velocity[0], 10.0, 1e-9);
  EXPECT_NEAR(mid->right.velocity[0], -10.0, 1e-9);
  EXPECT_NEAR(mid->left.acceleration[0], 0.0, 1e-6);
  // Continuity across knots.
  const auto knot_left = timeline.sample(3.0 - 1e-9);
  const auto knot_right = timeline.sample(3.0 + 1e-9);
  EXPECT_NEAR(knot_left->left.position[0], knot_right->left.position[0], 1e-6);
  EXPECT_NEAR(knot_left->left.velocity[0], knot_right->left.velocity[0], 1e-5);
}

TEST(JointTimeline, GapHoldsNewestPointAndTrimWorks) {
  JointTimeline timeline(0.1);
  timeline.replace(0, ramp(3, 0.0, 1.0), 0, 0);
  timeline.replace(6, ramp(2, 9.0, 0.0), 6, 0);
  const auto gap = timeline.sample(4.5);
  EXPECT_TRUE(gap->holding);
  EXPECT_DOUBLE_EQ(gap->left.position[0], 2.0);
  timeline.trim_before(6);
  EXPECT_EQ(*timeline.first_step(), 6);
  EXPECT_EQ(timeline.size(), 2U);
  EXPECT_FALSE(timeline.sample(std::nan("")).has_value());
}
