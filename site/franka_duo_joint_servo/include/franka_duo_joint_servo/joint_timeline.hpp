#pragma once

// Absolute-step joint timeline shared by the chunk executor and the 1 kHz
// tracker.  Steps never rewind.  A new chunk replaces only steps at or beyond
// the commit boundary and optionally blends into the previously planned
// points, so the running trajectory keeps its time axis across chunks.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace franka_duo_joint_servo {

constexpr std::size_t kJoints = 7;
using Joints = std::array<double, kJoints>;

struct TimelinePoint {
  Joints left{};
  Joints right{};
  double left_gripper{0.0};
  double right_gripper{0.0};
};

struct ArmSample {
  Joints position{};
  Joints velocity{};
  Joints acceleration{};
};

struct DualSample {
  ArmSample left;
  ArmSample right;
  double left_gripper{0.0};
  double right_gripper{0.0};
  // True when the sample is a held endpoint or a gap: velocity is zero and
  // the target does not move until new steps arrive.
  bool holding{true};
};

class JointTimeline {
 public:
  explicit JointTimeline(double step_period_s) : period_(step_period_s) {
    if (!std::isfinite(step_period_s) || step_period_s <= 0.0) {
      throw std::invalid_argument("step period must be finite and positive");
    }
  }

  double period() const { return period_; }
  bool empty() const { return points_.empty(); }
  std::size_t size() const { return points_.size(); }

  std::optional<std::int64_t> first_step() const {
    if (points_.empty()) {
      return std::nullopt;
    }
    return points_.begin()->first;
  }

  std::optional<std::int64_t> last_step() const {
    if (points_.empty()) {
      return std::nullopt;
    }
    return points_.rbegin()->first;
  }

  const TimelinePoint* at(std::int64_t step) const {
    const auto it = points_.find(step);
    return it == points_.end() ? nullptr : &it->second;
  }

  // rows[i] is the target for step start_step + i.  Rows before commit_step
  // are skipped and their count returned.  Every previously planned point at
  // or beyond commit_step is discarded. With blending enabled, the commit
  // point stays on the old plan and a quintic weight reaches the new plan
  // after blend_steps intervals. An exhausted old plan contributes its held
  // endpoint, so late chunks get the same transition as overlapping chunks.
  std::size_t replace(
      std::int64_t start_step,
      const std::vector<TimelinePoint>& rows,
      std::int64_t commit_step,
      std::size_t blend_steps) {
    std::size_t index = 0;
    while (index < rows.size() &&
           start_step + static_cast<std::int64_t>(index) < commit_step) {
      ++index;
    }
    const std::size_t skipped = index;
    if (index >= rows.size()) {
      return skipped;
    }
    const JointTimeline old = *this;
    points_.erase(points_.lower_bound(commit_step), points_.end());
    tangents_.erase(tangents_.lower_bound(commit_step), tangents_.end());
    const std::size_t intervals = std::min(blend_steps, rows.size() - index - 1U);
    // Keep a held anchor immediately before a late chunk. Otherwise sample()
    // would cross a gap by jumping from the old endpoint to the first new row.
    if (!old.empty() && intervals > 0U && at(commit_step - 1) == nullptr) {
      const auto held = old.sample(static_cast<double>(commit_step - 1));
      TimelinePoint anchor;
      anchor.left = held->left.position;
      anchor.right = held->right.position;
      anchor.left_gripper = held->left_gripper;
      anchor.right_gripper = held->right_gripper;
      points_[commit_step - 1] = anchor;
    }
    const auto first_blend_step = start_step + static_cast<std::int64_t>(index);
    if (!old.empty() && intervals > 0U) {
      const auto boundary = old.sample(static_cast<double>(first_blend_step));
      auto& tangent = tangents_[first_blend_step];
      for (std::size_t j = 0; j < kJoints; ++j) {
        tangent.first[j] = boundary->left.velocity[j] * period_;
        tangent.second[j] = boundary->right.velocity[j] * period_;
      }
    }
    for (; index < rows.size(); ++index) {
      const std::int64_t step = start_step + static_cast<std::int64_t>(index);
      TimelinePoint point = rows[index];
      const auto previous = old.sample(static_cast<double>(step));
      if (previous.has_value() && intervals > 0U &&
          step - first_blend_step < static_cast<std::int64_t>(intervals)) {
        const double u = static_cast<double>(step - first_blend_step) / intervals;
        const double weight = u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
        for (std::size_t j = 0; j < kJoints; ++j) {
          point.left[j] = (1.0 - weight) * previous->left.position[j] + weight * point.left[j];
          point.right[j] = (1.0 - weight) * previous->right.position[j] + weight * point.right[j];
        }
      }
      points_[step] = point;
    }
    return skipped;
  }

  void trim_before(std::int64_t step) {
    points_.erase(points_.begin(), points_.lower_bound(step));
    tangents_.erase(tangents_.begin(), tangents_.lower_bound(step));
  }

  // Cubic Hermite (Catmull-Rom) sample at a fractional step.  Endpoints and
  // gaps are held with zero velocity; the last segment decelerates into the
  // hold so a chunk that ends without a successor comes to rest smoothly.
  std::optional<DualSample> sample(double step) const {
    if (points_.empty() || !std::isfinite(step)) {
      return std::nullopt;
    }
    DualSample result;
    const auto hold = [&](const TimelinePoint& point) {
      result.left.position = point.left;
      result.right.position = point.right;
      result.left.velocity.fill(0.0);
      result.right.velocity.fill(0.0);
      result.left.acceleration.fill(0.0);
      result.right.acceleration.fill(0.0);
      result.left_gripper = point.left_gripper;
      result.right_gripper = point.right_gripper;
      result.holding = true;
      return result;
    };
    const auto first = points_.begin();
    const auto last = points_.rbegin();
    if (step <= static_cast<double>(first->first)) {
      return hold(first->second);
    }
    if (step >= static_cast<double>(last->first)) {
      return hold(last->second);
    }
    const auto k = static_cast<std::int64_t>(std::floor(step));
    const double u = step - static_cast<double>(k);
    const TimelinePoint* p1 = at(k);
    const TimelinePoint* p2 = at(k + 1);
    if (p1 == nullptr || p2 == nullptr) {
      auto it = points_.upper_bound(k);
      --it;
      return hold(it->second);
    }
    const TimelinePoint* p0 = at(k - 1);
    const TimelinePoint* p3 = at(k + 2);
    const auto tangent1 = tangents_.find(k);
    const auto tangent2 = tangents_.find(k + 1);
    const double h00 = 2.0 * u * u * u - 3.0 * u * u + 1.0;
    const double h10 = u * u * u - 2.0 * u * u + u;
    const double h01 = -2.0 * u * u * u + 3.0 * u * u;
    const double h11 = u * u * u - u * u;
    const double d00 = 6.0 * u * u - 6.0 * u;
    const double d10 = 3.0 * u * u - 4.0 * u + 1.0;
    const double d01 = -6.0 * u * u + 6.0 * u;
    const double d11 = 3.0 * u * u - 2.0 * u;
    const double a00 = 12.0 * u - 6.0;
    const double a10 = 6.0 * u - 4.0;
    const double a01 = -12.0 * u + 6.0;
    const double a11 = 6.0 * u - 2.0;
    const auto interpolate = [&](Joints TimelinePoint::*member, ArmSample& out) {
      for (std::size_t j = 0; j < kJoints; ++j) {
        const double q1 = (p1->*member)[j];
        const double q2 = (p2->*member)[j];
        // Freeze the old plan's derivative at a replacement boundary. New
        // future points must not change the segment before that boundary.
        const bool left = member == &TimelinePoint::left;
        const double m1 = tangent1 != tangents_.end()
            ? (left ? tangent1->second.first[j] : tangent1->second.second[j])
            : (p0 == nullptr ? 0.0 : 0.5 * (q2 - (p0->*member)[j]));
        const double m2 = tangent2 != tangents_.end()
            ? (left ? tangent2->second.first[j] : tangent2->second.second[j])
            : (p3 == nullptr ? 0.0 : 0.5 * ((p3->*member)[j] - q1));
        out.position[j] = h00 * q1 + h10 * m1 + h01 * q2 + h11 * m2;
        out.velocity[j] = (d00 * q1 + d10 * m1 + d01 * q2 + d11 * m2) / period_;
        out.acceleration[j] =
            (a00 * q1 + a10 * m1 + a01 * q2 + a11 * m2) / (period_ * period_);
      }
    };
    interpolate(&TimelinePoint::left, result.left);
    interpolate(&TimelinePoint::right, result.right);
    result.left_gripper = p1->left_gripper;
    result.right_gripper = p1->right_gripper;
    result.holding = false;
    return result;
  }

 private:
  double period_;
  std::map<std::int64_t, TimelinePoint> points_;
  // Derivatives in joint radians per timeline step, fixed at splice knots.
  std::map<std::int64_t, std::pair<Joints, Joints>> tangents_;
};

}  // namespace franka_duo_joint_servo
