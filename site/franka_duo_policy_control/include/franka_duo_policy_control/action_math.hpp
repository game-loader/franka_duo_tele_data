#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>

namespace franka_duo_policy_control {

using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;

inline Matrix4d matrixFromRowMajor(const std::array<double, 16>& values) {
  Matrix4d result;
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
          values[row * 4 + column];
    }
  }
  return result;
}

inline Matrix4d trainingMidpointFromLeftArmBase() {
  // T_midpoint_from_arm_base from the same mobile_fr3_duo_v0_2 USD asset
  // used by the offline converter.  The mirrored rotations are intentional.
  return matrixFromRowMajor({
      0.8809676, 0.40120238, 0.25086382, 0.0,
      -0.44015086, 0.50023556, 0.74567527, 0.05018,
      0.17367569, -0.7673337, 0.6172809, 0.0,
      0.0, 0.0, 0.0, 1.0});
}

inline Matrix4d trainingMidpointFromRightArmBase() {
  // T_midpoint_from_arm_base from the same training geometry as the left arm.
  return matrixFromRowMajor({
      0.8809676, -0.40120238, 0.25086382, 0.0,
      0.44015086, 0.50023556, -0.74567527, -0.05018,
      0.17367569, 0.7673337, 0.6172809, 0.0,
      0.0, 0.0, 0.0, 1.0});
}

inline bool finite(const Matrix4d& value) {
  return value.allFinite();
}

inline bool finite(const Matrix3d& value) {
  return value.allFinite();
}

inline std::array<double, 16> matrixToColumnMajor(const Matrix4d& matrix) {
  if (!finite(matrix)) {
    throw std::invalid_argument("Cartesian pose matrix contains non-finite values");
  }
  std::array<double, 16> result{};
  for (std::size_t column = 0; column < 4; ++column) {
    for (std::size_t row = 0; row < 4; ++row) {
      result[column * 4 + row] =
          matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
    }
  }
  return result;
}

inline bool isHomogeneous(const Matrix4d& matrix, double tolerance = 1e-8) {
  if (!finite(matrix)) {
    return false;
  }
  const Eigen::RowVector4d expected(0.0, 0.0, 0.0, 1.0);
  return (matrix.row(3) - expected).norm() <= tolerance;
}

inline bool isRotation(const Matrix3d& matrix, double tolerance = 1e-6) {
  if (!finite(matrix)) {
    return false;
  }
  return (matrix * matrix.transpose() - Matrix3d::Identity()).norm() <= tolerance &&
         std::abs(matrix.determinant() - 1.0) <= tolerance;
}

inline bool isFiniteQuaternion(const Eigen::Quaterniond& quaternion) {
  return quaternion.coeffs().allFinite() && quaternion.norm() > 1e-8;
}

inline Matrix3d rot6dRowsToMatrix(const std::array<double, 6>& values) {
  Eigen::Vector3d first(values[0], values[1], values[2]);
  Eigen::Vector3d second(values[3], values[4], values[5]);
  const double first_norm = first.norm();
  if (!std::isfinite(first_norm) || first_norm < 1e-8) {
    throw std::invalid_argument("rot6d first row is degenerate");
  }
  first /= first_norm;
  second -= first.dot(second) * first;
  const double second_norm = second.norm();
  if (!std::isfinite(second_norm) || second_norm < 1e-8) {
    throw std::invalid_argument("rot6d rows are collinear");
  }
  second /= second_norm;
  Eigen::Vector3d third = first.cross(second);
  if (!third.allFinite() || third.norm() < 1e-8) {
    throw std::invalid_argument("rot6d produced a degenerate rotation");
  }
  third.normalize();

  Matrix3d result;
  result.row(0) = first.transpose();
  result.row(1) = second.transpose();
  result.row(2) = third.transpose();
  if (!isRotation(result)) {
    throw std::invalid_argument("rot6d did not produce a valid rotation");
  }
  return result;
}

inline Matrix4d makeTransform(const Matrix3d& rotation, const Eigen::Vector3d& translation) {
  if (!isRotation(rotation) || !translation.allFinite()) {
    throw std::invalid_argument("invalid Cartesian transform components");
  }
  Matrix4d result = Matrix4d::Identity();
  result.block<3, 3>(0, 0) = rotation;
  result.block<3, 1>(0, 3) = translation;
  return result;
}

inline Eigen::Quaterniond quaternionFromRotation(const Matrix3d& rotation) {
  if (!isRotation(rotation)) {
    throw std::invalid_argument("invalid rotation matrix");
  }
  Eigen::Quaterniond result(rotation);
  result.normalize();
  if (!result.coeffs().allFinite()) {
    throw std::invalid_argument("rotation produced a non-finite quaternion");
  }
  return result;
}

inline Matrix4d checkedTransform(const std::array<double, 16>& values, const char* name) {
  const Matrix4d matrix = matrixFromRowMajor(values);
  if (!isHomogeneous(matrix)) {
    throw std::invalid_argument(std::string(name) + " must be a finite row-major homogeneous matrix");
  }
  const Matrix3d rotation = matrix.block<3, 3>(0, 0);
  if (!isRotation(rotation)) {
    throw std::invalid_argument(std::string(name) + " has an invalid rotation");
  }
  return matrix;
}

inline double rotationDistance(const Eigen::Quaterniond& first, const Eigen::Quaterniond& second) {
  if (!isFiniteQuaternion(first) || !isFiniteQuaternion(second)) {
    throw std::invalid_argument("rotation distance requires finite non-zero quaternions");
  }
  Eigen::Quaterniond a = first;
  Eigen::Quaterniond b = second;
  a.normalize();
  b.normalize();
  const double cosine = std::clamp(std::abs(a.dot(b)), 0.0, 1.0);
  return 2.0 * std::acos(cosine);
}

inline float gripperOpenFractionToSiteTarget(float open_fraction) {
  if (!std::isfinite(open_fraction) || open_fraction < 0.0F || open_fraction > 1.0F) {
    throw std::invalid_argument("gripper open fraction must be finite and in [0, 1]");
  }
  return open_fraction;
}

}  // namespace franka_duo_policy_control
