// Offline only: no robot state subscriptions or command publishers.
// Input per line: xyz(3), rotation columns(6), seed q(7).
// Output: OK q(7) position_error_m angle_error_rad max_joint_delta_rad; or FAIL reason.
#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <Eigen/Geometry>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(f), {});
}
int main(int argc, char** argv) {
  if (argc != 4) { std::cerr << "usage: labs_fr3_ik arm.urdf arm.srdf left|right\n"; return 2; }
  const std::string side = argv[3], group_name = side + "_arm", tip = side + "_fr3_link8";
  if (side != "left" && side != "right") return 2;
  rclcpp::init(0, nullptr);
  try {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    options.parameter_overrides({
      rclcpp::Parameter("robot_description", readFile(argv[1])),
      rclcpp::Parameter("robot_description_semantic", readFile(argv[2])),
      rclcpp::Parameter("robot_description_kinematics."+group_name+".kinematics_solver", "kdl_kinematics_plugin/KDLKinematicsPlugin"),
      rclcpp::Parameter("robot_description_kinematics."+group_name+".kinematics_solver_timeout", 0.05),
      rclcpp::Parameter("robot_description_kinematics."+group_name+".kinematics_solver_search_resolution", 0.005)
    });
    auto node = std::make_shared<rclcpp::Node>("labs_fr3_ik_"+side, options);
    robot_model_loader::RobotModelLoader loader(node);
    auto model = loader.getModel();
    if (!model) throw std::runtime_error("Robot model unavailable");
    const auto* group = model->getJointModelGroup(group_name);
    if (!group || group->getVariableCount()!=7 || !group->getSolverInstance())
      throw std::runtime_error("Seven-joint MoveIt KDL group unavailable");
    moveit::core::RobotState state(model);
    state.setToDefaultValues();
    std::string line;
    std::cout << std::setprecision(17);
    while (std::getline(std::cin,line)) {
      std::istringstream input(line);
      double v[16]; bool valid=true;
      for (double& x:v) if (!(input>>x) || !std::isfinite(x)) { valid=false; break; }
      if (!valid) { std::cout << "FAIL invalid_input\n" << std::flush; continue; }
      Eigen::Vector3d first(v[3],v[4],v[5]), second(v[6],v[7],v[8]);
      if (first.norm()<1e-8) { std::cout << "FAIL rotation\n" << std::flush; continue; }
      first.normalize(); second-=first.dot(second)*first;
      if (second.norm()<1e-8) { std::cout << "FAIL rotation\n" << std::flush; continue; }
      second.normalize();
      Eigen::Isometry3d local=Eigen::Isometry3d::Identity();
      local.translation()=Eigen::Vector3d(v[0],v[1],v[2]);
      local.linear().col(0)=first; local.linear().col(1)=second; local.linear().col(2)=first.cross(second);
      std::vector<double> seed(v+9,v+16), solution;
      state.setJointGroupPositions(group,seed); state.update();
      const Eigen::Isometry3d target=state.getGlobalLinkTransform(side+"_fr3_link0")*local;
      const std::vector<double> consistency(7,0.5);
      // Same seed/consistency/bounds/jump checks as policy_chunk_jtc_stream.
      if (!state.setFromIK(group,target,tip,consistency,0.05)) {
        std::cout << "FAIL ik\n" << std::flush; continue;
      }
      state.update(); state.copyJointGroupPositions(group,solution);
      if (!state.satisfiesBounds(group)) { std::cout << "FAIL bounds\n" << std::flush; continue; }
      double delta=0;
      for (size_t i=0;i<7;++i) delta=std::max(delta,std::abs(solution[i]-seed[i]));
      if (delta>0.5) { std::cout << "FAIL discontinuity\n" << std::flush; continue; }
      const auto achieved=state.getGlobalLinkTransform(tip);
      const double pos=(target.translation()-achieved.translation()).norm();
      const double angle=Eigen::AngleAxisd(target.linear().transpose()*achieved.linear()).angle();
      if (pos>1e-4 || angle>1e-3) { std::cout << "FAIL residual\n" << std::flush; continue; }
      std::cout << "OK"; for(double x:solution) std::cout << ' ' << x;
      std::cout << ' ' << pos << ' ' << angle << ' ' << delta << '\n' << std::flush;
    }
    rclcpp::shutdown(); return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; rclcpp::shutdown(); return 1; }
}
