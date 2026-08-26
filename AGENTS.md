# Agent Instructions

This repository has one narrow purpose: collect real Franka Duo ROS 2 topics as
raw MCAP episodes and run exported policy bundles on live observations while
recording their raw eval provenance.

- Do not add training pipelines, simulators, Docker runtimes, benchmark suites,
  paper tooling, or ARA artifacts.
- Keep ROS 2 and camera drivers host-managed; do not vendor them as Python
  dependencies.
- Keep live capture on rosbag2 MCAP with the `zstd_fast` storage preset. Do not
  decode, synchronize, resample, aggregate, normalize, or run FK online.
- Preserve the raw TMR topic contract, the offline-derived 16D joint-space
  contract, and the 20D Cartesian evaluation contract unless the user
  explicitly approves a versioned change.
- Never substitute measured joints for applied/desired action targets.
- Evaluation must remain dry-run by default. Robot publication requires both
  explicit safety gates and must target a site-owned relay, not a controller
  command topic.
- Keep base dependencies small. Put optional LeRobot policy compatibility and
  development tools in separate dependency groups.
- Run focused tests, `ruff check`, TOML parsing, and `bash -n scripts/*.sh`
  before committing changes.
