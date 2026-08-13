# ROS 2 / MoveIt status

ROS hardware control is retired for this benchmark. It is outside the
`rollout_session` evidence contract and would be an alternate motor path, so
`ros2/run_viola_driver.sh` exits before device access.

The earlier ROS setup depended on applying a repository patch to third-party
source. That setup path is also disabled because this project now uses public
APIs and repo-owned adapters only.

Motor-inert simulation may be retained for historical development, but it is
not policy verification, shadow evidence, physical evidence, or readiness.
