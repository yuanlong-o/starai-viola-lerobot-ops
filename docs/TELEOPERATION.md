# Teleoperation status

Leader/follower teleoperation is not authorized by the live-session contract.
The legacy shell and Python entrypoints exit with code 64 before loading a
plugin or checking a device.

This is deliberate. The installed StarAI 0.0.4 classes do not expose a public
safe-connect option, and the former keep-pose implementation replaced installed
class methods at runtime. Runtime patching is no longer allowed.

Do not run stock `lerobot-teleoperate` as a workaround. A future teleoperation
workflow needs its own reviewed authorization, public safe-connect adapter,
limits, evidence contract, and explicit operator gate.
