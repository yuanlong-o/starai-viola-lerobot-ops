# Episode-recording status

Robot demonstration recording is not authorized by the live-session contract.
The former recorder depended on runtime replacements of installed LeRobot and
StarAI methods; it has been replaced by a fail-closed stub.

The historical 34-episode release is frozen. `viola-ops dataset validate` and
`viola-ops dataset release` read and release those exact bytes without opening
hardware or modifying the dataset.

Camera-only preview and local video utilities remain motor-inert, but their
output is not a LeRobot training release or rollout evidence. A future dataset
version requires a separately reviewed recording authorization and new release
identity.
