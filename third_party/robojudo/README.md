# RoboJuDo G1 policy without gait clock

`policy_wo_gait.pt` was obtained from the `release` branch of
[HansZ8/RoboJuDo](https://github.com/HansZ8/RoboJuDo), path
`assets/models/g1/unitree/policy_wo_gait.pt`.

- Downloaded: 2026-11-02
- SHA-256: `b5861e91bba86cdb35dc10de9764e26336042c817e179ddcebc3427828677414`
- Upstream policy integration: `UnitreeWoGaitPolicy`
- Training project identified by upstream: [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab), task `Unitree-G1-29dof-Velocity`

RoboJuDo is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The local
integration changes the DDS transport, command source, safety checks and
MuJoCo launch flow for this repository's macOS setup. Unitree RL Lab is
licensed under Apache 2.0.

The model is for research and simulation. This integration is intentionally
restricted to DDS domain 1 and the local loopback interface; it must not be
used to command physical hardware.
