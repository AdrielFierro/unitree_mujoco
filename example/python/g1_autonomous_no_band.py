"""Demostracion autonoma del G1 en MuJoCo para macOS.

La marcha usa la politica preentrenada oficial de ``unitree_rl_gym``. No usa
DDS, banda elastica ni fuerzas externas sobre el robot. La sentadilla, el
saludo y la postura final se controlan con PD dentro del paso fisico.
"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from g1_squat_wave import HIGH_KD, HIGH_KP, NEUTRAL, minimum_jerk, stages


SIM_DT = 0.002
VIEWER_STEPS = 10
FALL_TILT_DEG = 45.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "unitree_robots/g1/scene.xml"
POLICY_PATH = (
    PROJECT_ROOT
    / "third_party/unitree_rl_gym/deploy/pre_train/g1/motion.pt"
)

# Parametros publicados por Unitree para la politica G1 de 12 articulaciones.
POLICY_KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=float)
POLICY_KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=float)
POLICY_DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
POLICY_COMMAND_SCALE = np.array([2.0, 2.0, 0.25])
POLICY_ACTION_SCALE = 0.25
POLICY_CONTROL_DECIMATION = 10
POLICY_OBSERVATIONS = 47
WALK_FORWARD_SECONDS = 0.8
WALK_SETTLE_SECONDS = 1.6


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "G1 autonomo: dos pasos con politica RL, sentadilla y saludo."
        )
    )
    parser.add_argument(
        "--movimiento",
        choices=("rutina", "sentadilla", "saludo"),
        default="rutina",
    )
    parser.add_argument("--repetir", type=int, default=1)
    parser.add_argument(
        "--sin-pasos",
        action="store_true",
        help="omite los dos pasos iniciales y comienza directamente la rutina",
    )
    parser.add_argument(
        "--mantener-segundos",
        type=float,
        help="si se omite, mantiene al G1 parado hasta cerrar la ventana",
    )
    return parser.parse_args()


def gravity_orientation(quaternion):
    """Gravedad proyectada en el marco del torso, igual que en Unitree RL."""
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ]
    )


class AutonomousG1:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        joint_ids = self.model.actuator_trnid[:, 0]
        self.qpos_adr = self.model.jnt_qposadr[joint_ids]
        self.qvel_adr = self.model.jnt_dofadr[joint_ids]
        self.ctrl_low, self.ctrl_high = self.model.actuator_ctrlrange.T
        self.pelvis = self.model.body("pelvis").id
        self.max_tilt_deg = 0.0
        self.step_count = 0
        self.next_tick = time.perf_counter()

        self.data.qpos[self.qpos_adr] = self.model.qpos0[self.qpos_adr]
        mujoco.mj_forward(self.model, self.data)

    def tilt_deg(self):
        vertical = self.data.xmat[self.pelvis][8]
        return float(np.degrees(np.arccos(np.clip(vertical, -1.0, 1.0))))

    def finish_physics_step(self):
        self.step_count += 1
        tilt = self.tilt_deg()
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        if tilt > FALL_TILT_DEG:
            raise RuntimeError(
                f"El G1 supero {FALL_TILT_DEG:.0f} grados de inclinacion "
                f"({tilt:.1f} grados)."
            )

        self.next_tick += SIM_DT
        delay = self.next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_tick = time.perf_counter()

    def advance(self, target):
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.qvel_adr]
        torque = HIGH_KP * (target - q) - HIGH_KD * dq
        self.data.ctrl[:] = np.clip(torque, self.ctrl_low, self.ctrl_high)
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_step(self.model, self.data)
        self.finish_physics_step()

    def advance_policy(self, lower_target):
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.qvel_adr]
        torque = np.zeros(self.model.nu)
        torque[:12] = POLICY_KP * (lower_target - q[:12])
        torque[:12] -= POLICY_KD * dq[:12]
        torque[12:] = HIGH_KP[12:] * (NEUTRAL[12:] - q[12:])
        torque[12:] -= HIGH_KD[12:] * dq[12:]
        self.data.ctrl[:] = np.clip(torque, self.ctrl_low, self.ctrl_high)

        # La politica controla articulaciones: nunca aplica fuerzas al cuerpo.
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_step(self.model, self.data)
        self.finish_physics_step()

    def sync_if_needed(self, viewer):
        if self.step_count % VIEWER_STEPS == 0:
            viewer.sync()

    def transition(self, viewer, start, target, duration):
        steps = max(1, round(duration / SIM_DT))
        for step in range(1, steps + 1):
            if not viewer.is_running():
                return False
            blend = minimum_jerk(step / steps)
            self.advance(start + (target - start) * blend)
            self.sync_if_needed(viewer)
        return True

    def walk_two_steps(self, viewer):
        """Ejecuta un ciclo de la politica oficial y vuelve a postura alta."""
        if torch is None:
            raise RuntimeError(
                "Falta PyTorch. Instala la dependencia con: "
                "../.venv/bin/python -m pip install torch"
            )
        if not POLICY_PATH.exists():
            raise RuntimeError(
                "Falta la politica de Unitree. Ejecuta: "
                "git submodule update --init --recursive"
            )

        policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
        policy.eval()
        action = np.zeros(12, dtype=np.float32)
        lower_target = POLICY_DEFAULT.copy()
        observation = np.zeros(POLICY_OBSERVATIONS, dtype=np.float32)
        start_x = float(self.data.xpos[self.pelvis, 0])

        forward_steps = round(WALK_FORWARD_SECONDS / SIM_DT)
        settle_steps = round(WALK_SETTLE_SECONDS / SIM_DT)
        total_steps = forward_steps + settle_steps

        print("[Fase] Politica RL: avanzando pie derecho e izquierdo")
        for policy_step in range(total_steps):
            if not viewer.is_running():
                return None

            command = np.zeros(3)
            if policy_step < forward_steps:
                command[0] = 0.25
            elif policy_step == forward_steps:
                print("[Fase] Politica RL: frenando y recuperando equilibrio")

            self.advance_policy(lower_target)
            self.sync_if_needed(viewer)

            if policy_step % POLICY_CONTROL_DECIMATION != 0:
                continue

            q = self.data.qpos[self.qpos_adr]
            dq = self.data.qvel[self.qvel_adr]
            phase = (policy_step * SIM_DT % 0.8) / 0.8

            observation[:3] = self.data.qvel[3:6] * 0.25
            observation[3:6] = gravity_orientation(self.data.qpos[3:7])
            observation[6:9] = command * POLICY_COMMAND_SCALE
            observation[9:21] = q[:12] - POLICY_DEFAULT
            observation[21:33] = dq[:12] * 0.05
            observation[33:45] = action
            observation[45:47] = [
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase),
            ]

            with torch.no_grad():
                tensor = torch.from_numpy(observation).unsqueeze(0)
                action = policy(tensor).numpy().squeeze()
            lower_target = action * POLICY_ACTION_SCALE + POLICY_DEFAULT

        print("[Fase] Volviendo a la postura alta")
        current = self.data.qpos[self.qpos_adr].copy()
        if not self.transition(viewer, current, current, 0.50):
            return None
        if not self.transition(viewer, current, NEUTRAL, 2.00):
            return None

        return 100.0 * (float(self.data.xpos[self.pelvis, 0]) - start_x)

    def hold(self, viewer, target, seconds=None):
        end_step = None
        if seconds is not None:
            end_step = self.step_count + round(seconds / SIM_DT)
        while viewer.is_running() and (
            end_step is None or self.step_count < end_step
        ):
            self.advance(target)
            self.sync_if_needed(viewer)


def main():
    args = parse_args()
    if args.repetir < 1:
        raise ValueError("--repetir debe ser al menos 1")
    if args.mantener_segundos is not None and args.mantener_segundos < 0:
        raise ValueError("--mantener-segundos no puede ser negativo")

    robot = AutonomousG1()
    current = robot.model.qpos0[robot.qpos_adr].copy()
    print("G1 autonomo sin banda ni DDS, exclusivo para MuJoCo.")

    with mujoco.viewer.launch_passive(
        robot.model,
        robot.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0.0, 0.0, 0.75]

        print("[Fase] Adoptando postura inicial")
        if not robot.transition(viewer, current, NEUTRAL, 3.0):
            return
        current = NEUTRAL

        if not args.sin_pasos:
            print("[Secuencia] Dos pasos con la politica oficial de Unitree")
            advance_cm = robot.walk_two_steps(viewer)
            if advance_cm is None:
                return
            print(f"[OK] Avance de la marcha: {advance_cm:.1f} cm")
            current = NEUTRAL

        for cycle in range(1, args.repetir + 1):
            print(f"[Ciclo {cycle}/{args.repetir}]")
            for name, duration, target in stages(args.movimiento):
                print(f"[Fase] {name}")
                if not robot.transition(viewer, current, target, duration):
                    return
                current = target

        print("[Fase] Manteniendo al G1 parado sin banda")
        if args.mantener_segundos is None:
            print("Cerra la ventana roja o presiona Control+C para terminar.")
        robot.hold(viewer, NEUTRAL, args.mantener_segundos)

    print(f"[OK] Inclinacion maxima: {robot.max_tilt_deg:.1f} grados")
    print("[OK] Simulador cerrado correctamente")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[OK] Demostracion interrumpida por el usuario")
