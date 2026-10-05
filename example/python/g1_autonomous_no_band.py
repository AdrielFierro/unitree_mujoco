"""Rutina autonoma del G1 en MuJoCo, sin DDS ni banda elastica.

El control PD se calcula en cada paso fisico de 2 ms. Es una demostracion
educativa exclusiva para simulacion: las ganancias son demasiado altas para
un robot real.
"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from g1_squat_wave import HIGH_KD, HIGH_KP, NEUTRAL, minimum_jerk, stages


SIM_DT = 0.002
VIEWER_STEPS = 10  # 50 FPS
FALL_TILT_DEG = 45.0
MODEL_PATH = Path(__file__).resolve().parents[2] / "unitree_robots/g1/scene.xml"

# Indices de los actuadores de las piernas en el modelo oficial G1 (29 DoF).
LEFT_HIP_PITCH = 0
LEFT_HIP_ROLL = 1
LEFT_KNEE = 3
LEFT_ANKLE_PITCH = 4
LEFT_ANKLE_ROLL = 5
RIGHT_HIP_PITCH = 6
RIGHT_HIP_ROLL = 7
RIGHT_KNEE = 9
RIGHT_ANKLE_PITCH = 10
RIGHT_ANKLE_ROLL = 11


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "G1 autonomo: dos pasos, sentadilla y saludo sin banda elastica."
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


def step_pose(weight_shift, swing_leg=None):
    """Crea una referencia cuasiestatica para dar un paso corto en el lugar.

    MuJoCo no incluye un controlador de balance/caminata para este ejemplo. La
    amplitud se mantiene deliberadamente pequena para conservar ambos apoyos
    cerca del poligono estable y permitir la rutina posterior.
    """
    target = NEUTRAL.copy()
    target[[LEFT_HIP_ROLL, LEFT_ANKLE_ROLL]] = [weight_shift, -weight_shift]
    target[[RIGHT_HIP_ROLL, RIGHT_ANKLE_ROLL]] = [weight_shift, -weight_shift]

    if swing_leg == "right":
        target[[RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH]] = [
            -0.30,
            0.40,
            -0.10,
        ]
    elif swing_leg == "left":
        target[[LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH]] = [
            -0.30,
            0.40,
            -0.10,
        ]
    return target


def walking_stages():
    """Dos pasos lentos en el lugar: derecho y luego izquierdo."""
    shift_left = step_pose(-0.08)
    shift_right = step_pose(0.08)
    return [
        ("Paso 1/2: trasladando el peso a la izquierda", 1.00, shift_left),
        ("Paso 1/2: levantando el pie derecho", 0.70, step_pose(-0.08, "right")),
        ("Paso 1/2: apoyando el pie derecho", 0.70, shift_left),
        ("Paso 1/2: volviendo al centro", 1.00, NEUTRAL),
        ("Paso 2/2: trasladando el peso a la derecha", 1.00, shift_right),
        ("Paso 2/2: levantando el pie izquierdo", 0.70, step_pose(0.08, "left")),
        ("Paso 2/2: apoyando el pie izquierdo", 0.70, shift_right),
        ("Paso 2/2: volviendo al centro", 1.00, NEUTRAL),
    ]


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

    def advance(self, target):
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.qvel_adr]
        torque = HIGH_KP * (target - q) - HIGH_KD * dq
        self.data.ctrl[:] = np.clip(torque, self.ctrl_low, self.ctrl_high)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        tilt = self.tilt_deg()
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        if tilt > FALL_TILT_DEG:
            raise RuntimeError(
                f"El G1 supero {FALL_TILT_DEG:.0f} grados de inclinacion "
                f"({tilt:.1f} grados)."
            )

        # Mantener la simulacion aproximadamente en tiempo real. Sin esta
        # espera, MuJoCo reproduce la rutina tan rapido como permite la CPU.
        self.next_tick += SIM_DT
        delay = self.next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_tick = time.perf_counter()

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
            print("[Secuencia] Dando dos pasos cortos en el lugar")
            for name, duration, target in walking_stages():
                print(f"[Fase] {name}")
                if not robot.transition(viewer, current, target, duration):
                    return
                current = target

        for cycle in range(1, args.repetir + 1):
            print(f"[Ciclo {cycle}/{args.repetir}]")
            for name, duration, target in stages(args.movimiento):
                print(f"[Fase] {name}")
                if not robot.transition(viewer, current, target, duration):
                    return
                current = target

        print("[Fase] Manteniendo al G1 parado sin banda")
        if args.mantener_segundos is None:
            print("Cerrá la ventana roja o presioná Control+C para terminar.")
        robot.hold(viewer, NEUTRAL, args.mantener_segundos)

    print(f"[OK] Inclinacion maxima: {robot.max_tilt_deg:.1f} grados")
    print("[OK] Simulador cerrado correctamente")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[OK] Demostracion interrumpida por el usuario")
