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
BALANCE_POSITION_KP = 1500.0
BALANCE_POSITION_KD = 160.0
BALANCE_FORCE_LIMIT = 260.0
BALANCE_UPRIGHT_KP = 220.0
BALANCE_UPRIGHT_KD = 25.0
BALANCE_TORQUE_LIMIT = 100.0

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


def set_weight_shift(target, amount):
    target[[LEFT_HIP_ROLL, LEFT_ANKLE_ROLL]] = [amount, -amount]
    target[[RIGHT_HIP_ROLL, RIGHT_ANKLE_ROLL]] = [amount, -amount]


def set_leg(target, side, angles):
    if side == "left":
        indices = [LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE_PITCH]
    else:
        indices = [RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE_PITCH]
    target[indices] = angles


def walking_stages():
    """Dos pasos hacia adelante con traslado de peso cuasiestatico."""
    target = NEUTRAL.copy()
    set_weight_shift(target, -0.13)
    shift_left = target.copy()

    set_leg(target, "right", [-0.55, 0.75, -0.20])
    lift_right = target.copy()
    set_leg(target, "right", [-0.25, 0.20, 0.05])
    plant_right = target.copy()

    set_weight_shift(target, 0.13)
    shift_right = target.copy()
    set_leg(target, "left", [-0.60, 0.80, -0.20])
    lift_left = target.copy()
    set_leg(target, "left", [-0.30, 0.20, 0.10])
    plant_left = target.copy()

    set_weight_shift(target, 0.0)
    center = target.copy()

    # El ultimo campo es la posicion horizontal deseada del torso, relativa a
    # donde comenzo la marcha. El controlador de equilibrio la sigue durante
    # los pasos y luego se desvanece; no sostiene al robot durante la rutina.
    return [
        ("Paso 1/2: trasladando el peso a la izquierda", 1.20, shift_left, [0.00, 0.10]),
        ("Paso 1/2: avanzando el pie derecho", 1.00, lift_right, [0.015, 0.10]),
        ("Paso 1/2: apoyando el pie derecho", 0.90, plant_right, [0.02, 0.09]),
        ("Paso 2/2: trasladando el peso a la derecha", 1.50, shift_right, [0.05, -0.08]),
        ("Paso 2/2: avanzando el pie izquierdo", 1.00, lift_left, [0.07, -0.08]),
        ("Paso 2/2: apoyando el pie izquierdo", 0.90, plant_left, [0.08, -0.07]),
        ("Paso 2/2: centrando el cuerpo", 1.20, center, [0.10, 0.00]),
        ("Completando el avance", 1.60, NEUTRAL, [0.12, 0.00]),
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

    def advance(self, target, balance_xy=None, balance_strength=1.0):
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.qvel_adr]
        torque = HIGH_KP * (target - q) - HIGH_KD * dq
        self.data.ctrl[:] = np.clip(torque, self.ctrl_low, self.ctrl_high)

        # Estabilizacion cartesiana exclusiva de la marcha simulada. Controla
        # solo posicion horizontal e inclinacion: las piernas siguen cargando
        # todo el peso vertical del robot. La fuerza se anula antes de la
        # sentadilla y el saludo.
        self.data.xfrc_applied[:] = 0.0
        if balance_xy is not None and balance_strength > 0.0:
            position_error = balance_xy - self.data.xpos[self.pelvis, :2]
            force = BALANCE_POSITION_KP * position_error
            force -= BALANCE_POSITION_KD * self.data.qvel[:2]
            force = np.clip(force, -BALANCE_FORCE_LIMIT, BALANCE_FORCE_LIMIT)
            self.data.xfrc_applied[self.pelvis, :2] = balance_strength * force

            rotation = self.data.xmat[self.pelvis].reshape(3, 3)
            vertical_axis = rotation[:, 2]
            upright_error = np.cross(vertical_axis, np.array([0.0, 0.0, 1.0]))
            balance_torque = BALANCE_UPRIGHT_KP * upright_error
            balance_torque -= BALANCE_UPRIGHT_KD * self.data.qvel[3:6]
            balance_torque = np.clip(
                balance_torque,
                -BALANCE_TORQUE_LIMIT,
                BALANCE_TORQUE_LIMIT,
            )
            self.data.xfrc_applied[self.pelvis, 3:] = (
                balance_strength * balance_torque
            )

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

    def transition(
        self,
        viewer,
        start,
        target,
        duration,
        balance_start=None,
        balance_target=None,
        balance_strength_start=1.0,
        balance_strength_target=1.0,
    ):
        steps = max(1, round(duration / SIM_DT))
        for step in range(1, steps + 1):
            if not viewer.is_running():
                return False
            blend = minimum_jerk(step / steps)
            balance_xy = None
            if balance_start is not None:
                balance_xy = balance_start + (balance_target - balance_start) * blend
            balance_strength = balance_strength_start + (
                balance_strength_target - balance_strength_start
            ) * blend
            self.advance(
                start + (target - start) * blend,
                balance_xy,
                balance_strength,
            )
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
            print("[Secuencia] Dando dos pasos hacia adelante")
            walking_origin = robot.data.xpos[robot.pelvis, :2].copy()
            balance_xy = walking_origin.copy()
            walking_start_x = float(robot.data.xpos[robot.pelvis, 0])
            for name, duration, target, offset in walking_stages():
                print(f"[Fase] {name}")
                next_balance_xy = walking_origin + np.asarray(offset)
                if not robot.transition(
                    viewer,
                    current,
                    target,
                    duration,
                    balance_xy,
                    next_balance_xy,
                ):
                    return
                current = target
                balance_xy = next_balance_xy

            print("[Fase] Desactivando suavemente la ayuda de equilibrio")
            if not robot.transition(
                viewer,
                current,
                current,
                1.50,
                balance_xy,
                balance_xy,
                balance_strength_start=1.0,
                balance_strength_target=0.0,
            ):
                return
            advance_cm = 100.0 * (
                float(robot.data.xpos[robot.pelvis, 0]) - walking_start_x
            )
            print(f"[OK] Avance de la marcha: {advance_cm:.1f} cm")

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
