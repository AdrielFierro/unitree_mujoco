"""Sentadilla y saludo del G1 en unitree_mujoco, adaptado para macOS.

Este controlador es exclusivamente para el simulador local. Usa el dominio
DDS 1 y la interfaz loopback (lo0 en macOS), por lo que no puede enviar
comandos a un robot fisico.
"""

import argparse
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


DOMAIN_ID = 1
INTERFACE = "lo0" if sys.platform == "darwin" else "lo"
CONTROL_DT = 0.005
UNASSISTED_SIM_DT = 0.002
NUM_MOTORS = 29
INITIAL_TRANSITION_S = 3.0

JOINT_INDEX = {
    "left_hip_pitch": 0,
    "left_knee": 3,
    "left_ankle_pitch": 4,
    "right_hip_pitch": 6,
    "right_knee": 9,
    "right_ankle_pitch": 10,
    "left_shoulder_pitch": 15,
    "left_elbow": 18,
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}

KP = np.array(
    [60, 60, 60, 100, 40, 40]
    + [60, 60, 60, 100, 40, 40]
    + [60, 40, 40]
    + [40] * 14,
    dtype=float,
)
KD = np.array(
    [1, 1, 1, 2, 1, 1]
    + [1, 1, 1, 2, 1, 1]
    + [1, 1, 1]
    + [1] * 14,
    dtype=float,
)

# Rigideces educativas para sostener el G1 sin banda. Son mucho mayores que
# las del robot real: reemplazan un controlador de equilibrio que este ejemplo
# todavia no implementa y nunca deben enviarse a hardware fisico.
HIGH_KP = np.array(
    [800] * 15 + [100] * 4 + [50] * 3 + [100] * 4 + [50] * 3,
    dtype=float,
)
HIGH_KD = np.array(
    [15] * 15 + [2.5] * 4 + [1.25] * 3 + [2.5] * 4 + [1.25] * 3,
    dtype=float,
)


def pose(**joints):
    result = np.zeros(NUM_MOTORS, dtype=float)
    result[JOINT_INDEX["left_shoulder_pitch"]] = 0.20
    result[JOINT_INDEX["left_elbow"]] = -0.30
    result[JOINT_INDEX["right_shoulder_pitch"]] = 0.20
    result[JOINT_INDEX["right_elbow"]] = -0.30
    for name, value in joints.items():
        result[JOINT_INDEX[name]] = value
    return result


NEUTRAL = pose()
SQUAT = pose(
    left_hip_pitch=-0.55,
    left_knee=1.05,
    left_ankle_pitch=-0.50,
    right_hip_pitch=-0.55,
    right_knee=1.05,
    right_ankle_pitch=-0.50,
)
ARM_UP = pose(
    right_shoulder_pitch=0.0,
    right_shoulder_roll=-1.71,
    right_shoulder_yaw=-1.57,
    right_elbow=0.14,
    right_wrist_pitch=0.0,
    right_wrist_yaw=0.0,
)
WAVE_LEFT = ARM_UP.copy()
WAVE_LEFT[JOINT_INDEX["right_elbow"]] = -0.11
WAVE_LEFT[JOINT_INDEX["right_wrist_pitch"]] = -0.35
WAVE_RIGHT = ARM_UP.copy()
WAVE_RIGHT[JOINT_INDEX["right_elbow"]] = 0.39
WAVE_RIGHT[JOINT_INDEX["right_wrist_pitch"]] = 0.35


def stages(movement):
    squat = [
        ("Sentadilla: bajando", 1.60, SQUAT),
        ("Sentadilla: sosteniendo", 0.70, SQUAT),
        ("Sentadilla: subiendo", 1.60, NEUTRAL),
    ]
    wave = [
        ("Saludo: levantando el brazo", 0.90, ARM_UP),
        ("Saludo: izquierda", 0.35, WAVE_LEFT),
        ("Saludo: derecha", 0.35, WAVE_RIGHT),
        ("Saludo: izquierda", 0.35, WAVE_LEFT),
        ("Saludo: derecha", 0.35, WAVE_RIGHT),
        ("Saludo: bajando el brazo", 0.90, NEUTRAL),
    ]
    if movement == "sentadilla":
        return squat
    if movement == "saludo":
        return wave
    return squat + wave


def minimum_jerk(value):
    value = float(np.clip(value, 0.0, 1.0))
    return 10 * value**3 - 15 * value**4 + 6 * value**5


class Controller:
    def __init__(self, without_band=False):
        ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_lock = threading.Lock()
        self._state_event = threading.Event()
        self._latest_state = None
        self.max_tilt_deg = 0.0
        self.use_sim_time = without_band
        self.subscriber.Init(self._on_state, 1)
        self.command = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.next_tick = time.perf_counter()
        self.mode_machine = 0
        kp = HIGH_KP if without_band else KP
        kd = HIGH_KD if without_band else KD
        for index in range(NUM_MOTORS):
            motor = self.command.motor_cmd[index]
            motor.mode = 1
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(kp[index])
            motor.kd = float(kd[index])

    def _on_state(self, state):
        with self._state_lock:
            self._latest_state = state
        self._state_event.set()

    def latest_state(self):
        with self._state_lock:
            return self._latest_state

    def check_stability(self):
        state = self.latest_state()
        if state is None:
            return
        w, x, y, z = state.imu_state.quaternion
        vertical = 1.0 - 2.0 * (x * x + y * y)
        tilt = float(np.degrees(np.arccos(np.clip(vertical, -1.0, 1.0))))
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        if tilt > 45.0:
            raise RuntimeError(
                f"El G1 superó 45° de inclinación ({tilt:.1f}°); "
                "se interrumpe el control."
            )

    def wait_for_state(self):
        print("Esperando el estado del G1...")
        if not self._state_event.wait(timeout=5.0):
            raise RuntimeError(
                "No llego rt/lowstate. Abri primero el simulador G1 y "
                "comproba que use DOMAIN_ID=1 e INTERFACE=lo0."
            )
        state = self.latest_state()
        self.mode_machine = int(getattr(state, "mode_machine", 0))
        return np.array(
            [state.motor_state[i].q for i in range(NUM_MOTORS)], dtype=float
        )

    def send(self, target):
        self.command.mode_pr = 0
        self.command.mode_machine = self.mode_machine
        for index in range(NUM_MOTORS):
            self.command.motor_cmd[index].q = float(target[index])
        self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)
        self.check_stability()
        self.next_tick += CONTROL_DT
        delay = self.next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_tick = time.perf_counter()

    def transition(self, start, target, duration):
        if self.use_sim_time:
            self.transition_sim_time(start, target, duration)
            return
        steps = max(1, round(duration / CONTROL_DT))
        for step in range(1, steps + 1):
            blend = minimum_jerk(step / steps)
            self.send(start + (target - start) * blend)

    def transition_sim_time(self, start, target, duration):
        """Interpola usando el contador de pasos del simulador, no el reloj."""
        state = self.latest_state()
        start_tick = int(state.tick)
        last_tick = start_tick
        last_progress = time.perf_counter()

        while True:
            state = self.latest_state()
            tick = int(state.tick)
            if tick != last_tick:
                last_tick = tick
                last_progress = time.perf_counter()
            elif time.perf_counter() - last_progress > 2.0:
                raise RuntimeError(
                    "El reloj fisico de MuJoCo no avanza; el simulador "
                    "puede estar pausado o bloqueado."
                )

            simulated = max(0.0, (tick - start_tick) * UNASSISTED_SIM_DT)
            blend = minimum_jerk(simulated / duration)
            self.send(start + (target - start) * blend)
            if simulated >= duration:
                break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hace que el G1 simulado realice una sentadilla y/o saludo."
    )
    parser.add_argument(
        "--movimiento",
        choices=("rutina", "sentadilla", "saludo"),
        default="rutina",
    )
    parser.add_argument("--repetir", type=int, default=1)
    parser.add_argument(
        "--sin-banda",
        action="store_true",
        help="usa rigidez alta y mantiene el control al terminar",
    )
    parser.add_argument(
        "--mantener-segundos",
        type=float,
        help="tiempo de postura final; sin este valor, --sin-banda mantiene indefinidamente",
    )
    parser.add_argument(
        "--cuenta-regresiva",
        type=int,
        default=2,
        help="segundos antes de comenzar; no hace falta presionar Enter",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repetir < 1:
        raise ValueError("--repetir debe ser al menos 1")
    if args.cuenta_regresiva < 0:
        raise ValueError("--cuenta-regresiva no puede ser negativa")
    if args.mantener_segundos is not None and args.mantener_segundos < 0:
        raise ValueError("--mantener-segundos no puede ser negativo")

    print("Controlador exclusivo para el G1 simulado (DDS 1, interfaz lo0).")
    if args.sin_banda:
        print("Modo sin banda: rigidez alta exclusiva para simulación.")
        print("El simulador debe haberse iniciado con su modo sin banda.")
    else:
        print("Mantené activa la banda elástica; 7 levanta, 8 baja y 9 la suelta.")
    controller = Controller(without_band=args.sin_banda)
    current = controller.wait_for_state()

    for remaining in range(args.cuenta_regresiva, 0, -1):
        print(f"Comenzando en {remaining}...")
        time.sleep(1.0)

    print("[Fase] Llevando el G1 suavemente a la postura inicial")
    controller.transition(current, NEUTRAL, INITIAL_TRANSITION_S)
    current = NEUTRAL

    try:
        for cycle in range(1, args.repetir + 1):
            print(f"[Ciclo {cycle}/{args.repetir}]")
            for name, duration, target in stages(args.movimiento):
                print(f"[Fase] {name}")
                controller.transition(current, target, duration)
                current = target
        print("[Fase] Manteniendo la postura final")
        hold_seconds = args.mantener_segundos
        if hold_seconds is None and not args.sin_banda:
            hold_seconds = 2.0
        if hold_seconds is None:
            print("[OK] G1 parado sin banda. Ctrl+C detiene el controlador.")
            while True:
                controller.send(NEUTRAL)
        else:
            for _ in range(round(hold_seconds / CONTROL_DT)):
                controller.send(NEUTRAL)
    except KeyboardInterrupt:
        print("\n[AVISO] Rutina interrumpida; volviendo a postura neutra.")
        controller.transition(current, NEUTRAL, 1.0)

    print(f"[OK] Inclinación máxima observada: {controller.max_tilt_deg:.1f}°")
    print("[OK] Rutina terminada.")


if __name__ == "__main__":
    main()
