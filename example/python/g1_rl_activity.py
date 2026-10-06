"""Controlador DDS separado: dos pasos RL, sentadilla y saludo del G1.

Debe ejecutarse con el simulador Python G1 ya abierto. La comunicacion queda
restringida al dominio DDS 1 y a la interfaz loopback local.
"""

import argparse
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from g1_squat_wave import (
    HIGH_KD,
    HIGH_KP,
    NEUTRAL,
    SQUAT,
    minimum_jerk,
    stages,
)


DOMAIN_ID = 1
INTERFACE = "lo0" if sys.platform == "darwin" else "lo"
CONTROL_DT = 0.002
NUM_MOTORS = 29
POLICY_DECIMATION = 10
FORWARD_SECONDS = 0.8
SETTLE_SECONDS = 1.6
FALL_TILT_DEG = 45.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = (
    PROJECT_ROOT
    / "third_party/unitree_rl_gym/deploy/pre_train/g1/motion.pt"
)

POLICY_KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=float)
POLICY_KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=float)
POLICY_DEFAULT = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
POLICY_COMMAND_SCALE = np.array([2.0, 2.0, 0.25])
RECOVERY_KP = HIGH_KP.copy()
RECOVERY_KD = HIGH_KD.copy()
RECOVERY_KP[:12] = np.array([220, 220, 180, 300, 100, 100] * 2)
RECOVERY_KD[:12] = np.array([8, 8, 7, 10, 5, 5] * 2)


def gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ]
    )


class DdsController:
    def __init__(self):
        ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lock = threading.Lock()
        self.state_event = threading.Event()
        self.state = None
        self.subscriber.Init(self._on_state, 1)

        self.command = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.mode_machine = 0
        self.next_tick = time.perf_counter()
        self.max_tilt_deg = 0.0
        for motor in self.command.motor_cmd[:NUM_MOTORS]:
            motor.mode = 1
            motor.dq = 0.0
            motor.tau = 0.0

    def _on_state(self, state):
        with self.lock:
            self.state = state
        self.state_event.set()

    def snapshot(self):
        with self.lock:
            state = self.state
            if state is None:
                return None
            q = np.array(
                [state.motor_state[i].q for i in range(NUM_MOTORS)],
                dtype=float,
            )
            dq = np.array(
                [state.motor_state[i].dq for i in range(NUM_MOTORS)],
                dtype=float,
            )
            quaternion = np.array(state.imu_state.quaternion, dtype=float)
            gyroscope = np.array(state.imu_state.gyroscope, dtype=float)
            tick = int(state.tick)
            mode_machine = int(getattr(state, "mode_machine", 0))
        return q, dq, quaternion, gyroscope, tick, mode_machine

    def wait_for_state(self):
        print("Esperando el estado del simulador G1...")
        if not self.state_event.wait(timeout=5.0):
            raise RuntimeError(
                "No llego rt/lowstate. Abri primero el simulador con "
                "./iniciar_simulador_g1_mac.command"
            )
        snapshot = self.snapshot()
        self.mode_machine = snapshot[5]
        print("[OK] Simulador conectado por DDS")
        return snapshot[0]

    def check_stability(self, quaternion):
        qw, qx, qy, qz = quaternion
        vertical = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt = float(np.degrees(np.arccos(np.clip(vertical, -1.0, 1.0))))
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        if tilt > FALL_TILT_DEG:
            raise RuntimeError(
                f"El G1 supero {FALL_TILT_DEG:.0f} grados de inclinacion "
                f"({tilt:.1f} grados)."
            )

    def send(self, target, kp, kd):
        self.command.mode_pr = 0
        self.command.mode_machine = self.mode_machine
        for index in range(NUM_MOTORS):
            motor = self.command.motor_cmd[index]
            motor.q = float(target[index])
            motor.kp = float(kp[index])
            motor.kd = float(kd[index])
        self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)

        snapshot = self.snapshot()
        if snapshot is not None:
            self.check_stability(snapshot[2])

        self.next_tick += CONTROL_DT
        delay = self.next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_tick = time.perf_counter()

    def transition(self, start, target, duration, kp=HIGH_KP, kd=HIGH_KD):
        steps_count = max(1, round(duration / CONTROL_DT))
        for step in range(1, steps_count + 1):
            blend = minimum_jerk(step / steps_count)
            self.send(start + (target - start) * blend, kp, kd)

    def walk_two_steps(self):
        if not POLICY_PATH.exists():
            raise RuntimeError(
                "Falta la politica oficial. Ejecuta: "
                "git submodule update --init --recursive"
            )

        policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
        policy.eval()
        action = np.zeros(12, dtype=np.float32)
        lower_target = POLICY_DEFAULT.copy()
        target = NEUTRAL.copy()
        observation = np.zeros(47, dtype=np.float32)
        policy_kp = HIGH_KP.copy()
        policy_kd = HIGH_KD.copy()
        policy_kp[:12] = POLICY_KP
        policy_kd[:12] = POLICY_KD

        forward_steps = round(FORWARD_SECONDS / CONTROL_DT)
        settle_steps = round(SETTLE_SECONDS / CONTROL_DT)
        total_steps = forward_steps + settle_steps

        print("[Fase] Politica RL: avanzando pie derecho e izquierdo")
        for policy_step in range(total_steps):
            command = np.zeros(3)
            if policy_step < forward_steps:
                command[0] = 0.25
            elif policy_step == forward_steps:
                print("[Fase] Politica RL: frenando y recuperando equilibrio")

            target[:12] = lower_target
            self.send(target, policy_kp, policy_kd)

            if policy_step % POLICY_DECIMATION != 0:
                continue

            snapshot = self.snapshot()
            if snapshot is None:
                continue
            q, dq, quaternion, gyroscope, _, _ = snapshot
            phase = (policy_step * CONTROL_DT % 0.8) / 0.8

            observation[:3] = gyroscope * 0.25
            observation[3:6] = gravity_orientation(quaternion)
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
            lower_target = action * 0.25 + POLICY_DEFAULT

        print(
            f"[OK] Marcha terminada; inclinacion maxima "
            f"{self.max_tilt_deg:.1f} grados"
        )
        print("[Fase] Centrando el peso en una sentadilla de recuperacion")
        current = self.snapshot()[0]
        self.transition(current, SQUAT, 2.00, RECOVERY_KP, RECOVERY_KD)
        for _ in range(round(0.50 / CONTROL_DT)):
            self.send(SQUAT, HIGH_KP, HIGH_KD)

        print("[Fase] Levantando desde ambos pies")
        self.transition(SQUAT, NEUTRAL, 2.50, HIGH_KP, HIGH_KD)
        for _ in range(round(0.50 / CONTROL_DT)):
            self.send(NEUTRAL, HIGH_KP, HIGH_KD)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ordena al G1 abierto caminar y ejecutar la rutina."
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
        help="ejecuta la rutina sin los dos pasos iniciales",
    )
    parser.add_argument(
        "--mantener-segundos",
        type=float,
        default=2.0,
        help="segundos de postura final antes de devolver el control de la Terminal",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    controller = DdsController()
    current = controller.wait_for_state()

    print("[Fase] Adoptando postura inicial")
    controller.transition(current, NEUTRAL, 3.0)
    current = NEUTRAL

    if not args.sin_pasos:
        controller.walk_two_steps()
        current = NEUTRAL

    for cycle in range(1, args.repetir + 1):
        print(f"[Ciclo {cycle}/{args.repetir}]")
        for name, duration, target in stages(args.movimiento):
            print(f"[Fase] {name}")
            controller.transition(current, target, duration)
            current = target

    print("[Fase] Manteniendo al G1 parado")
    try:
        steps_count = round(args.mantener_segundos / CONTROL_DT)
        for _ in range(steps_count):
            controller.send(NEUTRAL, HIGH_KP, HIGH_KD)
    except KeyboardInterrupt:
        print("\n[OK] Controlador detenido; el simulador sigue abierto.")

    print(f"[OK] Inclinacion maxima: {controller.max_tilt_deg:.1f} grados")
    print("[OK] Actividad terminada. Podes ejecutar el mismo comando otra vez.")


if __name__ == "__main__":
    main()
