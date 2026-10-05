"""Levanta y mantiene de pie al G1 dentro de unitree_mujoco.

Este controlador es exclusivamente para simulacion: usa DDS domain 1 y la
interfaz loopback local. No acepta una interfaz de red de un robot fisico.
"""

import sys
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
CONTROL_DT = 0.005  # 200 Hz, igual que la configuracion local del simulador.
TRANSITION_SECONDS = 5.0
NUM_MOTORS = 29

# Orden de motores definido por unitree_robots/g1/g1_29dof.xml.
STAND_Q = np.zeros(NUM_MOTORS, dtype=float)
STAND_Q[0] = -0.15   # left hip pitch
STAND_Q[3] = 0.30    # left knee
STAND_Q[4] = -0.15   # left ankle pitch
STAND_Q[6] = -0.15   # right hip pitch
STAND_Q[9] = 0.30    # right knee
STAND_Q[10] = -0.15  # right ankle pitch

KP = np.array(
    [50, 40, 40, 80, 30, 30]  # left leg
    + [50, 40, 40, 80, 30, 30]  # right leg
    + [40, 30, 30]  # waist
    + [20] * 14,  # arms and wrists
    dtype=float,
)

KD = np.array(
    [2, 2, 2, 3, 2, 2]
    + [2, 2, 2, 3, 2, 2]
    + [2, 2, 2]
    + [1.5] * 14,
    dtype=float,
)


def smoothstep(value: float) -> float:
    """Interpolacion suave con velocidad nula al inicio y al final."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def configure_command(command, mode_machine: int) -> None:
    command.mode_pr = 0
    command.mode_machine = mode_machine
    for index in range(NUM_MOTORS):
        motor = command.motor_cmd[index]
        motor.mode = 1
        motor.q = 0.0
        motor.dq = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0


def send_zero_torque(publisher, command, crc) -> None:
    """Deja de aplicar torque al interrumpir el controlador."""
    for motor in command.motor_cmd[:NUM_MOTORS]:
        motor.kp = 0.0
        motor.kd = 0.0
        motor.tau = 0.0
    for _ in range(10):
        command.crc = crc.Crc(command)
        publisher.Write(command)
        time.sleep(CONTROL_DT)


def main() -> None:
    print("Controlador G1 exclusivamente para unitree_mujoco.")
    print(f"DDS domain={DOMAIN_ID}, interface={INTERFACE}")
    print("Mantene activa la banda elastica durante toda la prueba.")
    print("En el visor: 8 baja el robot y 7 lo levanta.")
    print("No presiones 9 para soltarlo: esta version aun no equilibra la base.")
    input("Presiona Enter en esta Terminal para comenzar...")

    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)

    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init()

    print("Esperando el estado del G1...")
    initial_state = subscriber.Read(timeout=5.0)
    if initial_state is None:
        raise RuntimeError(
            "No llego rt/lowstate. Verifica que el simulador G1 este abierto "
            "con DOMAIN_ID=1 e INTERFACE=lo0."
        )

    initial_q = np.array(
        [initial_state.motor_state[index].q for index in range(NUM_MOTORS)],
        dtype=float,
    )
    mode_machine = int(getattr(initial_state, "mode_machine", 0))

    command = unitree_hg_msg_dds__LowCmd_()
    configure_command(command, mode_machine)
    crc = CRC()
    start = time.perf_counter()
    next_tick = start
    last_reported_second = -1
    standing_reported = False

    print("Transicionando hacia la postura de pie...")
    try:
        while True:
            elapsed = time.perf_counter() - start
            phase = smoothstep(elapsed / TRANSITION_SECONDS)
            desired_q = (1.0 - phase) * initial_q + phase * STAND_Q
            gain_scale = 0.20 + 0.80 * phase

            for index in range(NUM_MOTORS):
                motor = command.motor_cmd[index]
                motor.q = float(desired_q[index])
                motor.dq = 0.0
                motor.kp = float(KP[index] * gain_scale)
                motor.kd = float(KD[index])
                motor.tau = 0.0

            command.crc = crc.Crc(command)
            publisher.Write(command)

            current_second = int(elapsed)
            if phase < 1.0 and current_second != last_reported_second:
                last_reported_second = current_second
                print(f"  postura: {phase * 100:5.1f}%")
            elif phase >= 1.0 and not standing_reported:
                standing_reported = True
                print("Postura alcanzada; manteniendo el G1 de pie.")

            next_tick += CONTROL_DT
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nDeteniendo el controlador y anulando el torque...")
        send_zero_torque(publisher, command, crc)


if __name__ == "__main__":
    main()
