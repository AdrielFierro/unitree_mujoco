"""Servicio persistente de equilibrio y comandos para el G1 simulado."""

from pathlib import Path
import os
import socket
import time

import numpy as np
import torch

from g1_rl_activity import (
    CONTROL_DT,
    LOW_POWER_MODE,
    NUM_MOTORS,
    POLICY_COMMAND_SCALE,
    POLICY_DECIMATION,
    POLICY_DEFAULT,
    POLICY_KD,
    POLICY_KP,
    POLICY_PATH,
    DdsController,
    gravity_orientation,
)
from g1_squat_wave import HIGH_KD, HIGH_KP, NEUTRAL, stages


SOCKET_PATH = Path("/tmp/unitree_g1_stabilizer.sock")
MOTION_STEPS = round(0.8 / CONTROL_DT)
SETTLE_STEPS = round(1.6 / CONTROL_DT)
IDLE_PHASE = 0.35
STARTUP_TILT_LIMIT_DEG = 15.0
COMMAND_TILT_LIMIT_DEG = 20.0
UNSTABLE_TILT_DEG = 25.0
UNSTABLE_TIME_S = 0.5
EMERGENCY_TILT_DEG = 35.0
EMERGENCY_JOINT_SPEED = 50.0
POLICY_OBSERVATION_LIMIT = 100.0
POLICY_ACTION_LIMIT = 3.0
MOTION_COMMANDS = {
    "adelante": (np.array([0.40, 0.0, 0.0]), "avanzando"),
    "atras": (np.array([-0.30, 0.0, 0.0]), "retrocediendo"),
    "izquierda": (np.array([0.0, 0.28, 0.0]), "desplazandose a la izquierda"),
    "derecha": (np.array([0.0, -0.28, 0.0]), "desplazandose a la derecha"),
    "girar_izquierda": (np.array([0.0, 0.0, 0.50]), "girando a la izquierda"),
    "girar_derecha": (np.array([0.0, 0.0, -0.50]), "girando a la derecha"),
}
# Sentadilla moderada compatible con la postura base de la politica. La
# demostracion autonoma conserva por separado su sentadilla profunda.
MENU_SQUAT_OFFSET = np.array([-0.08, 0.0, 0.0, 0.15, -0.07, 0.0] * 2)


class StabilizerService:
    def __init__(self):
        self.controller = DdsController()
        self.kp = HIGH_KP.copy()
        self.kd = HIGH_KD.copy()
        self.kp[:12] = POLICY_KP
        self.kd[:12] = POLICY_KD

        current = self.controller.wait_for_state()
        snapshot = self.controller.snapshot()
        startup_error, startup_tilt = self.state_error(snapshot)
        if startup_error or startup_tilt > STARTUP_TILT_LIMIT_DEG:
            self.controller.publish_safe_damping()
            detail = startup_error or (
                f"inclinacion inicial {startup_tilt:.1f} grados; "
                f"el limite es {STARTUP_TILT_LIMIT_DEG:.0f}"
            )
            raise RuntimeError(
                "El G1 no esta en una postura segura para arrancar: " + detail
            )
        print("[Fase] Adoptando postura inicial")
        self.controller.transition(current, NEUTRAL, 3.0, self.kp, self.kd)

        self.policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
        self.policy.eval()
        self.action = np.zeros(12, dtype=np.float32)
        self.lower_target = POLICY_DEFAULT.copy()
        self.observation = np.zeros(47, dtype=np.float32)
        self.policy_step = 0
        self.gait_step = 0

        self.motion_remaining = 0
        self.motion_command = np.zeros(3)
        self.motion_label = ""
        self.settle_remaining = 0
        self.pending_wave = False
        self.wave_stages = []
        self.wave_index = 0
        self.wave_step = 0
        self.upper_start = NEUTRAL[12:].copy()
        self.upper_target = NEUTRAL[12:].copy()
        self.upper_current = NEUTRAL[12:].copy()
        self.squat_stages = []
        self.squat_index = 0
        self.squat_step = 0
        self.lower_override_start = np.zeros(12)
        self.lower_override_target = np.zeros(12)
        self.lower_override_current = np.zeros(12)
        self.running = True
        self.emergency_reason = None
        self.unstable_steps = 0

        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if SOCKET_PATH.exists():
            os.unlink(SOCKET_PATH)
        self.server.bind(str(SOCKET_PATH))
        self.server.listen(4)
        self.server.setblocking(False)

    def busy(self):
        return bool(
            self.motion_remaining
            or self.settle_remaining
            or self.pending_wave
            or self.wave_stages
            or self.squat_stages
        )

    def state_error(self, snapshot):
        if snapshot is None:
            return "no hay estado DDS", 180.0
        q, dq, quaternion, gyroscope, _, _ = snapshot
        if not all(
            np.all(np.isfinite(values))
            for values in (q, dq, quaternion, gyroscope)
        ):
            return "el estado contiene NaN o Inf", 180.0
        max_joint_speed = float(np.max(np.abs(dq)))
        if max_joint_speed > EMERGENCY_JOINT_SPEED:
            return (
                f"velocidad articular excesiva: {max_joint_speed:.1f} rad/s",
                180.0,
            )
        tilt = self.controller.tilt_deg(quaternion)
        self.controller.max_tilt_deg = max(
            self.controller.max_tilt_deg,
            tilt,
        )
        return None, tilt

    def enter_emergency(self, reason):
        if self.emergency_reason is None:
            print(f"[EMERGENCIA] {reason}")
            print("[EMERGENCIA] Activando amortiguacion segura")
        self.emergency_reason = reason
        self.motion_remaining = 0
        self.settle_remaining = 0
        self.pending_wave = False
        self.wave_stages = []
        self.squat_stages = []
        self.controller.publish_safe_damping()

    def status(self):
        state_error, current_tilt = self.state_error(self.controller.snapshot())
        if self.emergency_reason:
            return (
                f"EMERGENCIA: {self.emergency_reason}; "
                "reinicia el simulador y el estabilizador"
            )
        if state_error:
            return "estado no seguro: " + state_error
        if self.motion_remaining:
            phase = self.motion_label
        elif self.settle_remaining:
            phase = "estabilizando"
        elif self.squat_stages:
            phase = "haciendo una sentadilla"
        elif self.wave_stages:
            phase = "saludando"
        else:
            phase = "parado y equilibrando"
        return (
            f"{phase}; inclinacion actual={current_tilt:.1f} grados; "
            "inclinacion maxima="
            f"{self.controller.max_tilt_deg:.1f} grados"
        )

    def command_is_safe(self):
        state_error, tilt = self.state_error(self.controller.snapshot())
        if state_error:
            self.enter_emergency(state_error)
            return False, state_error
        if tilt >= EMERGENCY_TILT_DEG:
            reason = f"caida detectada: {tilt:.1f} grados"
            self.enter_emergency(reason)
            return False, reason
        if tilt > COMMAND_TILT_LIMIT_DEG:
            return False, (
                f"el G1 esta inclinado {tilt:.1f} grados; "
                f"espera a que baje de {COMMAND_TILT_LIMIT_DEG:.0f}"
            )
        return True, None

    def monitor_safety(self):
        state_error, current_tilt = self.state_error(
            self.controller.snapshot()
        )
        if state_error:
            self.enter_emergency(state_error)
        elif current_tilt >= EMERGENCY_TILT_DEG:
            self.enter_emergency(
                f"caida detectada: {current_tilt:.1f} grados"
            )
        elif current_tilt > UNSTABLE_TILT_DEG:
            self.unstable_steps += 1
            if self.unstable_steps >= round(
                UNSTABLE_TIME_S / CONTROL_DT
            ):
                self.enter_emergency(
                    "no recupero el equilibrio durante "
                    f"{UNSTABLE_TIME_S:.1f} segundos "
                    f"({current_tilt:.1f} grados)"
                )
        else:
            self.unstable_steps = 0
        return current_tilt

    def start_motion(self, command_name, then_wave=False):
        # Cada orden debe empezar al principio del ciclo de marcha. Antes se
        # reutilizaba la fase global del estabilizador y algunas ordenes
        # comenzaban a mitad de zancada, moviendo los pies casi en el lugar.
        self.gait_step = 0
        self.action.fill(0.0)
        self.lower_target = POLICY_DEFAULT.copy()
        command, label = MOTION_COMMANDS[command_name]
        self.motion_command = command.copy()
        self.motion_label = label
        self.motion_remaining = MOTION_STEPS
        self.settle_remaining = SETTLE_STEPS
        self.pending_wave = then_wave
        print(f"[Comando] {label.capitalize()}")

    def start_squat(self):
        self.squat_stages = [
            ("bajando", 1.20, MENU_SQUAT_OFFSET),
            ("sosteniendo", 0.50, MENU_SQUAT_OFFSET),
            ("subiendo", 1.20, np.zeros(12)),
        ]
        self.squat_index = 0
        self.squat_step = 0
        self.lower_override_start = np.zeros(12)
        self.lower_override_current = self.lower_override_start.copy()
        self.lower_override_target = self.squat_stages[0][2].copy()
        print("[Comando] Comenzando la sentadilla")

    def start_wave(self):
        self.wave_stages = stages("saludo")
        self.wave_index = 0
        self.wave_step = 0
        self.upper_start = self.upper_current.copy()
        self.upper_target = self.wave_stages[0][2][12:].copy()
        print("[Comando] Comenzando el saludo")

    def handle_request(self, request):
        request = request.strip().lower()
        if request == "estado":
            return "OK " + self.status()
        if request in ("salir", "cerrar"):
            self.running = False
            return "OK cerrando el estabilizador"
        if self.emergency_reason:
            return "ERROR " + self.status()
        if self.busy():
            return "OCUPADO " + self.status()
        safe, reason = self.command_is_safe()
        if not safe:
            return "ERROR comando rechazado: " + reason
        aliases = {
            "pasos": "adelante",
            "caminar": "adelante",
            "encarar_izquierda": "girar_izquierda",
            "encarar_derecha": "girar_derecha",
        }
        request = aliases.get(request, request)
        if request in MOTION_COMMANDS:
            self.start_motion(request)
            return f"OK comando {request} aceptado"
        if request == "sentadilla":
            self.start_squat()
            return "OK comando sentadilla aceptado"
        if request == "saludo":
            self.start_wave()
            return "OK comando saludo aceptado"
        if request in ("actividad", "rutina"):
            self.start_motion("adelante", then_wave=True)
            return "OK actividad aceptada: pasos y saludo"
        return (
            "ERROR usa: adelante, atras, izquierda, derecha, "
            "girar_izquierda, girar_derecha, sentadilla, saludo, estado o salir"
        )

    def poll_commands(self):
        while True:
            try:
                connection, _ = self.server.accept()
            except BlockingIOError:
                return
            with connection:
                request = connection.recv(128).decode("utf-8", errors="replace")
                response = self.handle_request(request)
                connection.sendall((response + "\n").encode("utf-8"))

    def update_upper_body(self):
        if not self.wave_stages:
            return
        _, duration, _ = self.wave_stages[self.wave_index]
        steps_count = max(1, round(duration / CONTROL_DT))
        self.wave_step += 1
        value = min(1.0, self.wave_step / steps_count)
        blend = 10 * value**3 - 15 * value**4 + 6 * value**5
        self.upper_current = self.upper_start + (
            self.upper_target - self.upper_start
        ) * blend
        if self.wave_step < steps_count:
            return

        self.wave_index += 1
        if self.wave_index >= len(self.wave_stages):
            self.wave_stages = []
            self.upper_current = NEUTRAL[12:].copy()
            print("[OK] Saludo terminado; sigo equilibrando al G1")
            return
        self.wave_step = 0
        self.upper_start = self.upper_current.copy()
        self.upper_target = self.wave_stages[self.wave_index][2][12:].copy()

    def update_squat(self):
        if not self.squat_stages:
            return
        name, duration, _ = self.squat_stages[self.squat_index]
        steps_count = max(1, round(duration / CONTROL_DT))
        self.squat_step += 1
        value = min(1.0, self.squat_step / steps_count)
        blend = 10 * value**3 - 15 * value**4 + 6 * value**5
        self.lower_override_current = self.lower_override_start + (
            self.lower_override_target - self.lower_override_start
        ) * blend
        if self.squat_step < steps_count:
            return

        print(f"[Fase] Sentadilla: {name}")
        self.squat_index += 1
        if self.squat_index >= len(self.squat_stages):
            self.squat_stages = []
            self.lower_override_current = np.zeros(12)
            print("[OK] Sentadilla terminada; sigo equilibrando al G1")
            return
        self.squat_step = 0
        self.lower_override_start = self.lower_override_current.copy()
        self.lower_override_target = self.squat_stages[self.squat_index][2].copy()

    def update_policy(self, command, gait_active):
        snapshot = self.controller.snapshot()
        if snapshot is None:
            return
        q, dq, quaternion, gyroscope, _, _ = snapshot
        if not all(
            np.all(np.isfinite(values))
            for values in (q, dq, quaternion, gyroscope, self.action)
        ):
            raise RuntimeError("estado u accion de politica con NaN o Inf")
        self.observation[:3] = gyroscope * 0.25
        self.observation[3:6] = gravity_orientation(quaternion)
        self.observation[6:9] = command * POLICY_COMMAND_SCALE
        self.observation[9:21] = q[:12] - POLICY_DEFAULT
        self.observation[21:33] = dq[:12] * 0.05
        self.observation[33:45] = self.action
        np.clip(
            self.observation,
            -POLICY_OBSERVATION_LIMIT,
            POLICY_OBSERVATION_LIMIT,
            out=self.observation,
        )
        if gait_active:
            phases = [(self.gait_step * CONTROL_DT % 0.8) / 0.8]
        else:
            # La politica aprendio una marcha ciclica incluso con velocidad
            # cero. Promediar dos fases opuestas cancela la preferencia por
            # una pierna que causaba mini pasos o desplazamiento lateral.
            phases = [IDLE_PHASE, (IDLE_PHASE + 0.5) % 1.0]

        actions = []
        with torch.no_grad():
            for phase in phases:
                self.observation[45:47] = [
                    np.sin(2.0 * np.pi * phase),
                    np.cos(2.0 * np.pi * phase),
                ]
                tensor = torch.from_numpy(self.observation).unsqueeze(0)
                actions.append(self.policy(tensor).numpy().squeeze())
        self.action = np.mean(actions, axis=0)
        if not np.all(np.isfinite(self.action)):
            raise RuntimeError("la politica produjo NaN o Inf")
        np.clip(
            self.action,
            -POLICY_ACTION_LIMIT,
            POLICY_ACTION_LIMIT,
            out=self.action,
        )
        self.lower_target = self.action * 0.25 + POLICY_DEFAULT

    def run(self):
        if LOW_POWER_MODE:
            print("[OK] Modo bajo consumo: control 250 Hz, politica 50 Hz")
        print("[OK] Estabilizador activo y esperando comandos.")
        print(f"[OK] Socket local: {SOCKET_PATH}")
        try:
            while self.running:
                self.poll_commands()
                self.monitor_safety()

                if self.emergency_reason:
                    self.controller.publish_safe_damping()
                    time.sleep(CONTROL_DT)
                    continue

                command = np.zeros(3)
                # Durante la recuperacion usamos la postura de reposo
                # promediada. Mantener el ciclo de marcha con velocidad cero
                # producia varios pasitos pequenos despues de cada orden.
                gait_active = bool(self.motion_remaining)
                if self.motion_remaining:
                    command[:] = self.motion_command
                    self.motion_remaining -= 1
                    if not self.motion_remaining:
                        print("[Fase] Frenando y recuperando equilibrio")
                elif self.settle_remaining:
                    self.settle_remaining -= 1
                    if not self.settle_remaining:
                        print(f"[OK] Movimiento terminado: {self.motion_label}")
                        if self.pending_wave:
                            self.pending_wave = False
                            self.start_wave()

                self.update_upper_body()
                self.update_squat()
                target = np.empty(NUM_MOTORS)
                if self.squat_stages:
                    target[:12] = self.lower_target + self.lower_override_current
                else:
                    target[:12] = self.lower_target
                target[12:] = self.upper_current
                try:
                    self.controller.send(target, self.kp, self.kd)
                except RuntimeError as error:
                    self.enter_emergency(str(error))
                    continue

                if self.policy_step % POLICY_DECIMATION == 0:
                    try:
                        self.update_policy(command, gait_active)
                    except RuntimeError as error:
                        self.enter_emergency(str(error))
                        continue
                if gait_active:
                    self.gait_step += 1
                self.policy_step += 1
        except KeyboardInterrupt:
            print("\n[AVISO] Estabilizador detenido.")
        finally:
            self.controller.publish_safe_damping()
            self.server.close()
            if SOCKET_PATH.exists():
                os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    StabilizerService().run()
