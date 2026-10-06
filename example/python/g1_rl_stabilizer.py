"""Servicio persistente de equilibrio y comandos para el G1 simulado."""

from pathlib import Path
import os
import socket

import numpy as np
import torch

from g1_rl_activity import (
    CONTROL_DT,
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
FORWARD_STEPS = round(0.8 / CONTROL_DT)
SETTLE_STEPS = round(1.6 / CONTROL_DT)
IDLE_PHASE = 0.35


class StabilizerService:
    def __init__(self):
        self.controller = DdsController()
        current = self.controller.wait_for_state()
        print("[Fase] Adoptando postura inicial")
        self.controller.transition(current, NEUTRAL, 3.0)

        self.policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
        self.policy.eval()
        self.action = np.zeros(12, dtype=np.float32)
        self.lower_target = POLICY_DEFAULT.copy()
        self.observation = np.zeros(47, dtype=np.float32)
        self.policy_step = 0
        self.gait_step = 0

        self.kp = HIGH_KP.copy()
        self.kd = HIGH_KD.copy()
        self.kp[:12] = POLICY_KP
        self.kd[:12] = POLICY_KD

        self.forward_remaining = 0
        self.settle_remaining = 0
        self.pending_wave = False
        self.wave_stages = []
        self.wave_index = 0
        self.wave_step = 0
        self.upper_start = NEUTRAL[12:].copy()
        self.upper_target = NEUTRAL[12:].copy()
        self.upper_current = NEUTRAL[12:].copy()
        self.running = True

        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if SOCKET_PATH.exists():
            os.unlink(SOCKET_PATH)
        self.server.bind(str(SOCKET_PATH))
        self.server.listen(4)
        self.server.setblocking(False)

    def busy(self):
        return bool(
            self.forward_remaining
            or self.settle_remaining
            or self.pending_wave
            or self.wave_stages
        )

    def status(self):
        if self.forward_remaining:
            phase = "caminando"
        elif self.settle_remaining:
            phase = "estabilizando"
        elif self.wave_stages:
            phase = "saludando"
        else:
            phase = "parado y equilibrando"
        return (
            f"{phase}; inclinacion maxima="
            f"{self.controller.max_tilt_deg:.1f} grados"
        )

    def start_steps(self, then_wave=False):
        # Cada orden debe empezar al principio del ciclo de marcha. Antes se
        # reutilizaba la fase global del estabilizador y algunas ordenes
        # comenzaban a mitad de zancada, moviendo los pies casi en el lugar.
        self.gait_step = 0
        self.action.fill(0.0)
        self.lower_target = POLICY_DEFAULT.copy()
        self.forward_remaining = FORWARD_STEPS
        self.settle_remaining = SETTLE_STEPS
        self.pending_wave = then_wave
        print("[Comando] Dando dos pasos hacia adelante")

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
        if self.busy():
            return "OCUPADO " + self.status()
        if request in ("pasos", "caminar"):
            self.start_steps()
            return "OK comando pasos aceptado"
        if request == "saludo":
            self.start_wave()
            return "OK comando saludo aceptado"
        if request in ("actividad", "rutina"):
            self.start_steps(then_wave=True)
            return "OK actividad aceptada: pasos y saludo"
        return "ERROR usa: actividad, pasos, saludo, estado o salir"

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

    def update_policy(self, command, gait_active):
        snapshot = self.controller.snapshot()
        if snapshot is None:
            return
        q, dq, quaternion, gyroscope, _, _ = snapshot
        self.observation[:3] = gyroscope * 0.25
        self.observation[3:6] = gravity_orientation(quaternion)
        self.observation[6:9] = command * POLICY_COMMAND_SCALE
        self.observation[9:21] = q[:12] - POLICY_DEFAULT
        self.observation[21:33] = dq[:12] * 0.05
        self.observation[33:45] = self.action
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
        self.lower_target = self.action * 0.25 + POLICY_DEFAULT

    def run(self):
        print("[OK] Estabilizador activo y esperando comandos.")
        print(f"[OK] Socket local: {SOCKET_PATH}")
        try:
            while self.running:
                self.poll_commands()
                command = np.zeros(3)
                gait_active = bool(
                    self.forward_remaining or self.settle_remaining
                )
                if self.forward_remaining:
                    command[0] = 0.25
                    self.forward_remaining -= 1
                    if not self.forward_remaining:
                        print("[Fase] Frenando y recuperando equilibrio")
                elif self.settle_remaining:
                    self.settle_remaining -= 1
                    if not self.settle_remaining:
                        print("[OK] Dos pasos terminados")
                        if self.pending_wave:
                            self.pending_wave = False
                            self.start_wave()

                self.update_upper_body()
                target = np.empty(NUM_MOTORS)
                target[:12] = self.lower_target
                target[12:] = self.upper_current
                self.controller.send(target, self.kp, self.kd)

                if self.policy_step % POLICY_DECIMATION == 0:
                    self.update_policy(command, gait_active)
                if gait_active:
                    self.gait_step += 1
                self.policy_step += 1
        except KeyboardInterrupt:
            print("\n[AVISO] Estabilizador detenido.")
        finally:
            self.server.close()
            if SOCKET_PATH.exists():
                os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    StabilizerService().run()
