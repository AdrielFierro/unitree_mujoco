"""Servicio persistente para el G1 con una politica de locomocion sin gait.

La politica y el armado de la observacion se adaptaron de UnitreeWoGaitPolicy
de RoboJuDo (CC BY 4.0), basado en Unitree RL Lab (Apache 2.0). A diferencia
de la politica ciclica anterior, no recibe una fase de marcha: con comando
cero fue entrenada para permanecer parada sin dar mini pasos.
"""

from collections import deque
from pathlib import Path
import os
import socket
import time

import numpy as np
import torch

from g1_protomotions_tracker import ProtoMotionTracker
from g1_rl_activity import (
    CONTROL_DT,
    LOW_POWER_MODE,
    NUM_MOTORS,
    POLICY_DECIMATION,
    PROJECT_ROOT,
    DdsController,
    gravity_orientation,
)
SOCKET_PATH = Path("/tmp/unitree_g1_stabilizer.sock")
POLICY_PATH = PROJECT_ROOT / "third_party/robojudo/policy_wo_gait.pt"
POLICY_HISTORY_LENGTH = 5
# Orden de la politica exportada por Unitree RL Lab -> indices LowCmd del G1.
POLICY_TO_MOTOR = np.array(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
        15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
        18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=int,
)
POLICY_DEFAULT = np.array(
    [
        -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3,
        0.3, 0.3, -0.2, -0.2, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0,
        0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)
POLICY_KP = np.array(
    [
        100, 100, 200, 100, 100, 40, 100, 100, 40, 150, 150,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40, 40,
    ],
    dtype=float,
)
POLICY_KD = np.array(
    [
        2, 2, 5, 2, 2, 5, 2, 2, 5, 4, 4,
        10, 10, 2, 2, 10, 10, 2, 2, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10,
    ],
    dtype=float,
)
POLICY_DEFAULT_MOTOR = np.empty(NUM_MOTORS, dtype=float)
POLICY_DEFAULT_MOTOR[POLICY_TO_MOTOR] = POLICY_DEFAULT
POLICY_KP_MOTOR = np.empty(NUM_MOTORS, dtype=float)
POLICY_KP_MOTOR[POLICY_TO_MOTOR] = POLICY_KP
POLICY_KD_MOTOR = np.empty(NUM_MOTORS, dtype=float)
POLICY_KD_MOTOR[POLICY_TO_MOTOR] = POLICY_KD
MOTION_STEPS = round(0.8 / CONTROL_DT)
SETTLE_STEPS = round(1.2 / CONTROL_DT)
SETTLE_MIN_STEPS = round(0.35 / CONTROL_DT)
SETTLE_STABLE_STEPS = round(0.20 / CONTROL_DT)
SETTLE_TILT_DEG = 5.0
SETTLE_GYRO_RAD_S = 0.40
SETTLE_LEG_SPEED_RAD_S = 1.0
STARTUP_TILT_LIMIT_DEG = 15.0
COMMAND_TILT_LIMIT_DEG = 20.0
UNSTABLE_TILT_DEG = 25.0
UNSTABLE_TIME_S = 0.5
EMERGENCY_TILT_DEG = 35.0
EMERGENCY_JOINT_SPEED = 50.0
POLICY_OBSERVATION_LIMIT = 100.0
POLICY_ACTION_LIMIT = 3.0
TRACKER_BLEND_STEPS = round(2.0 / 0.020)
TRACKER_BLEND_OUT_STEPS = round(4.0 / 0.020)
MOTION_COMMANDS = {
    "adelante": (np.array([0.32, 0.0, 0.0]), "avanzando"),
    "atras": (np.array([-0.24, 0.0, 0.0]), "retrocediendo"),
    "izquierda": (np.array([0.0, 0.22, 0.0]), "desplazandose a la izquierda"),
    "derecha": (np.array([0.0, -0.22, 0.0]), "desplazandose a la derecha"),
    "girar_izquierda": (np.array([0.0, 0.0, 0.38]), "girando a la izquierda"),
    "girar_derecha": (np.array([0.0, 0.0, -0.38]), "girando a la derecha"),
}
# Sentadilla moderada compatible con la postura base de la politica. La
# demostracion autonoma conserva por separado su sentadilla profunda.
MENU_SQUAT_OFFSET = np.array([-0.18, 0.0, 0.0, 0.38, -0.18, 0.0] * 2)
SQUAT_MOTOR_INDICES = np.array([0, 3, 4, 6, 9, 10], dtype=int)
GESTURE_BALANCE_GAIN = 1.0
GESTURE_LEG_CORRECTION_LIMIT = 0.45
GESTURE_WAIST_CORRECTION_LIMIT = 0.25
WAVE_WAIST_ROLL_COMPENSATION = -0.076
WAVE_WAIST_PITCH_COMPENSATION = -0.021


def wave_offsets():
    """Saludo moderado relativo a la salida equilibrada de la politica."""
    raised = np.zeros(17, dtype=float)
    # Postura visible del saludo original. El estabilizador compensa el cambio
    # del centro de masa con piernas y cintura; no se achica el gesto.
    raised[10] = -0.30
    raised[11] = -1.46  # roll absoluto: -1.71
    raised[12] = -1.57
    raised[13] = -0.83  # codo absoluto: 0.14
    raised[14] = 0.15  # wrist roll: -0.15 -> 0.00
    left = raised.copy()
    left[15] = -0.35  # muneca derecha: pitch
    right = raised.copy()
    right[15] = 0.35
    return [
        ("levantando el brazo", 2.00, raised),
        ("saludando a un lado", 0.55, left),
        ("saludando al otro", 0.55, right),
        ("saludando a un lado", 0.55, left),
        ("saludando al otro", 0.55, right),
        ("bajando el brazo", 2.00, np.zeros(17, dtype=float)),
    ]


class StabilizerService:
    def __init__(self):
        self.controller = DdsController()
        self.kp = POLICY_KP_MOTOR.copy()
        self.kd = POLICY_KD_MOTOR.copy()

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
        self.controller.transition(
            current,
            POLICY_DEFAULT_MOTOR,
            3.0,
            self.kp,
            self.kd,
        )

        if not POLICY_PATH.exists():
            raise RuntimeError(
                "Falta la politica sin gait: " + str(POLICY_PATH)
            )
        self.policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
        self.policy.eval()
        print("[Fase] Cargando tracker whole-body para gestos")
        self.tracker = ProtoMotionTracker(PROJECT_ROOT)
        self.tracker_active = False
        self.tracker_phase = ""
        self.tracker_phase_step = 0
        self.tracker_target = POLICY_DEFAULT_MOTOR.copy()
        self.tracker_kp = self.kp.copy()
        self.tracker_kd = self.kd.copy()
        self.tracker_last_target = POLICY_DEFAULT_MOTOR.copy()
        self.action = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.policy_target = POLICY_DEFAULT_MOTOR.copy()
        self.observation_history = deque(maxlen=POLICY_HISTORY_LENGTH)
        empty_frame = (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(NUM_MOTORS, dtype=np.float32),
            np.zeros(NUM_MOTORS, dtype=np.float32),
            np.zeros(NUM_MOTORS, dtype=np.float32),
        )
        for _ in range(POLICY_HISTORY_LENGTH):
            self.observation_history.append(
                tuple(values.copy() for values in empty_frame)
            )
        self.policy_step = 0

        self.motion_remaining = 0
        self.motion_command = np.zeros(3)
        self.motion_label = ""
        self.settle_remaining = 0
        self.settle_elapsed = 0
        self.settle_stable_steps = 0
        self.pending_wave = False
        self.wave_stages = []
        self.wave_index = 0
        self.wave_step = 0
        self.upper_start = np.zeros(17, dtype=float)
        self.upper_target = np.zeros(17, dtype=float)
        self.upper_current = np.zeros(17, dtype=float)
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
            or self.tracker_active
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
        self.tracker_active = False
        self.squat_stages = []
        self.controller.publish_safe_damping()

    def status(self):
        snapshot = self.controller.snapshot()
        state_error, current_tilt = self.state_error(snapshot)
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
        elif self.tracker_active:
            phase = "saludando"
        else:
            phase = "parado y equilibrando sin ciclo de marcha"
        _, angular_speed, leg_speed = self.motion_metrics(snapshot)
        return (
            f"{phase}; inclinacion actual={current_tilt:.1f} grados; "
            f"velocidad angular={angular_speed:.2f} rad/s; "
            f"velocidad pierna={leg_speed:.2f} rad/s; "
            f"inclinacion maxima={self.controller.max_tilt_deg:.1f} grados"
        )

    def motion_metrics(self, snapshot=None):
        if snapshot is None:
            snapshot = self.controller.snapshot()
        if snapshot is None:
            return 180.0, float("inf"), float("inf")
        _, dq, quaternion, gyroscope, _, _ = snapshot
        return (
            self.controller.tilt_deg(quaternion),
            float(np.linalg.norm(gyroscope)),
            float(np.max(np.abs(dq[:12]))),
        )

    def settle_is_complete(self):
        self.settle_elapsed += 1
        tilt, angular_speed, leg_speed = self.motion_metrics()
        if (
            self.settle_elapsed >= SETTLE_MIN_STEPS
            and tilt <= SETTLE_TILT_DEG
            and angular_speed <= SETTLE_GYRO_RAD_S
            and leg_speed <= SETTLE_LEG_SPEED_RAD_S
        ):
            self.settle_stable_steps += 1
        else:
            self.settle_stable_steps = 0
        return (
            self.settle_stable_steps >= SETTLE_STABLE_STEPS
            or self.settle_remaining <= 0
        )

    def finish_motion(self):
        self.settle_remaining = 0
        print(f"[OK] Movimiento terminado: {self.motion_label}")
        if self.pending_wave:
            self.pending_wave = False
            self.start_wave()

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
        command, label = MOTION_COMMANDS[command_name]
        self.motion_command = command.copy()
        self.motion_label = label
        self.motion_remaining = MOTION_STEPS
        self.settle_remaining = SETTLE_STEPS
        self.settle_elapsed = 0
        self.settle_stable_steps = 0
        self.pending_wave = then_wave
        print(f"[Comando] {label.capitalize()}")

    def start_squat(self):
        self.squat_stages = [
            ("bajando", 1.80, MENU_SQUAT_OFFSET),
            ("sosteniendo", 2.00, MENU_SQUAT_OFFSET),
            ("subiendo", 1.80, np.zeros(12)),
        ]
        self.squat_index = 0
        self.squat_step = 0
        self.lower_override_start = np.zeros(12)
        self.lower_override_current = self.lower_override_start.copy()
        self.lower_override_target = self.squat_stages[0][2].copy()
        print("[Comando] Comenzando la sentadilla")

    def start_wave(self):
        self.tracker.reset()
        self.tracker_active = True
        self.tracker_phase = "blend_in"
        self.tracker_phase_step = 0
        snapshot = self.controller.snapshot()
        if snapshot is not None:
            self.tracker_last_target = snapshot[0].copy()
        print(
            "[Comando] Comenzando saludo whole-body equilibrado: "
            + self.tracker.motion_name
        )

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
            self.upper_current = np.zeros(17, dtype=float)
            print("[OK] Saludo terminado; sigo equilibrando al G1")
            return
        self.wave_step = 0
        self.upper_start = self.upper_current.copy()
        self.upper_target = self.wave_stages[self.wave_index][2].copy()

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

    @staticmethod
    def smooth_blend(value):
        value = float(np.clip(value, 0.0, 1.0))
        return 10 * value**3 - 15 * value**4 + 6 * value**5

    def update_tracker(self):
        if not self.tracker_active:
            return
        snapshot = self.controller.snapshot()
        if snapshot is None:
            return

        if self.tracker_phase == "blend_in":
            target, kp, kd, _ = self.tracker.step(snapshot)
            # Durante la entrada se mantiene el primer cuadro del movimiento.
            self.tracker.time = 0.0
            self.tracker_phase_step += 1
            blend = self.smooth_blend(
                self.tracker_phase_step / TRACKER_BLEND_STEPS
            )
            self.tracker_target = (
                self.tracker_last_target * (1.0 - blend) + target * blend
            )
            self.tracker_kp = self.kp * (1.0 - blend) + kp * blend
            self.tracker_kd = self.kd * (1.0 - blend) + kd * blend
            if self.tracker_phase_step >= TRACKER_BLEND_STEPS:
                self.tracker_phase = "tracking"
                self.tracker_phase_step = 0
                print("[Fase] Ejecutando el saludo whole-body")
            return

        if self.tracker_phase == "tracking":
            target, kp, kd, done = self.tracker.step(snapshot)
            self.tracker_target = target
            self.tracker_kp = kp
            self.tracker_kd = kd
            if done:
                self.tracker_phase = "blend_out"
                self.tracker_phase_step = 0
                self.tracker_last_target = target.copy()
                self.reset_policy_history(snapshot)
                print("[Fase] Volviendo al estabilizador de locomocion")
            return

        self.tracker_phase_step += 1
        blend = self.smooth_blend(
            self.tracker_phase_step / TRACKER_BLEND_OUT_STEPS
        )
        self.tracker_target = (
            self.tracker_last_target * (1.0 - blend)
            + self.policy_target * blend
        )
        self.tracker_kp = self.tracker.kp * (1.0 - blend) + self.kp * blend
        self.tracker_kd = self.tracker.kd * (1.0 - blend) + self.kd * blend
        if self.tracker_phase_step >= TRACKER_BLEND_OUT_STEPS:
            self.tracker_active = False
            self.tracker_phase = ""
            print("[OK] Saludo terminado; sigo equilibrando al G1")

    def update_policy(self, command):
        snapshot = self.controller.snapshot()
        if snapshot is None:
            return
        q, dq, quaternion, gyroscope, _, _ = snapshot
        if not all(
            np.all(np.isfinite(values))
            for values in (q, dq, quaternion, gyroscope, self.action)
        ):
            raise RuntimeError("estado u accion de politica con NaN o Inf")

        # La red espera cinco cuadros, agrupados por tipo de dato (primero
        # las cinco velocidades angulares, despues las cinco orientaciones,
        # etc.), en el orden articular usado durante el entrenamiento.
        frame = (
            (gyroscope * 0.2).astype(np.float32),
            gravity_orientation(quaternion).astype(np.float32),
            np.asarray(command, dtype=np.float32),
            (q[POLICY_TO_MOTOR] - POLICY_DEFAULT).astype(np.float32),
            (dq[POLICY_TO_MOTOR] * 0.05).astype(np.float32),
            self.action.copy(),
        )
        self.observation_history.append(frame)
        observation = np.concatenate(
            [
                np.concatenate(items, axis=0)
                for items in zip(*self.observation_history)
            ]
        ).astype(np.float32, copy=False)
        np.clip(
            observation,
            -POLICY_OBSERVATION_LIMIT,
            POLICY_OBSERVATION_LIMIT,
            out=observation,
        )
        with torch.no_grad():
            tensor = torch.from_numpy(observation).unsqueeze(0)
            action = self.policy(tensor).numpy().squeeze()
        if action.shape != (NUM_MOTORS,) or not np.all(np.isfinite(action)):
            raise RuntimeError("la politica produjo NaN o Inf")
        self.action[:] = action
        np.clip(
            self.action,
            -POLICY_ACTION_LIMIT,
            POLICY_ACTION_LIMIT,
            out=self.action,
        )
        policy_target = self.action * 0.25 + POLICY_DEFAULT
        self.policy_target[POLICY_TO_MOTOR] = policy_target

    def reset_policy_history(self, snapshot):
        q, dq, quaternion, gyroscope, _, _ = snapshot
        self.action.fill(0.0)
        frame = (
            (gyroscope * 0.2).astype(np.float32),
            gravity_orientation(quaternion).astype(np.float32),
            np.zeros(3, dtype=np.float32),
            (q[POLICY_TO_MOTOR] - POLICY_DEFAULT).astype(np.float32),
            (dq[POLICY_TO_MOTOR] * 0.05).astype(np.float32),
            self.action.copy(),
        )
        self.observation_history.clear()
        for _ in range(POLICY_HISTORY_LENGTH):
            self.observation_history.append(
                tuple(values.copy() for values in frame)
            )
        self.policy_target = q.copy()

    def target_with_gesture_balance(self):
        """Conserva y refuerza la respuesta aprendida durante movimientos."""
        target = self.policy_target.copy()
        if not (self.wave_stages or self.squat_stages):
            return target

        lower_delta = self.policy_target[:12] - POLICY_DEFAULT_MOTOR[:12]
        waist_delta = self.policy_target[12:15] - POLICY_DEFAULT_MOTOR[12:15]
        target[:12] = POLICY_DEFAULT_MOTOR[:12] + np.clip(
            lower_delta * GESTURE_BALANCE_GAIN,
            -GESTURE_LEG_CORRECTION_LIMIT,
            GESTURE_LEG_CORRECTION_LIMIT,
        )
        target[12:15] = POLICY_DEFAULT_MOTOR[12:15] + np.clip(
            waist_delta * GESTURE_BALANCE_GAIN,
            -GESTURE_WAIST_CORRECTION_LIMIT,
            GESTURE_WAIST_CORRECTION_LIMIT,
        )
        if self.wave_stages:
            arm_fraction = float(
                np.clip(abs(self.upper_current[11]) / 1.46, 0.0, 1.0)
            )
            target[13] += WAVE_WAIST_ROLL_COMPENSATION * arm_fraction
            target[14] += WAVE_WAIST_PITCH_COMPENSATION * arm_fraction
        return target

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
                if self.motion_remaining:
                    command[:] = self.motion_command
                    self.motion_remaining -= 1
                    if not self.motion_remaining:
                        print("[Fase] Frenando y recuperando equilibrio")
                elif self.settle_remaining:
                    self.settle_remaining -= 1
                    if self.settle_is_complete():
                        self.finish_motion()

                self.update_upper_body()
                self.update_squat()
                target = self.target_with_gesture_balance()
                active_kp = self.kp
                active_kd = self.kd
                if self.tracker_active:
                    target = self.tracker_target.copy()
                    active_kp = self.tracker_kp
                    active_kd = self.tracker_kd
                if self.squat_stages:
                    target[SQUAT_MOTOR_INDICES] += (
                        self.lower_override_current[SQUAT_MOTOR_INDICES]
                    )
                if self.wave_stages:
                    target[22:29] = (
                        POLICY_DEFAULT_MOTOR[22:29]
                        + self.upper_current[10:17]
                    )
                try:
                    self.controller.send(target, active_kp, active_kd)
                except RuntimeError as error:
                    self.enter_emergency(str(error))
                    continue

                if self.policy_step % POLICY_DECIMATION == 0:
                    try:
                        self.update_policy(command)
                        self.update_tracker()
                    except RuntimeError as error:
                        self.enter_emergency(str(error))
                        continue
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
