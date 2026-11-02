"""Adaptador minimo del tracker whole-body ProtoMotions para el G1 local.

Basado en la interfaz de despliegue de NVLabs ProtoMotions y la integracion
ProtoMotionsTrackerPolicy de RoboJuDo. Solo usa el modelo ONNX, el movimiento
cacheado y el estado DDS; no requiere Isaac Sim ni GPU.
"""

from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch


FUTURE_STEPS = (1, 2, 4, 8)
ANCHOR_BODY_INDEX = 16
WAVE_MOTION_INDEX = 36


def quat_multiply_xyzw(left, right):
    lx, ly, lz, lw = np.moveaxis(np.asarray(left), -1, 0)
    rx, ry, rz, rw = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def yaw_from_xyzw(quaternion):
    x, y, z, w = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def slerp_xyzw(first, second, fraction):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return (result / np.linalg.norm(result)).astype(np.float32)
    angle = np.arccos(dot)
    result = (
        np.sin((1.0 - fraction) * angle) * first
        + np.sin(fraction * angle) * second
    ) / np.sin(angle)
    return result.astype(np.float32)


class ProtoMotionTracker:
    """Ejecuta una secuencia whole-body equilibrada a 50 Hz."""

    def __init__(self, project_root: Path, motion_index=WAVE_MOTION_INDEX):
        assets = project_root / "third_party/protomotions"
        self.session = ort.InferenceSession(
            str(assets / "unified_pipeline.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

        library = torch.load(
            assets / "g1_bones_seed_mini.pt",
            map_location="cpu",
            weights_only=False,
        )
        start = int(library["length_starts"][motion_index])
        count = int(library["motion_num_frames"][motion_index])
        stop = start + count
        self.motion_name = str(library["motion_files"][motion_index]).split("/")[-1]
        self.motion_dt = float(library["motion_dt"][motion_index])
        self.dof_pos = library["dps"][start:stop].numpy().astype(np.float32)
        self.dof_vel = library["dvs"][start:stop].numpy().astype(np.float32)
        self.anchor_rot = (
            library["grs"][start:stop, ANCHOR_BODY_INDEX]
            .numpy()
            .astype(np.float32)
        )
        self.duration = (count - 1) * self.motion_dt

        self.kp = np.array(
            [
                40.179238, 99.09843, 40.179238, 99.09843, 28.501247, 28.501247,
                40.179238, 99.09843, 40.179238, 99.09843, 28.501247, 28.501247,
                40.179238, 28.501247, 28.501247,
                14.250623, 14.250623, 14.250623, 14.250623, 14.250623,
                16.778328, 16.778328, 14.250623, 14.250623, 14.250623,
                14.250623, 14.250623, 16.778328, 16.778328,
            ],
            dtype=float,
        )
        self.kd = np.array(
            [
                2.55789, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
                2.55789, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
                2.55789, 1.814446, 1.814446,
                0.907223, 0.907223, 0.907223, 0.907223, 0.907223,
                1.068142, 1.068142, 0.907223, 0.907223, 0.907223,
                0.907223, 0.907223, 1.068142, 1.068142,
            ],
            dtype=float,
        )

        self.fk_model = mujoco.MjModel.from_xml_path(
            str(project_root / "unitree_robots/g1/scene_29dof.xml")
        )
        self.fk_data = mujoco.MjData(self.fk_model)
        self.torso_id = mujoco.mj_name2id(
            self.fk_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.reset()

    def reset(self):
        self.time = 0.0
        self.heading_offset = None
        self.previous_target = np.zeros(29, dtype=np.float32)

    def _sample(self, time_value):
        position = np.clip(time_value / self.motion_dt, 0.0, len(self.dof_pos) - 1)
        low = int(np.floor(position))
        high = min(low + 1, len(self.dof_pos) - 1)
        fraction = float(position - low)
        return (
            self.dof_pos[low] * (1.0 - fraction) + self.dof_pos[high] * fraction,
            self.dof_vel[low] * (1.0 - fraction) + self.dof_vel[high] * fraction,
            slerp_xyzw(self.anchor_rot[low], self.anchor_rot[high], fraction),
        )

    def _torso_quaternion(self, q, pelvis_quaternion_wxyz):
        self.fk_data.qpos[:] = self.fk_model.qpos0
        self.fk_data.qpos[3:7] = pelvis_quaternion_wxyz
        self.fk_data.qpos[7:36] = q
        mujoco.mj_forward(self.fk_model, self.fk_data)
        w, x, y, z = self.fk_data.xquat[self.torso_id]
        return np.array([x, y, z, w], dtype=np.float32)

    def step(self, snapshot, control_dt=0.02):
        q, dq, pelvis_quaternion, gyroscope, _, _ = snapshot
        torso_quaternion = self._torso_quaternion(q, pelvis_quaternion)
        if self.heading_offset is None:
            delta_yaw = yaw_from_xyzw(torso_quaternion) - yaw_from_xyzw(
                self.anchor_rot[0]
            )
            self.heading_offset = np.array(
                [0.0, 0.0, np.sin(delta_yaw / 2.0), np.cos(delta_yaw / 2.0)],
                dtype=np.float32,
            )

        future_pos = []
        future_vel = []
        future_rot = []
        for step in FUTURE_STEPS:
            pos, vel, rotation = self._sample(self.time + step * control_dt)
            future_pos.append(pos)
            future_vel.append(vel)
            future_rot.append(quat_multiply_xyzw(self.heading_offset, rotation))

        values = {
            "current_anchor_rot": torso_quaternion[None],
            "current_dof_pos": np.asarray(q, dtype=np.float32)[None],
            "current_dof_vel": np.asarray(dq, dtype=np.float32)[None],
            "current_root_local_ang_vel": np.asarray(gyroscope, dtype=np.float32)[None],
            "historical_processed_actions": self.previous_target[None, None],
            "mimic_future_anchor_rot": np.asarray(future_rot, dtype=np.float32)[None],
            "mimic_future_dof_pos": np.asarray(future_pos, dtype=np.float32)[None],
            "mimic_future_dof_vel": np.asarray(future_vel, dtype=np.float32)[None],
        }
        outputs = self.session.run(
            self.output_names,
            {name: values[name] for name in self.input_names},
        )
        target = np.asarray(outputs[1], dtype=np.float32).squeeze()
        stiffness = np.asarray(outputs[2], dtype=np.float32).squeeze()
        damping = np.asarray(outputs[3], dtype=np.float32).squeeze()
        if not all(np.all(np.isfinite(item)) for item in (target, stiffness, damping)):
            raise RuntimeError("el tracker whole-body produjo NaN o Inf")
        self.previous_target = target.copy()
        self.time += control_dt
        return target, stiffness, damping, self.time >= self.duration

