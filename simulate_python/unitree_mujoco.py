import time
import signal
import mujoco
import mujoco.viewer
import numpy as np
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()
stop_event = threading.Event()
simulation_errors = []

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    if config.AUTO_RELEASE_ELASTIC_BAND:
        # En el modo autonomo, el PD interno se aplica antes del primer paso
        # fisico. No hace falta sostener el torso y los pies conservan toda su
        # carga contra el suelo desde el inicio.
        elastic_band.enable = False
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = (
    config.UNASSISTED_G1_DT
    if config.AUTO_RELEASE_ELASTIC_BAND and config.ROBOT == "g1"
    else config.SIMULATE_DT
)
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model
    unitree = None
    try:
        ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
        unitree = UnitreeSdk2Bridge(mj_model, mj_data)
        band_auto_released = config.AUTO_RELEASE_ELASTIC_BAND
        band_release_steps = None

        # Control interno de arranque: mantiene las articulaciones en la postura
        # inicial mientras el usuario abre la segunda Terminal. Se reemplaza por
        # rt/lowcmd apenas comienza el controlador externo.
        actuator_joints = mj_model.actuator_trnid[:, 0]
        qpos_adr = mj_model.jnt_qposadr[actuator_joints]
        qvel_adr = mj_model.jnt_dofadr[actuator_joints]
        startup_target = mj_data.qpos[qpos_adr].copy()
        startup_kp = np.array(
            [800] * 15 + [100] * 4 + [50] * 3 + [100] * 4 + [50] * 3,
            dtype=float,
        )
        startup_kd = np.array(
            [15] * 15 + [2.5] * 4 + [1.25] * 3 + [2.5] * 4 + [1.25] * 3,
            dtype=float,
        )
        ctrl_low, ctrl_high = mj_model.actuator_ctrlrange.T

        if config.USE_JOYSTICK:
            unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
        if config.PRINT_SCENE_INFORMATION:
            unitree.PrintSceneInformation()

        while not stop_event.is_set() and viewer.is_running():
            step_start = time.perf_counter()

            # El context manager siempre libera el candado, incluso si MuJoCo
            # genera una excepcion durante el cierre de la ventana.
            with locker:
                if config.ENABLE_ELASTIC_BAND:
                    if config.AUTO_RELEASE_ELASTIC_BAND and not band_auto_released:
                        if unitree.lowcmd_received and band_release_steps is None:
                            band_release_steps = max(
                                1, round(0.5 / mj_model.opt.timestep)
                            )
                        if band_release_steps is not None:
                            band_release_steps -= 1
                            if band_release_steps <= 0:
                                elastic_band.enable = False
                                band_auto_released = True
                                print(
                                    "Elastic band: automatically released "
                                    "by controller"
                                )
                    band_force = elastic_band.Advance(
                        mj_data.qpos[:3], mj_data.qvel[:3]
                    )
                    mj_data.xfrc_applied[band_attached_link, :] = 0.0
                    if elastic_band.enable:
                        mj_data.xfrc_applied[band_attached_link, :3] = band_force

                if config.AUTO_RELEASE_ELASTIC_BAND and not unitree.lowcmd_received:
                    control = startup_kp * (
                        startup_target - mj_data.qpos[qpos_adr]
                    )
                    control -= startup_kd * mj_data.qvel[qvel_adr]
                    mj_data.ctrl[:] = np.clip(control, ctrl_low, ctrl_high)
                else:
                    unitree.ApplyLowCmd()
                mujoco.mj_step(mj_model, mj_data)
                unitree.StepCompleted()

            time_until_next_step = mj_model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if time_until_next_step > 0:
                stop_event.wait(time_until_next_step)
    except BaseException as error:
        simulation_errors.append(error)
        stop_event.set()
    finally:
        if unitree is not None:
            try:
                unitree.Close()
            except BaseException as error:
                simulation_errors.append(error)
        stop_event.set()


def request_stop(_signal=None, _frame=None):
    stop_event.set()


def main():
    signal.signal(signal.SIGTERM, request_stop)
    sim_thread = threading.Thread(
        target=SimulationThread,
        name="mujoco_physics",
        daemon=True,
    )
    sim_thread.start()

    try:
        # En macOS el hilo principal conserva el ciclo de vida del visor. La
        # fisica corre aparte, pero sync y el cierre se coordinan desde aqui.
        while (
            viewer.is_running()
            and sim_thread.is_alive()
            and not stop_event.is_set()
        ):
            with locker:
                viewer.sync()
            stop_event.wait(config.VIEWER_DT)
    except KeyboardInterrupt:
        print("\nClosing simulator...")
    finally:
        stop_event.set()
        sim_thread.join(timeout=5.0)
        viewer.close()

    if sim_thread.is_alive():
        print("[WARN] Physics thread did not finish within 5 seconds.")
    if simulation_errors:
        raise simulation_errors[0]
    print("Simulator closed cleanly.")


if __name__ == "__main__":
    main()
