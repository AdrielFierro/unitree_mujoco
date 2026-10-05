import os
import sys


ROBOT = "g1" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1"
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo0" if sys.platform == "darwin" else "lo" # Loopback interface

USE_JOYSTICK = 0 if sys.platform == "darwin" else 1 # Linux joystick device is /dev/input/js0
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = True # Virtual spring band, used for lifting h1
# El lanzador especial de macOS activa esta opcion mediante una variable de
# entorno. La banda sostiene al G1 durante el arranque y se suelta cuando llega
# el primer comando del controlador sin banda.
AUTO_RELEASE_ELASTIC_BAND = os.environ.get("UNITREE_AUTO_RELEASE_BAND") == "1"

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
# El control de pie sin banda usa rigideces altas. Con 5 ms la integracion del
# bipedo se vuelve inestable; 2 ms coincide con el paso del modelo G1 y del
# simulador DDS validado.
UNASSISTED_G1_DT = 0.002
VIEWER_DT = 0.02  # 50 fps for viewer
