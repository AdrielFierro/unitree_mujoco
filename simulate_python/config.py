import os
import sys


LOW_POWER_MODE = os.environ.get("UNITREE_LOW_POWER") == "1"
ROBOT = "g1" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1"
if LOW_POWER_MODE and ROBOT == "g1":
    ROBOT_SCENE = "../unitree_robots/g1/scene_macos_flat.xml"
else:
    ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo0" if sys.platform == "darwin" else "lo" # Loopback interface

USE_JOYSTICK = 0 if sys.platform == "darwin" else 1 # Linux joystick device is /dev/input/js0
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = not LOW_POWER_MODE # Print link, joint and sensor information
ENABLE_ELASTIC_BAND = True # Virtual spring band, used for lifting h1
# El lanzador especial de macOS activa esta opcion mediante una variable de
# entorno. La banda sostiene al G1 durante el arranque y se suelta cuando llega
# el primer comando del controlador sin banda.
AUTO_RELEASE_ELASTIC_BAND = os.environ.get("UNITREE_AUTO_RELEASE_BAND") == "1"
# Con longitud cero la banda sostiene mas que el peso completo del G1 y deja
# los pies en el aire. Esta longitud lo acerca a su altura nominal (0.793 m)
# y permite que el piso cargue parte del peso antes de soltarlo.
AUTO_RELEASE_BAND_LENGTH_M = 0.70
AUTO_RELEASE_MIN_BASE_HEIGHT_M = 0.65
AUTO_RELEASE_MAX_BASE_HEIGHT_M = 0.90
AUTO_RELEASE_MAX_VERTICAL_SPEED_M_S = 0.15
# Dar tiempo a que el controlador valide el estado, complete su transicion
# inicial y empiece a ejecutar la politica antes de retirar el soporte.
AUTO_RELEASE_DELAY_S = 7.0

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
# El control de pie sin banda usa rigideces altas. Con 5 ms la integracion del
# bipedo se vuelve inestable; 2 ms coincide con el paso del modelo G1 y del
# simulador DDS validado.
UNASSISTED_G1_DT = 0.002
VIEWER_DT = 0.04 if LOW_POWER_MODE else 0.02  # 25 or 50 fps
LOWSTATE_DT = 0.004 if LOW_POWER_MODE else None  # 250 Hz or physics rate
HIGHSTATE_DT = 0.02 if LOW_POWER_MODE else None  # 50 Hz or physics rate
PUBLISH_WIRELESS_CONTROLLER = not LOW_POWER_MODE
