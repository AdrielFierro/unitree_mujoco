# G1 nativo en macOS con MuJoCo

Esta adaptación ejecuta el G1 de forma nativa en macOS, sin Docker. El modo
autónomo no necesita banda elástica ni un segundo proceso controlador. Para
caminar utiliza la política neuronal preentrenada oficial de
[`unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym), incluida
como submódulo y distribuida por Unitree bajo licencia BSD-3-Clause.

La demostración realiza esta secuencia:

1. Adopta la postura inicial.
2. Da dos pasos físicos hacia adelante (aproximadamente 20–25 cm en total).
3. Hace una sentadilla.
4. Saluda con el brazo derecho.
5. Permanece parado hasta cerrar la ventana.

## Preparación

Se verificó en Apple Silicon usando Python 3.9, MuJoCo 3.3.6, PyTorch 2.8 y
`unitree_sdk2_python`. Al clonar, incluir los submódulos:

```bash
git clone --recurse-submodules https://github.com/AdrielFierro/unitree_mujoco.git
```

Los submódulos incluyen tanto la política de locomoción como el fork del SDK
Python con el temporizador compatible con macOS. Si el repositorio ya estaba
clonado:

```bash
git submodule update --init --recursive
```

Desde la raíz de este repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install mujoco==3.3.6 cyclonedds==0.10.2 pygame==2.6.1 "numpy<2.1" torch==2.8.0
python -m pip install --no-build-isolation -e ./third_party/unitree_sdk2_python
```

El submódulo del SDK apunta al fork que implementa una alternativa a `timerfd`
para macOS. La implementación Linux original de ese temporizador no funciona
en Darwin.

## Ejecución

La opción recomendada separa física, equilibrio y órdenes en tres Terminales.
Cada proceso tiene una única responsabilidad y solo el estabilizador publica
comandos de motor por DDS.

Terminal 1 — abrir MuJoCo y dejarlo abierto:

```bash
./iniciar_simulador_g1_mac.command
```

Terminal 2 — iniciar la política de equilibrio y dejarla ejecutándose:

```bash
./iniciar_estabilizador_g1_mac.command
```

Terminal 3 — enviar una actividad cada vez que se desee:

```bash
./mandar_comando_g1_mac.command actividad
```

El mismo comando puede ejecutarse nuevamente sobre el mismo simulador. También
se aceptan órdenes individuales:

```bash
./mandar_comando_g1_mac.command pasos
./mandar_comando_g1_mac.command saludo
./mandar_comando_g1_mac.command estado
./mandar_comando_g1_mac.command salir
```

`salir` detiene el estabilizador. Después se puede cerrar la ventana de MuJoCo
para terminar el simulador. No hay que cerrar la Terminal completa.

`actividad` realiza dos pasos y un saludo. La sentadilla profunda todavía no
se envía al estabilizador porque la política de locomoción no fue entrenada
para modificar la altura del torso; forzar esa referencia haría que el G1
salga de la distribución de entrenamiento y pueda caer.

### Ejecución autónoma en un solo proceso

La demostración anterior, que incluye sentadilla, sigue disponible como modo
autónomo:

```bash
./iniciar_g1_sin_banda_mac.command
```

Para omitir los pasos y ejecutar directamente la rutina:

```bash
./iniciar_g1_sin_banda_mac.command --sin-pasos
```

También se puede seleccionar un movimiento:

```bash
./iniciar_g1_sin_banda_mac.command --movimiento sentadilla
./iniciar_g1_sin_banda_mac.command --movimiento saludo
```

Se cierra con el botón rojo de la ventana o con `Control+C` en la Terminal.

### Probar el equilibrio con un empujón

Con el estabilizador activo, hacer doble clic sobre la pelvis del G1. Mantener
la tecla `Control` —no `Command`— y arrastrar brevemente con el botón derecho.
Un desplazamiento corto aplica una fuerza pequeña; al soltar, la fuerza deja de
aplicarse. `F1` abre la ayuda de controles de MuJoCo.

Conviene seleccionar la pelvis y no `torso_link`, porque este último es el
punto reservado para la banda elástica de arranque. La banda se libera
automáticamente cuando comienza a publicar el estabilizador.

## Alcance y seguridad

Durante la marcha los controladores no inyectan `xfrc_applied` ni ninguna
fuerza externa automática: la
política recibe velocidad angular, orientación gravitatoria, comando de
velocidad, posiciones y velocidades articulares, fase de marcha y su acción
anterior. Su salida controla por PD las 12 articulaciones de las piernas; las
17 articulaciones restantes se mantienen en postura.

La política se entrenó para simulación y el ejemplo no debe enviarse a un robot
físico sin seguir el procedimiento oficial de seguridad y despliegue de
Unitree.

Los ejemplos DDS incluidos usan el dominio `1` y la interfaz local `lo0` en
macOS. Esto mantiene las comunicaciones dentro de la computadora.
