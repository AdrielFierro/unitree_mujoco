# G1 nativo en macOS con MuJoCo

Esta adaptación ejecuta el G1 de forma nativa en macOS, sin Docker. El modo
autónomo no necesita banda elástica ni un segundo proceso controlador. El
estabilizador persistente usa ahora la política G1 de 29 articulaciones sin
reloj de marcha de
[`RoboJuDo`](https://github.com/HansZ8/RoboJuDo), basada en
[`unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab). Esta
política fue entrenada tanto para caminar como para permanecer quieta: al
recibir velocidad cero no continúa alternando los pies.

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

Terminal 3 — abrir el menu interactivo recomendado:

```bash
./menu_g1_mac.command
```

El menu acepta una tecla por movimiento: `W/S/A/D` (o las flechas) desplazan
al robot, `Q/E` cambian hacia donde mira, `C` hace una sentadilla moderada,
`H` saluda e `I` muestra el estado. `Esc` cierra solamente el menu; el
simulador y el estabilizador siguen funcionando.

Cada orden de desplazamiento mantiene la velocidad solicitada durante 0,8 s y
luego vuelve a comando cero. La política decide la pisada y recupera la postura
sin el promedio artificial de fases que usaba la versión anterior.

Como alternativa, se puede enviar una actividad directamente:

```bash
./mandar_comando_g1_mac.command actividad
```

El mismo comando puede ejecutarse nuevamente sobre el mismo simulador. También
se aceptan órdenes individuales:

```bash
./mandar_comando_g1_mac.command pasos
./mandar_comando_g1_mac.command atras
./mandar_comando_g1_mac.command izquierda
./mandar_comando_g1_mac.command derecha
./mandar_comando_g1_mac.command girar_izquierda
./mandar_comando_g1_mac.command girar_derecha
./mandar_comando_g1_mac.command sentadilla
./mandar_comando_g1_mac.command saludo
./mandar_comando_g1_mac.command estado
./mandar_comando_g1_mac.command salir
```

`salir` detiene el estabilizador. Después se puede cerrar la ventana de MuJoCo
para terminar el simulador. No hay que cerrar la Terminal completa.

### Protecciones del estabilizador

Durante el arranque, la banda sostiene temporalmente el torso mientras el
controlador valida la orientación y adopta la postura inicial. Solo se libera
después de recibir comandos continuos durante siete segundos: tres para adoptar
la postura y cuatro para estabilizarla. Un comando
aislado o un controlador que falla al arrancar no libera la banda.

La banda de arranque usa una longitud de 0,70 m para depositar al G1 cerca de
su altura normal de pie, en lugar de dejarlo suspendido. La cuenta para soltarla
solo avanza cuando la pelvis está entre 0,65 y 0,90 m y casi no tiene velocidad
vertical; así los pies ya están apoyados cuando desaparece la asistencia.

El estabilizador rechaza actividades si el G1 supera 20 grados de inclinación.
Si detecta 35 grados, datos no numéricos o una salida inválida de la política,
cancela la actividad y pasa los motores a amortiguación segura. Este estado no
intenta levantar al robot desde el suelo: hay que reiniciar MuJoCo y luego el
estabilizador.

También activa la emergencia si permanece por encima de 25 grados durante
medio segundo o si alguna articulación alcanza una velocidad extrema. Así se
permite una corrección breve tras un empujón, pero no se insiste cuando el
robot claramente no consigue recuperar el equilibrio.

El simulador tiene además un watchdog de 250 ms. Si dejan de llegar comandos,
descarta la última postura rígida y aplica solamente amortiguación. Esto evita
que un controlador cerrado o bloqueado continúe forzando las articulaciones.
El comando `estado` informa tanto la inclinación actual como la máxima.

`actividad` realiza un desplazamiento hacia adelante y un saludo. El menu utiliza una sentadilla
moderada, con transiciones suaves y las mismas protecciones del estabilizador.
La sentadilla profunda queda reservada para la demostración autónoma porque
es menos robusta ante perturbaciones.

### Modo de bajo consumo para macOS

Este modo conserva la física de MuJoCo a 500 Hz y la política neuronal a
50 Hz, pero reduce el control y `LowState` a 250 Hz, `HighState` a 50 Hz y el
visor a 25 FPS. También desactiva el publicador de joystick, oculta los paneles
laterales y utiliza una escena plana de 74 geometrías en lugar de las 179 de la
escena completa.

Terminal 1:

```bash
./iniciar_simulador_g1_bajo_consumo_mac.command
```

Terminal 2:

```bash
./iniciar_estabilizador_g1_bajo_consumo_mac.command
```

Terminal 3 utiliza exactamente los mismos comandos:

```bash
./menu_g1_mac.command
# o, para enviar una orden sin abrir el menu:
./mandar_comando_g1_mac.command saludo
./mandar_comando_g1_mac.command pasos
./mandar_comando_g1_mac.command actividad
./mandar_comando_g1_mac.command estado
```

El menu y la linea de comandos usan una unica funcion de envio local. Esto
permite conectar mas adelante reconocimiento de voz sin modificar el
estabilizador ni duplicar los movimientos; esta version no incorpora audio.

En la Mac de prueba, una medición representativa bajó el simulador de 139% a
105% de CPU y el estabilizador de 74% a 40%; la memoria del simulador bajó de
aproximadamente 1,39 GB a 1,08 GB. Los valores varían con el tamaño de la
ventana y el momento de la medición.

La política neuronal sigue ejecutándose a 50 Hz también en reposo; el modo de
bajo consumo reduce el transporte DDS y el visor, no la frecuencia aprendida
por el controlador. Para ensayos con empujones fuertes o máxima robustez,
conviene seguir usando el modo normal.

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
fuerza externa automática. La política recibe cinco cuadros de velocidad
angular, orientación gravitatoria, comando de velocidad, posiciones y
velocidades articulares y su acción anterior. No recibe fase de marcha. Su
salida controla por PD las 29 articulaciones; el saludo y la sentadilla agregan
referencias suaves sobre la salida de equilibrio.

### Procedencia de la política sin gait

El archivo `third_party/robojudo/policy_wo_gait.pt` proviene de la rama
`release` de RoboJuDo y conserva su atribución y licencia CC BY 4.0 en
`third_party/robojudo/`. El código local adapta el armado de observaciones, el
orden de motores, DDS local, las protecciones y los comandos del menú a este
simulador en macOS. El proyecto de entrenamiento Unitree RL Lab usa licencia
Apache 2.0.

La política se entrenó para simulación y el ejemplo no debe enviarse a un robot
físico sin seguir el procedimiento oficial de seguridad y despliegue de
Unitree.

Los ejemplos DDS incluidos usan el dominio `1` y la interfaz local `lo0` en
macOS. Esto mantiene las comunicaciones dentro de la computadora.
