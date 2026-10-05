# G1 nativo en macOS con MuJoCo

Esta adaptación ejecuta el G1 de forma nativa en macOS, sin Docker. El modo
autónomo no necesita banda elástica ni un segundo proceso controlador: aplica
el control PD dentro del mismo paso físico de MuJoCo.

La demostración realiza esta secuencia:

1. Adopta la postura inicial.
2. Da dos pasos visibles hacia adelante (aproximadamente 12 cm en total).
3. Hace una sentadilla.
4. Saluda con el brazo derecho.
5. Permanece parado hasta cerrar la ventana.

## Preparación

Se verificó en Apple Silicon usando Python 3.9, MuJoCo 3.3.6 y
`unitree_sdk2_python`. Desde la carpeta que contiene este repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install mujoco==3.3.6 cyclonedds==0.10.2 pygame==2.6.1 "numpy<2.1"
python -m pip install --no-build-isolation -e ./unitree_sdk2_python
```

El SDK Python debe tener una alternativa a `timerfd` para macOS. La
implementación Linux original de ese temporizador no funciona en Darwin.

## Ejecución

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

## Alcance y seguridad

Los pasos usan una ayuda cartesiana de equilibrio horizontal y orientación que
se desactiva suavemente antes de la rutina. No aplica fuerza vertical: las
piernas soportan el peso del G1. Sigue siendo una demostración cuasiestática,
no un controlador dinámico general de locomoción. Las ganancias PD son
deliberadamente altas para sostener el robot simulado y no deben enviarse a un
robot físico.

Los ejemplos DDS incluidos usan el dominio `1` y la interfaz local `lo0` en
macOS. Esto mantiene las comunicaciones dentro de la computadora.
