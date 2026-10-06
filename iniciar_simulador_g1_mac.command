#!/bin/zsh
set -e

ROOT="${0:A:h}"
CONFIG="$ROOT/simulate_python/config.py"

if [[ -x "$ROOT/.venv/bin/mjpython" ]]; then
  MJ_PYTHON="$ROOT/.venv/bin/mjpython"
elif [[ -x "$ROOT/../.venv/bin/mjpython" ]]; then
  MJ_PYTHON="$ROOT/../.venv/bin/mjpython"
else
  echo "[ERROR] No se encontro mjpython en .venv/bin/."
  echo "Segui la instalacion indicada en MACOS_G1.md."
  exit 1
fi

if ! grep -Eq 'ROBOT[[:space:]]*=[[:space:]]*["'\'']g1["'\'']' "$CONFIG"; then
  echo "[ERROR] El simulador no esta configurado para G1."
  echo 'En simulate_python/config.py usa: ROBOT = "g1"'
  exit 1
fi

cd "$ROOT/simulate_python"
echo "Abriendo el G1 en MuJoCo..."
echo "Deja esta Terminal abierta y ejecuta el estabilizador en otra."
echo "Cerra la ventana de MuJoCo para terminar."
export UNITREE_AUTO_RELEASE_BAND=1
exec "$MJ_PYTHON" unitree_mujoco.py
