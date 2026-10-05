#!/bin/zsh
set -e

ROOT="${0:A:h}"

if [[ -x "$ROOT/.venv/bin/mjpython" ]]; then
  MJ_PYTHON="$ROOT/.venv/bin/mjpython"
elif [[ -x "$ROOT/../.venv/bin/mjpython" ]]; then
  MJ_PYTHON="$ROOT/../.venv/bin/mjpython"
else
  echo "[ERROR] No se encontro mjpython en .venv/bin/."
  echo "Segui la instalacion indicada en MACOS_G1.md."
  read "?Presiona Enter para cerrar..."
  exit 1
fi

cd "$ROOT/example/python"
echo "Abriendo el G1 autonomo sin banda."
echo "Dara dos pasos cortos, hara la rutina y permanecera parado."
exec "$MJ_PYTHON" g1_autonomous_no_band.py "$@"
