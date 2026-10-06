#!/bin/zsh
set -e
ROOT="${0:A:h}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/../.venv/bin/python" ]]; then
  PYTHON="$ROOT/../.venv/bin/python"
else
  echo "[ERROR] No se encontro Python en .venv/bin/."
  echo "Segui la instalacion indicada en MACOS_G1.md."
  exit 1
fi
cd "$ROOT/example/python"
exec "$PYTHON" -u g1_rl_stabilizer.py
