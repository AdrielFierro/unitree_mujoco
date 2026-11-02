#!/bin/zsh
set -e

ROOT="${0:A:h}"
export UNITREE_LOW_POWER=1
exec "$ROOT/iniciar_simulador_g1_mac.command"
