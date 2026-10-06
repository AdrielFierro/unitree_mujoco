#!/bin/zsh
set -e

ROOT="${0:A:h}"
COMMAND="${1:-actividad}"
exec "$ROOT/mandar_comando_g1_mac.command" "$COMMAND"
