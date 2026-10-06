#!/bin/zsh
set -e

ROOT="${0:A:h}"
exec "$ROOT/iniciar_g1_sin_banda_mac.command" "$@"
