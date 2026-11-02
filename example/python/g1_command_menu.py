"""Menu de teclado para controlar el estabilizador persistente del G1."""

import select
import sys
import termios
import tty

from g1_send_command import send_command


KEY_COMMANDS = {
    "w": "adelante",
    "s": "atras",
    "a": "izquierda",
    "d": "derecha",
    "q": "girar_izquierda",
    "e": "girar_derecha",
    "c": "sentadilla",
    "h": "saludo",
    "i": "estado",
    "\x1b[A": "adelante",
    "\x1b[B": "atras",
    "\x1b[D": "izquierda",
    "\x1b[C": "derecha",
}


def show_menu():
    print(
        """
================= CONTROL DEL G1 =================
  W / flecha arriba     Paso adelante
  S / flecha abajo      Paso atras
  A / flecha izquierda  Paso hacia la izquierda
  D / flecha derecha    Paso hacia la derecha
  Q                     Encarar hacia la izquierda
  E                     Encarar hacia la derecha
  C                     Sentadilla
  H                     Saludar
  I                     Ver estado
  ESC                   Cerrar solamente este menu
==================================================
Pulsa una tecla. Espera a que termine cada movimiento.
""".strip()
    )


def read_key():
    first = sys.stdin.read(1)
    if first != "\x1b":
        return first.lower()

    sequence = first
    # Las flechas llegan como ESC, [, letra. ESC solo cierra el menu.
    for _ in range(2):
        ready, _, _ = select.select([sys.stdin], [], [], 0.03)
        if not ready:
            break
        sequence += sys.stdin.read(1)
    return sequence


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("El menu necesita ejecutarse dentro de una Terminal")

    show_menu()
    previous_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            key = read_key()
            if key == "\x03":
                print("\r\n[OK] Menu cerrado; el estabilizador sigue activo.")
                return
            if key == "\x1b":
                print("\r\n[OK] Menu cerrado; el estabilizador sigue activo.")
                return
            command = KEY_COMMANDS.get(key)
            if command is None:
                print("\r\n[AVISO] Tecla sin funcion.")
                continue
            try:
                response = send_command(command)
            except (RuntimeError, ValueError) as error:
                print(f"\r\n[ERROR] {error}")
            else:
                print(f"\r\n[{command}] {response}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, previous_settings)


if __name__ == "__main__":
    main()
