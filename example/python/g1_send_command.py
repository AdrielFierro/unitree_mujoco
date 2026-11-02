"""Envia una orden de alto nivel al estabilizador local del G1."""

import argparse
import socket


SOCKET_PATH = "/tmp/unitree_g1_stabilizer.sock"
COMMANDS = (
    "actividad",
    "rutina",
    "pasos",
    "caminar",
    "adelante",
    "atras",
    "izquierda",
    "derecha",
    "girar_izquierda",
    "girar_derecha",
    "encarar_izquierda",
    "encarar_derecha",
    "sentadilla",
    "saludo",
    "estado",
    "salir",
)


def send_command(command):
    """Envia una orden reutilizable por CLI, menu o una futura entrada de voz."""
    if command not in COMMANDS:
        raise ValueError(f"Comando desconocido: {command}")

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as error:
        raise RuntimeError(
            "El estabilizador no esta activo. Ejecuta primero "
            "./iniciar_estabilizador_g1_mac.command"
        ) from error
    with client:
        client.sendall((command + "\n").encode("utf-8"))
        return client.recv(512).decode("utf-8").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "comando",
        choices=COMMANDS,
    )
    args = parser.parse_args()

    print(send_command(args.comando))


if __name__ == "__main__":
    main()
