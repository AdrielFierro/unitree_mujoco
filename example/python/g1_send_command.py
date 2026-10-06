"""Envia una orden de alto nivel al estabilizador local del G1."""

import argparse
import socket


SOCKET_PATH = "/tmp/unitree_g1_stabilizer.sock"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "comando",
        choices=("actividad", "rutina", "pasos", "caminar", "saludo", "estado", "salir"),
    )
    args = parser.parse_args()

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(SOCKET_PATH)
    except FileNotFoundError as error:
        raise RuntimeError(
            "El estabilizador no esta activo. Ejecuta primero "
            "./iniciar_estabilizador_g1_mac.command"
        ) from error
    with client:
        client.sendall((args.comando + "\n").encode("utf-8"))
        print(client.recv(512).decode("utf-8").strip())


if __name__ == "__main__":
    main()
