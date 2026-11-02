# ProtoMotions assets

Estos archivos se usan en la rama experimental `fixstabilizer` para probar un
controlador de cuerpo completo del G1 durante gestos:

- `unified_pipeline.onnx`: política de despliegue de ProtoMotions para G1.
- `unified_pipeline.yaml`: configuración asociada al modelo.
- `g1_bones_seed_mini.pt`: biblioteca reducida de movimientos; el saludo usa
  `wave_right_hand_360_002__A117_M.motion` (índice 36).

Procedencia:

- [NVLabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions), licencia
  Apache-2.0.
- Integración de referencia de
  [HansZ8/RoboJuDo](https://github.com/HansZ8/RoboJuDo).

El código local ejecuta el modelo mediante ONNX Runtime en CPU para poder
probarlo nativamente en macOS, sin Isaac Sim ni CUDA. Esta integración sigue
siendo experimental: el saludo se reproduce, pero la transición de regreso al
estabilizador de locomoción todavía puede desestabilizar al robot.
