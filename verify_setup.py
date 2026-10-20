#!/usr/bin/env python3
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMES = ["chango_pensando.png", "gato_lentes.png", "gato_puno.png"]

print("=== VERIFICACIÓN DE PROYECTO ===")
print()
en_venv = sys.prefix != sys.base_prefix
print("✓ main.py:", "✔" if os.path.exists(os.path.join(BASE_DIR, "main.py")) else "✗")
print("✓ venv:", "✔" if en_venv else "✗ (no se está usando un entorno virtual)", f"[{sys.executable}]")
print("✓ Python:", sys.version.split()[0])
print()

try:
    import cv2
    print("✓ OpenCV (cv2):", cv2.__version__)
except ImportError:
    print("✗ OpenCV no disponible")

try:
    import mediapipe as mp
    print("✓ MediaPipe:", mp.__version__)
except ImportError:
    print("✗ MediaPipe no disponible")

print()
for nombre in MEMES:
    ruta = os.path.join(BASE_DIR, "memes", nombre)
    print(f"✓ memes/{nombre}:", "✔" if os.path.exists(ruta) else "✗ falta")

print()
print("Para ejecutar: run_main.bat  (o  venv\\Scripts\\python.exe main.py)")
