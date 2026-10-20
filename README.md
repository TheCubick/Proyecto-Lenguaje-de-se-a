# Proyecto-Lenguaje-de-se-a

Este proyecto es de la materia Inteligencia Artificial: detecta el movimiento de las manos asociándolo a lenguaje de señas.

## 🖐️ Reacción con Memes por Gestos (MediaPipe + OpenCV)

Detecta gestos de la mano en tiempo real con la cámara y muestra un meme de reacción en una segunda ventana.

## 📋 Estructura

```
Proyecto/
├── main.py              # Programa principal
├── memes/
│   ├── chango_pensando.png  # Gesto DUDA   (mano en la cabeza)
│   ├── gato_lentes.png      # Gesto LENTES (índice arriba)
│   └── gato_puno.png        # Gesto PUNO   (puño cerrado)
├── venv/                # Único entorno virtual del proyecto (Python 3.12)
├── run_main.bat         # Script para ejecutar en Windows
├── verify_setup.py      # Comprueba dependencias e imágenes
└── README.md
```

## ⚙️ Dependencias

- **Python** 3.12 (MediaPipe 0.10.18 no es compatible con 3.13)
- **OpenCV** (cv2) 4.11.0
- **MediaPipe** 0.10.18

## 🚀 Cómo ejecutar

```cmd
run_main.bat
```
o bien
```powershell
.\venv\Scripts\python.exe main.py
```

## 🎮 Gestos

| Gesto | Cómo hacerlo | Meme |
|---|---|---|
| `NINGUNO` | Sin mano, o mano relajada | Ventana en negro |
| `DUDA` | Lleva la mano a la cabeza (frente, coronilla o sien) como rascándote | Chango pensando |
| `LENTES` | Solo el índice extendido hacia arriba, los demás dedos cerrados | Gato con lentes |
| `PUNO` | Los cuatro dedos cerrados y el pulgar recogido sobre el puño | Gato golpeando |

- **ESC**: Salir del programa

## 🔧 Configuración

En `main.py`:
- `FRAMES_ESTABLES`: frames seguidos que debe mantenerse un gesto antes de mostrarse (anti-parpadeo)
- `FRAMES_MEMORIA_CARA`: frames que se recuerda la cara cuando la mano la tapa
- `min_detection_confidence` / `min_tracking_confidence`: confianza de MediaPipe
