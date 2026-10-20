# Proyecto-Lenguaje-de-se-a

## Sistema de Reconocimiento de Lengua de Señas con Visión Artificial (MediaPipe + OpenCV)

Proyecto de la materia Inteligencia Artificial. Reconoce en tiempo real letras estáticas del alfabeto dactilológico de la Lengua de Señas Mexicana (LSM) usando los 21 puntos de la mano que detecta MediaPipe, y las convierte en texto.

Para escribir, **mantén una seña 1.5 segundos**. Una barra de progreso muestra cuánto falta y la letra se agrega al texto. Para repetir la misma letra, cambia de seña o baja la mano y vuelve a hacerla.

## Señas soportadas

| Seña | Cómo hacerla |
|---|---|
| **A** | Puño cerrado con el pulgar estirado al costado del índice |
| **B** | Los 4 dedos estirados y juntos, pulgar doblado sobre la palma |
| **E** | Dedos doblados en garra, con las puntas sobre el pulgar |
| **I** | Solo el meñique levantado |
| **L** | Índice arriba y pulgar hacia el lado, formando una L |
| **O** | Punta del índice tocando la punta del pulgar, formando un círculo |
| **U** | Índice y medio levantados y juntos |
| **V** | Índice y medio levantados y separados |
| **W** | Índice, medio y anular levantados |
| **Y** | Pulgar y meñique estirados, los 3 dedos centrales cerrados |
| **ESPACIO** | Mano abierta con los 5 dedos separados |
| **BORRAR** | Puño cerrado con el pulgar por delante de los dedos (borra la última letra) |

## Interfaz

- **Camara IA - Deteccion LSM:** video en espejo con los puntos de la mano, recuadro alrededor de la mano, la seña detectada y la barra de progreso.
- **Panel Traductor LSM:** la seña actual en grande, la guía rápida de señas y la caja con el texto acumulado.

## Teclas

| Tecla | Acción |
|---|---|
| `Espacio` | Agregar espacio |
| `Backspace` | Borrar la última letra |
| `C` | Limpiar todo el texto |
| `D` | Mostrar/ocultar el diagnóstico (estado de cada dedo, para calibrar) |
| `ESC` o la `X` de una ventana | Salir |

## Cómo ejecutar

1. Abre la carpeta del proyecto en VS Code.
2. Pulsa **F5** (o ▶️ en *Ejecutar y depurar* → "Traductor LSM (main.py)").

`.vscode/launch.json` ya usa el Python de `venv`. También puedes ejecutarlo desde la terminal:

```powershell
.\venv\Scripts\python.exe main.py
```

### Crear el entorno desde cero

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18
```

> MediaPipe 0.10.18 requiere Python 3.12 (no es compatible con 3.13).

## Calibración

Los umbrales geométricos están agrupados al inicio de `main.py`, por ejemplo `R_EXTENDIDO`, `T_PULGAR_CRUZADO`, `ANG_U_MAX` y `O_CONTACTO`. Si una seña cuesta trabajo que se reconozca, activa el diagnóstico con la tecla **D**: verás el estado de cada dedo (`EXT` estirado, `DOB` doblado, `CER` cerrado) y la posición del pulgar. Con eso puedes ajustar el umbral correspondiente.

## Estructura

```
Proyecto/
├── .vscode/       # Configuración para ejecutar con F5
├── venv/          # Entorno virtual (no se sube a Git)
├── .gitignore
├── main.py        # Programa principal
├── README.md
└── DOCUMENTACION.md  # Documentación técnica completa
```

📄 Consulta la [documentación técnica](DOCUMENTACION.md) para conocer la arquitectura, el algoritmo de clasificación, la calibración y las pruebas.
