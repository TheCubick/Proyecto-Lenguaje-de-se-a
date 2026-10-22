# Proyecto-Lenguaje-de-se-a

## Sistema Inteligente de Traducción de Lengua de Señas Mexicana (LSM): manos, rostro y voz

Proyecto de la materia Inteligencia Artificial. Con una cámara web reconoce en tiempo real el **alfabeto de la Lengua de Señas Mexicana**, lo convierte en texto y lo **dice en voz alta**. Usa MediaPipe (manos y rostro), OpenCV y `pyttsx3`.

- **Letras estáticas:** se escriben **sosteniendo la mano quieta 1.5 s**; una barra verde muestra el avance.
- **Letras con movimiento (J, K, Ñ, Q, X, Z):** se registran al completar el trazo.
- **Rostro:** las cejas levantadas agregan `?`, negar con la cabeza borra y asentir confirma la palabra.
- **Voz:** dice cada letra al registrarla y lee la palabra completa al confirmar o tras 4 s sin escribir.

Para repetir la misma letra, cambia de seña o baja la mano y vuelve a hacerla.

## Alfabeto (según la lámina oficial de la SEP)

| Seña | Cómo hacerla |
|---|---|
| **A** | Puño con el pulgar estirado al lado |
| **B** | 4 dedos juntos arriba, pulgar sobre la palma |
| **C** | Mano curva en forma de C |
| **D** | Índice arriba, los demás dedos cerrados |
| **E** | Dedos doblados sobre el pulgar |
| **F** | Índice tocando el pulgar y los otros 3 dedos arriba |
| **G** | "Pistola": índice apuntando de lado, pulgar arriba |
| **H** | "Pistola" con índice y medio de lado |
| **I** | Solo el meñique arriba |
| **J** | La I, y el meñique baja haciendo un gancho (*con movimiento*) |
| **K** | Índice y medio juntos o cruzados, agitando la mano de lado a lado (*con movimiento*) |
| **L** | Índice arriba y pulgar al lado |
| **M** | 3 dedos hacia abajo |
| **N** | 2 dedos hacia abajo |
| **Ñ** | La N agitada de lado a lado (*con movimiento*) |
| **O** | Círculo con el índice y el pulgar |
| **P** | Índice y medio cruzados en "X" |
| **Q** | Índice curvo colgando hacia abajo, balanceándolo (*con movimiento*) |
| **R** | Índice y medio entrelazados |
| **S** | Puño con el pulgar sobre los dedos |
| **T** | Puño con el pulgar entre los dedos |
| **U** | Índice y medio juntos arriba |
| **V** | Índice y medio en V |
| **W** | Índice, medio y anular abiertos |
| **X** | Índice curvo apuntando de lado, moviéndolo (*con movimiento*) |
| **Y** | Pulgar y meñique estirados |
| **Z** | La D dibujando una Z en el aire (*con movimiento*) |
| **ESPACIO** | Mano abierta con los 5 dedos separados |
| **BORRAR** | Mano abierta que **barre rápido hacia la izquierda** (borra la última letra) |
| **?** | Índice dibujando un signo de interrogación en el aire (respaldo de las cejas) |

> **BORRAR cambió.** Con el alfabeto completo, un puño con el pulgar adentro es la S, la T o la A, así que borrar con un puño habría borrado letras al escribirlas. Quedan también `Backspace` y negar con la cabeza.
>
> **Los trazos deben ser pequeños, del tamaño de la mano**, y sin salirse del cuadro de la cámara.
>
> **CH, LL y RR** no están incluidas.

## Gestos del rostro (MediaPipe Face Mesh)

Al iniciar, mantén la cara neutra un momento mientras el sistema calibra (`Modo: Calibrando...`). En la ventana de cámara hay un **medidor de cejas** siempre visible (`CEJAS x1.11`) con la marca del umbral.

| Gesto | Efecto |
|---|---|
| **Cejas levantadas 1.5 s** | Agrega `?` al final de la palabra |
| **Negar con la cabeza** (izquierda-derecha-izquierda) | Borra la última letra |
| **Asentir con la cabeza** | Confirma la palabra (agrega un espacio) y la lee en voz alta |

## Voz

Un hilo aparte (`pyttsx3`, con una voz en español) habla **sin congelar la cámara**. Dice el nombre de cada letra (`jota`, `eñe`, `doble ve`…) y lee la palabra completa al asentir o tras 4 s sin escribir. Si no hay voz en español instalada o falta `pyttsx3`, el programa funciona igual y el panel indica `Voz: no disponible`.

## Interfaz

- **Camara IA - Deteccion LSM:** video en espejo con la mano y los puntos del rostro, recuadro, seña detectada, estela del dedo al dibujar un trazo, medidor de cejas, `Modo:` del estado facial y barra de progreso.
- **Panel Traductor LSM:** la seña actual en grande, cómo se hace, la **guía del alfabeto** (en naranja las que llevan movimiento), el texto acumulado y el estado de la voz. Los espacios se ven como espacios.

## Teclas

| Tecla | Acción |
|---|---|
| `Espacio` | Agregar espacio |
| `Backspace` | Borrar la última letra |
| `C` | Limpiar todo el texto |
| `D` | Mostrar u ocultar el diagnóstico (estado de cada dedo y métricas, para calibrar) |
| `R` | Recalibrar el rostro |
| `ESC` o la `X` de una ventana | Salir |

## Cómo ejecutar

1. Abre la carpeta del proyecto en VS Code.
2. Pulsa **F5** (o el botón en *Ejecutar y depurar* → "Traductor LSM (main.py)").

`.vscode/launch.json` ya usa el Python de `venv`. También puedes ejecutarlo desde la terminal:

```powershell
.\venv\Scripts\python.exe main.py
```

### Crear el entorno desde cero

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18 numpy pyttsx3
```

> MediaPipe 0.10.18 requiere Python 3.12 (no es compatible con 3.13). La voz usa las voces SAPI5 de Windows (por ejemplo *Microsoft Sabina*, español de México).

## Qué tan bien funciona

Se calibró y se probó con **grabaciones reales** de la cámara (solo coordenadas de landmarks, sin imágenes), de **una sola persona**:

| Parte | Resultado |
|---|---|
| Letras estáticas | 16 de 21 letras con precisión por fotograma ≥ 96 %; **C, P y U** entre 78 y 91 %; **R (68 %) y F (39 %)** son las menos fiables |
| Mano relajada | no escribe ninguna letra |
| Pregunta con las cejas | **2 de 2** y cero falsos eventos (la versión anterior detectaba 0 de 2) |
| Trazos | "?" 4 de 4 · BORRAR 3 de 3 · Z 3 de 4 · J 2 de 3 · sin falsos positivos en movimiento libre |
| Invarianza | 100 % igual con escala de 0.4× a 2.5×, desplazada o con la otra mano |
| Voz | no afecta a los fotogramas (mediana de 47 ms con y sin voz) |
| Velocidad | 27-29 FPS sin GPU |

**Limitaciones:** la **K** puede aparecer por error si agitas la mano con dos dedos arriba; **Q y X** se parecen mucho y se separan solo por la dirección del índice; **S y T** también. Detalles, cifras y la metodología completa en la [documentación técnica](DOCUMENTACION.md).

## Calibración

Los umbrales están agrupados al inicio de `main.py`. Si una seña cuesta trabajo, activa el diagnóstico con la tecla **D**: verás el estado de cada dedo (`EXT` estirado, `DOB` doblado, `CER` cerrado), la posición del pulgar y las métricas de la cabeza, y con eso puedes ajustar el umbral correspondiente.

## Estructura

```
Proyecto/
├── .vscode/          # Configuración para ejecutar con F5
├── venv/             # Entorno virtual (no se sube a Git)
├── .gitignore
├── main.py           # Programa principal
├── README.md
└── DOCUMENTACION.md  # Documentación técnica completa
```
