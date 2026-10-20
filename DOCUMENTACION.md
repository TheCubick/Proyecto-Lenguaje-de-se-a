# 🤟 Sistema Inteligente de Traducción de Lengua de Señas con Visión Artificial y Componentes Multimodales

> **Documentación técnica oficial**
> **Materia:** Inteligencia Artificial
> **Stack:** Python 3.12 · MediaPipe 0.10.18 · OpenCV 4.11 · NumPy
> **Repositorio:** https://github.com/TheCubick/Proyecto-Lenguaje-de-se-a

### 🏷️ Leyenda de estado

En este documento cada componente lleva una etiqueta que indica su grado de avance real:

| Etiqueta | Significado |
|---|---|
| ✅ **Implementado** | Funciona en el código actual (`main.py`) y fue probado |
| 🧪 **Diseñado** | Arquitectura y algoritmo definidos; pendiente de implementar |
| 🔭 **Planeado** | Forma parte del roadmap a mediano o largo plazo |

---

## 📑 Índice

1. [Descripción general](#1--descripción-general)
2. [Arquitectura del sistema (enfoque multimodal)](#2-️-arquitectura-del-sistema-enfoque-multimodal)
   - 2.1 [Módulo de visión de manos](#21--módulo-de-visión-de-manos-mediapipe-hands--implementado)
   - 2.2 [Módulo de análisis facial](#22--módulo-de-análisis-facial-componentes-no-manuales--diseñado)
3. [Flujo de trabajo y experiencia de usuario](#3--flujo-de-trabajo-y-experiencia-de-usuario-ux--ui)
4. [Fase de integración de hardware](#4--fase-de-integración-de-hardware--diseñado)
5. [Roadmap](#5-️-roadmap-y-futuro-del-proyecto)
6. [Guía de instalación y ejecución](#6-️-guía-de-instalación-y-ejecución)
7. [Calibración](#7--calibración)
8. [Validación y pruebas](#8--validación-y-pruebas)
9. [Limitaciones conocidas](#9-️-limitaciones-conocidas)
10. [Referencia del código](#10--referencia-del-código)

---

## 1. 📌 Descripción general

### 1.1 Propósito

El sistema traduce **en tiempo real** la **dactilología** (deletreo con las manos) y **señas estáticas** de la **Lengua de Señas Mexicana (LSM)** a texto escrito, usando solamente una cámara web convencional.

La LSM no se comunica solo con las manos. También tiene **componentes no manuales**: las expresiones faciales y los movimientos de cabeza tienen función gramatical. Por eso el sistema se diseñó con una **arquitectura multimodal** que combina dos fuentes de información:

| Modalidad | Qué aporta | Estado |
|---|---|---|
| ✋ **Manual** (manos) | Qué letra o seña se está formando | ✅ Implementado |
| 🙂 **No manual** (rostro y cabeza) | Si la frase es pregunta, negación o afirmación | 🧪 Diseñado |

### 1.2 Problema que atiende

La comunidad sorda enfrenta barreras de comunicación cotidianas con personas oyentes que no conocen la lengua de señas. Las soluciones comerciales suelen requerir guantes con sensores o hardware costoso. Este proyecto explora una alternativa **de bajo costo, sin contacto y basada en visión artificial**, que funciona en cualquier computadora con cámara.

### 1.3 Enfoque técnico

El reconocimiento combina dos técnicas:

1. **Estimación de pose de la mano mediante aprendizaje profundo** (modelos preentrenados de MediaPipe), que localizan 21 puntos clave por mano.
2. **Un clasificador geométrico basado en reglas**, diseñado por el equipo, que interpreta esos puntos con distancias euclidianas, ángulos articulares y proyecciones vectoriales normalizadas.

| Ventaja del enfoque | Explicación |
|---|---|
| 🧠 **Explicable** | Cada decisión se justifica con una regla geométrica concreta |
| 📦 **Sin dataset** | No requiere recolectar ni etiquetar miles de imágenes |
| ⚡ **Tiempo real** | Funciona en una laptop sin GPU |
| 🔁 **Invariante** | No depende de la distancia a la cámara ni de si se usa la mano izquierda o la derecha |

---

## 2. 🏗️ Arquitectura del sistema (enfoque multimodal)

### 2.0 Vista general

```
                         ┌────────────────────────┐
                         │      📷 Cámara web     │
                         └───────────┬────────────┘
                                     │ frame BGR (≈30 FPS)
                         ┌───────────▼────────────┐
                         │ Preprocesamiento       │
                         │ espejo · BGR→RGB       │
                         └─────┬────────────┬─────┘
                               │            │
          ┌────────────────────▼───┐   ┌────▼──────────────────────┐
          │ ✋ MÓDULO DE MANOS  ✅ │   │ 🙂 MÓDULO FACIAL       🧪 │
          │ MediaPipe Hands        │   │ MediaPipe Face Mesh       │
          │ 21 landmarks           │   │ 468 landmarks             │
          ├────────────────────────┤   ├───────────────────────────┤
          │ Normalización espacial │   │ Línea base neutral        │
          │ Rasgos geométricos     │   │ Cejas · giro · asentir    │
          │ Clasificador de reglas │   │ Detector de oscilaciones  │
          └───────────┬────────────┘   └─────────────┬─────────────┘
                      │ letra / seña                  │ marcador gramatical
                      │                               │ (¿? · NO · SÍ)
                      └───────────────┬───────────────┘
                          ┌───────────▼────────────┐
                          │ 🧩 FUSIÓN MULTIMODAL   │
                          │ Estabilizador (4 fr.)  │
                          │ Hold-to-Type (1.5 s)   │
                          └───────────┬────────────┘
              ┌───────────────────────┼────────────────────────┐
     ┌────────▼────────┐    ┌─────────▼─────────┐    ┌─────────▼─────────┐
     │ 🖥️ Ventana      │    │ 📋 Panel          │    │ 🔌 Salida serial  │
     │ Cámara      ✅  │    │ Traductor     ✅  │    │ ESP32/Arduino 🧪  │
     └─────────────────┘    └───────────────────┘    └───────────────────┘
```

---

### 2.1 ✋ Módulo de visión de manos (MediaPipe Hands) ✅ Implementado

#### 2.1.1 Detección y extracción de puntos clave

MediaPipe Hands trabaja en dos etapas:
1. Un **detector de palma** (SSD) localiza la mano en la imagen.
2. Un **modelo de regresión de landmarks** estima 21 puntos sobre la región detectada.

En modo video, la segunda etapa reutiliza la posición del frame anterior, lo que reduce el costo de cómputo.

| Parámetro | Valor | Justificación |
|---|---|---|
| `static_image_mode` | `False` | Seguimiento temporal: más rápido y estable |
| `max_num_hands` | `1` | Las señas del alcance actual son de una mano |
| `min_detection_confidence` | `0.7` | Reduce detecciones falsas |
| `min_tracking_confidence` | `0.6` | Evita perder la mano en movimientos rápidos |

**Topología de los 21 landmarks:**

```
          8    12   16   20      ← TIP  (puntas)
          7    11   15   19      ← DIP
          6    10   14   18      ← PIP  (articulación media)
    4     5     9   13   17      ← MCP  (nudillos)
     3
      2                          Pulgar: 1 CMC · 2 MCP · 3 IP · 4 TIP
       1
            0                    ← WRIST (muñeca)
```

#### 2.1.2 Normalización espacial

| Paso | Operación | Objetivo |
|---|---|---|
| 1️⃣ Desnormalización | `(x·ancho, y·alto)` | MediaPipe entrega coordenadas en [0, 1]. Sin este paso, en una imagen 4:3 o 16:9 una misma distancia mediría distinto en horizontal y en vertical |
| 2️⃣ Escala de referencia | `palma = ‖P₀ − P₉‖` | Todas las distancias se expresan en "palmas", así que la medida no depende de qué tan lejos esté la mano |
| 3️⃣ Marco de la mano | Eje `P₅ → P₁₇` | Las posiciones del pulgar se miden respecto a la propia mano, no respecto a la imagen. Esto lo hace funcionar igual con la mano izquierda y con la derecha |

#### 2.1.3 Rasgos geométricos

**a) Estado de cada dedo (índice a meñique).** Para cada dedo se calculan dos valores:

```
            ‖P_tip − P_wrist‖
r_dedo  =  ───────────────────            θ_PIP = ∠(P_mcp, P_pip, P_tip)
            ‖P_mcp − P_wrist‖
```

| Estado | Condición | Interpretación |
|---|---|---|
| `EXT` estirado | θ_PIP > 150° **y** r > 1.45 | Dedo recto y alejado de la palma |
| `CER` cerrado | r < 1.05 | La punta regresa hacia la palma (puño) |
| `DOB` doblado | otro caso | Dedo curvo (garra o la letra O) |

**b) Posición del pulgar (proyección escalar).** La punta del pulgar se proyecta sobre el eje de los nudillos:

```
      (P₄ − P₅) · (P₁₇ − P₅)
t = ─────────────────────────        t < 0     → fuera, del lado del índice
          ‖P₁₇ − P₅‖²                t > 0.20  → cruzado sobre la palma
```

Con ese valor se definen dos rasgos:
- `pulgar_fuera`: el pulgar está recto (θ_IP > 140°), `‖P₄ − P₅‖ / palma > 0.6` y `t < 0`.
- `pulgar_cruzado`: `t > 0.20`.

**c) Apertura entre dedos (ángulo vectorial).** Para saber si dos dedos están juntos o separados se mide el ángulo entre sus vectores nudillo → punta:

```
θ = arccos( (v₁ · v₂) / (‖v₁‖ ‖v₂‖) )
```

Se usa el ángulo y no la distancia entre las puntas. El índice es más corto que el medio, y esa diferencia de largo infla la distancia entre puntas aunque los dedos estén pegados. Este error se detectó durante las pruebas: hacía que una U no se reconociera.

#### 2.1.4 Clasificador de señas

El clasificador aplica reglas en un **orden de prioridad fijo**. Si ninguna se cumple, devuelve `None`: el sistema prefiere no responder antes que equivocarse.

| Seña | Dedos estirados (I·M·A·Me) | Pulgar | Regla discriminante |
|---|---|---|---|
| **ESPACIO** ␣ | 4 | fuera | ∠(índice, meñique) ≥ 35° |
| **B** | 4 | no fuera | ∠(índice, meñique) < 35° |
| **W** | I · M · A | — | Meñique no estirado |
| **U** | I · M | — | ∠(índice, medio) < 10° |
| **V** | I · M | — | ∠(índice, medio) > 18° |
| **I** | Me | no fuera | — |
| **Y** | Me | fuera | — |
| **L** | I | fuera | 50° ≤ ∠(pulgar, índice) ≤ 130° |
| **O** | ninguno | t < 0.30 | ‖P₄ − P₈‖ < 0.3 palmas; índice no cerrado |
| **A** | ninguno (todos `CER`) | no cruzado | Puño con el pulgar al costado |
| **BORRAR** ⌫ | ninguno (todos `CER`) | cruzado | Puño con el pulgar al frente |
| **E** | ninguno (≥ 2 `DOB`) | cruzado | Garra sobre el pulgar |

**Árbol de decisión:**

```
                        ¿Cuántos dedos estirados?
     ┌──────────┬──────────┬────────┼──────────┬───────────┐
     4        I·M·A       I·M     solo Me    solo I        0
     │          │          │        │          │           │
 ¿pulgar      [W]     ∠ I-M ?   ¿pulgar    ¿pulgar     ¿pulgar toca
  fuera y              │         fuera?     fuera a     índice?
  abierta?        <10° │ >18°    │    │     ~90°?        │     │
  │      │         [U] │ [V]    sí    no     │          sí    no
 sí    no,             │        [Y]  [I]    [L]        [O]    │
 [ESP] juntos     10–18°: None                          ¿todos cerrados?
       [B]        (zona muerta)                  ┌──────────┼──────────┐
                                           pulgar al   pulgar      garra +
                                             lado     cruzado    pulgar cruzado
                                              [A]    [BORRAR]        [E]
```

> 💡 Las **zonas muertas** (por ejemplo, un ángulo entre 10° y 18° para U/V) son intencionales. Introducen histéresis y evitan que la detección oscile entre dos letras parecidas.

---

### 2.2 🙂 Módulo de análisis facial (componentes no manuales) 🧪 Diseñado

> ⚠️ **Estado:** este módulo está **diseñado, pero no implementado** en la versión actual de `main.py`. Esta sección describe su fundamento teórico y la especificación técnica para implementarlo.

#### 2.2.1 Fundamento lingüístico

En la LSM, como en otras lenguas de señas, los **componentes no manuales (CNM)** cumplen funciones gramaticales. La misma secuencia de señas manuales cambia de significado según el rostro:

| Componente no manual | Función gramatical | Ejemplo |
|---|---|---|
| 🤨 Cejas levantadas | Pregunta cerrada (sí/no) | "¿Tú vienes?" |
| 😠 Cejas fruncidas | Pregunta abierta (qué, dónde, quién…) | "¿Dónde vives?" |
| ↔️ Giro lateral repetido de la cabeza | Negación | "No vengo" |
| ↕️ Asentir con la cabeza | Afirmación | "Sí, vengo" |

Un traductor que solo mira las manos pierde esta información. Por eso el diseño contempla un segundo canal de percepción.

#### 2.2.2 Extracción: MediaPipe Face Mesh

Face Mesh estima **468 landmarks faciales en 3D** (478 si se activa `refine_landmarks` para el iris). Para los CNM solo se necesita un subconjunto:

| Región | Landmarks (índices de Face Mesh) | Uso |
|---|---|---|
| Ceja izquierda / derecha | 105 · 334 (arco superior); 55 · 285 (extremos internos) | Elevación y fruncido |
| Párpado superior | 159 · 386 | Referencia vertical de cada ojo |
| Comisuras externas de los ojos | 33 · 263 | Escala: distancia interocular |
| Punta de la nariz | 1 | Centro para el giro y la inclinación de la cabeza |
| Mejillas (contorno) | 234 · 454 | Referencia horizontal para el giro |
| Frente / mentón | 10 · 152 | Referencia vertical para el asentimiento |

#### 2.2.3 Métricas propuestas

Todas las métricas se normalizan con la **distancia interocular** `D = ‖P₃₃ − P₂₆₃‖`, por la misma razón de invariancia a la escala que en el módulo de manos.

| Métrica | Fórmula | Evento detectado |
|---|---|---|
| **Elevación de cejas** | `E = ½(‖P₁₀₅ − P₁₅₉‖ + ‖P₃₃₄ − P₃₈₆‖) / D` | `E > 1.15 · E_base` → cejas levantadas |
| **Fruncido** | `F = ‖P₅₅ − P₂₈₅‖ / D` | `F < 0.90 · F_base` → cejas fruncidas |
| **Giro (yaw)** | `Y = (‖P₁ − P₂₃₄‖ − ‖P₁ − P₄₅₄‖) / ‖P₂₃₄ − P₄₅₄‖` | Valor con signo: izquierda (−) o derecha (+) |
| **Inclinación (pitch)** | `Pt = (‖P₁ − P₁₀‖ − ‖P₁ − P₁₅₂‖) / ‖P₁₀ − P₁₅₂‖` | Arriba (−) o abajo (+) |

**Calibración de la línea base (`_base`).** Cada rostro es distinto. Por eso, durante los primeros 2 segundos con rostro neutro se calcula la **mediana** de `E` y `F`, y las decisiones se toman en relación con esa base personal y no con valores absolutos.

#### 2.2.4 Detección de movimientos de cabeza (análisis temporal)

Un giro de negación es un **patrón temporal**, no una postura. Se detecta con una ventana deslizante:

```
Ventana de ~1 s (30 valores de Y)
  Y │    ╱╲        ╱╲
    │   ╱  ╲      ╱  ╲          NEGACIÓN si:
  0 ┼──╱────╲────╱────╲──        • ≥ 2 cruces por cero (cambios de signo)
    │        ╲  ╱                • amplitud |Y| > 0.15 en ambos lados
    │         ╲╱                 • todo dentro de 1.2 s
    └──────────────────▶ t
```

La **afirmación** se detecta igual sobre la señal `Pt` (oscilación vertical).

#### 2.2.5 Integración con el módulo de manos (fusión)

| Evento facial | Efecto en el texto |
|---|---|
| 🤨 Cejas levantadas sostenidas al registrar la última palabra | Se agrega `?` al final de la frase |
| 😠 Cejas fruncidas | Se marca como pregunta abierta: `¿…?` |
| ↔️ Negación | Se antepone el marcador `[NO]` a la frase actual |
| ↕️ Afirmación | Se agrega el marcador `[SÍ]` |

**Consideraciones de rendimiento:**
- Ejecutar Hands y Face Mesh en cada frame reduce los FPS en equipos sin GPU. Por eso se propone procesar el rostro **en frames alternos**: los CNM cambian más lento que la configuración de la mano.
- La fusión pasaría por el mismo `Estabilizador`, para que los marcadores gramaticales también sean robustos al ruido.

---

## 3. 🔄 Flujo de trabajo y experiencia de usuario (UX / UI)

### 3.1 Ciclo de procesamiento por frame ✅

```
 ┌─────────┐   ┌──────────┐   ┌───────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐
 │ Captura │──▶│ Detección│──▶│ Rasgos    │──▶│ Clasificador│──▶│Estabilizador│──▶│ Escritor │
 │ + espejo│   │ MediaPipe│   │ geométric.│   │ de reglas   │   │  4 frames   │   │  1.5 s   │
 └─────────┘   └──────────┘   └───────────┘   └─────────────┘   └─────────────┘   └────┬─────┘
                                                                                       │
                                                       ┌───────────────────────────────┤
                                                ┌──────▼──────┐                 ┌──────▼──────┐
                                                │ Ventana     │                 │ Panel       │
                                                │ Cámara      │                 │ Traductor   │
                                                └─────────────┘                 └─────────────┘
```

### 3.2 🛡️ Estabilización (debounce por frames consecutivos) ✅

La clasificación de un frame aislado puede fallar por desenfoque de movimiento o por oclusión parcial. La clase `Estabilizador` aplica un **filtro de consenso temporal**: una seña nueva solo se acepta si aparece en **4 frames consecutivos**.

| Sin estabilizador | Con estabilizador |
|---|---|
| `A A E A A A B A` → la etiqueta parpadea | `A A A A A A A A` → etiqueta estable |
| Letras falsas durante las transiciones | Las transiciones se ignoran |

A unos 30 FPS, 4 frames equivalen a unos **133 ms**: el retraso es imperceptible para el usuario.

### 3.3 ⌨️ Constructor de palabras *Hold-to-Type* ✅

La clase `Escritor` implementa una **máquina de estados finitos**:

```
                 aparece seña                    progreso ≥ 100 %
  ┌──────────┐ ───────────────▶ ┌──────────┐ ───────────────────▶ ┌────────────┐
  │ SIN SEÑA │                  │ CARGANDO │                      │ REGISTRADA │
  └──────────┘ ◀─────────────── └──────────┘ ◀─────────────────── └────────────┘
       ▲          baja la mano     │    ▲         cambia la seña         │
       │                           └────┘                                │
       │                     otra seña: progreso = 0                     │
       └─────────────────────────────────────────────────────────────────┘
                                  baja la mano
```

| Característica | Detalle |
|---|---|
| ⏱️ Tiempo de retención | 1.5 s (`SEGUNDOS_PARA_ESCRIBIR`), medido con `time.monotonic()`, así que no depende de los FPS |
| 📊 Retroalimentación | La barra de progreso pasa de 0 % a 100 %; al registrar, el recuadro de la mano se pone verde y la letra destella |
| 🔁 Anti-repetición | Una seña registrada no se repite hasta cambiar de seña o bajar la mano. Para escribir letras dobles ("LL") se baja la mano un momento entre ambas |
| ␣ ⌫ Señas de control | **ESPACIO** agrega un espacio y **BORRAR** elimina el último carácter |

### 3.4 🖥️ Interfaz dual generada por software ✅

Al arrancar, las dos ventanas se colocan **lado a lado** con `cv2.moveWindow`, sin encimarse.

**Ventana 1 — "Camara IA - Deteccion LSM"**
- 📹 Video en espejo, para que el usuario se vea como en un espejo.
- 🦴 Esqueleto de la mano: 21 landmarks y sus conexiones.
- 🔲 Recuadro con las esquinas resaltadas, que cambia de color según el estado:
  - ⚪ gris: no se reconoce ninguna seña;
  - 🔵 azul: se está cargando una seña;
  - 🟢 verde: la seña ya se registró.
- 🏷️ Etiqueta con la seña detectada sobre el recuadro.
- 📊 Barra de progreso inferior con porcentaje.
- 🔧 **Modo diagnóstico** (tecla `D`): muestra en vivo el estado y el valor `r` de cada dedo, y los valores `t` y `d` del pulgar.

**Ventana 2 — "Panel Traductor LSM"** (lienzo generado con NumPy, sin imágenes externas)

```
┌────────────────────────────────────────────────────────┐
│  TRADUCTOR LSM                     MediaPipe + OpenCV  │
├───────────────────────┬────────────────────────────────┤
│  SEÑA ACTUAL          │  GUÍA RÁPIDA (acordeón)        │
│                       │  A    Puño + pulgar al costado │
│          V            │  B    4 dedos juntos, pulgar…  │
│                       │ ▶V    Índice + medio en V ◀    │
│                       │  …    (seña activa resaltada)  │
│  [██████████░░░░░]    │                                │
├───────────────────────┴────────────────────────────────┤
│  TEXTO                                                 │
│  HOLA_LSM|                                             │
│  Espacio · Backspace · C limpiar · D diagnóstico · ESC │
└────────────────────────────────────────────────────────┘
```

**Controles de teclado (respaldo):**

| Tecla | Acción |
|---|---|
| `Espacio` | Agregar un espacio |
| `Backspace` | Borrar la última letra |
| `C` | Limpiar todo el texto |
| `D` | Activar o desactivar el modo diagnóstico |
| `ESC` o la `X` de cualquier ventana | Salir limpiamente (libera la cámara y los modelos) |

---

## 4. 🔌 Fase de integración de hardware 🧪 Diseñado

> ⚠️ **Estado:** fase de **diseño**. Esta sección especifica la arquitectura, el protocolo y las conexiones propuestas. El hardware aún no está construido.

### 4.1 Objetivo

Llevar el texto traducido **fuera de la pantalla de la PC**, hacia un dispositivo físico con una pantalla propia montada en protoboard. Es el primer paso hacia un **dispositivo portátil y autónomo**.

### 4.2 Arquitectura propuesta

```
┌─────────────────────────────┐                 ┌───────────────────────────────────┐
│  💻 PC (visión artificial)  │                 │  🔧 Protoboard                     │
│                             │   USB / UART    │  ┌─────────────┐   I²C   ┌───────┐ │
│  main.py                    │ ──────────────▶ │  │ ESP32 /     │ ──────▶ │ OLED  │ │
│   └─ Escritor ──▶ pyserial  │   115200 baud   │  │ Arduino Uno │         │128×64 │ │
│                             │                 │  └─────────────┘         └───────┘ │
│                             │   ── o bien ──  │        ▲                           │
│   └─ Escritor ──▶ WebSocket │ ~~~ Wi-Fi ~~~~▶ │        │ (solo ESP32)              │
└─────────────────────────────┘                 └───────────────────────────────────┘
```

### 4.3 Componentes

| Componente | Opción recomendada | Alternativa | Función |
|---|---|---|---|
| Microcontrolador | **ESP32 DevKit** (Wi-Fi + Bluetooth, 240 MHz) | Arduino Uno / Nano (solo serial) | Recibe el texto y controla la pantalla |
| Pantalla | **OLED SSD1306 de 0.96" (128×64, I²C)** | LCD 16×2 con adaptador I²C (PCF8574) | Muestra la letra y la palabra |
| Conexión | Cable USB (alimentación + datos) | Wi-Fi (solo ESP32) | Canal PC → dispositivo |
| Prototipado | Protoboard + cables Dupont | — | Montaje sin soldadura |
| Autonomía (fase posterior) | Batería LiPo / 18650 + módulo de carga TP4056 | Power bank USB | Portabilidad |

**Conexiones (ESP32 ↔ OLED SSD1306):**

| OLED | ESP32 | Arduino Uno |
|---|---|---|
| VCC | 3V3 | 5V |
| GND | GND | GND |
| SDA | GPIO 21 | A4 |
| SCL | GPIO 22 | A5 |

### 4.4 Protocolo de comunicación

Se propone un **protocolo de texto línea por línea**, sencillo de depurar con cualquier monitor serial:

| Mensaje | Significado | Ejemplo |
|---|---|---|
| `L:<seña>\n` | Seña detectada en este momento (vista previa) | `L:V` |
| `T:<texto>\n` | Texto acumulado completo | `T:HOLA LSM` |
| `C\n` | Limpiar la pantalla | `C` |

**Reglas de diseño del protocolo:**
- Se envía **solo cuando algo cambia**, no en cada frame. Así se evita saturar el canal: a 30 FPS serían 30 mensajes por segundo innecesarios.
- Se envía el **texto completo** (`T:`) en lugar de diferencias. Si se pierde un mensaje, el siguiente corrige el estado: el protocolo se recupera solo.
- Se usa `\n` como delimitador. Es compatible con `Serial.readStringUntil('\n')` en Arduino.

**Envío desde la PC (propuesta con `pyserial`):**

```python
import serial  # pip install pyserial

puerto = serial.Serial("COM3", 115200, timeout=0)
ultimo_texto = None

def enviar_si_cambia(texto):
    global ultimo_texto
    if texto != ultimo_texto:
        puerto.write(f"T:{texto}\n".encode("ascii", "replace"))
        ultimo_texto = texto
```

**Recepción en el ESP32 (propuesta en Arduino/C++):**

```cpp
#include <Wire.h>
#include <Adafruit_SSD1306.h>

Adafruit_SSD1306 oled(128, 64, &Wire, -1);

void setup() {
  Serial.begin(115200);
  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  oled.clearDisplay(); oled.display();
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    if (msg.startsWith("T:")) {
      oled.clearDisplay();
      oled.setTextSize(2); oled.setTextColor(SSD1306_WHITE);
      oled.setCursor(0, 0);
      oled.print(msg.substring(2));
      oled.display();
    } else if (msg == "C") {
      oled.clearDisplay(); oled.display();
    }
  }
}
```

### 4.5 Variante inalámbrica (Wi-Fi, solo ESP32)

| Aspecto | Propuesta |
|---|---|
| Rol del ESP32 | Servidor WebSocket en la red local (o *access point* propio) |
| Rol de la PC | Cliente que envía los mismos mensajes `T:` y `L:` |
| Ventaja | El dispositivo no necesita cable de datos: solo alimentación |
| Riesgo | Latencia variable de la red; conviene usar un *timeout* y reintentos |

### 4.6 Camino hacia la autonomía total

```
 Etapa 1 (actual)         Etapa 2 (próxima)             Etapa 3 (portátil)
 ┌──────────────┐         ┌──────────────────────┐      ┌──────────────────────────┐
 │ PC: visión + │   ──▶   │ PC: visión           │ ──▶  │ SBC (Raspberry Pi 5)     │
 │ pantalla     │         │ ESP32: pantalla OLED │      │ o teléfono: visión       │
 └──────────────┘         └──────────────────────┘      │ ESP32 + batería: pantalla│
                                                        └──────────────────────────┘
```

> 📝 **Nota técnica:** un ESP32 **no tiene capacidad** para ejecutar MediaPipe Hands. La visión debe correr en un equipo con más cómputo (PC, Raspberry Pi o teléfono), y el microcontrolador actúa como **periférico de salida**. Un ESP32-S3 con cámara podría ejecutar modelos muy reducidos con TensorFlow Lite Micro, pero eso requeriría entrenar un clasificador propio (ver el roadmap).

---

## 5. 🗺️ Roadmap y futuro del proyecto

| Fase | Entregable | Tecnologías | Estado |
|---|---|---|---|
| **1. Prototipo manual** | 10 letras + ESPACIO/BORRAR, hold-to-type, interfaz dual | MediaPipe Hands, OpenCV | ✅ Completado |
| **2. Componentes no manuales** | Preguntas, negación y afirmación por rostro y cabeza | MediaPipe Face Mesh | 🧪 Diseñado |
| **3. Hardware** | Pantalla OLED en protoboard por USB o Wi-Fi | ESP32, pyserial, WebSocket | 🧪 Diseñado |
| **4. Alfabeto completo** | Resto de letras estáticas (C, D, F, G, H, M, N, P, R, S, T) | Reglas geométricas + coordenada z | 🔭 Planeado |
| **5. Machine Learning** | Clasificador entrenado con landmarks propios | scikit-learn (k-NN, Random Forest), red densa | 🔭 Planeado |
| **6. Señas dinámicas** | Letras con movimiento (J, K, Ñ, Q, X, Z) y señas de acción | Secuencias de landmarks + **LSTM / GRU** | 🔭 Planeado |
| **7. Frases continuas** | Traducción de frases completas | Modelos secuenciales / Transformers | 🔭 Planeado |
| **8. Impacto social** | App móvil o dispositivo cerrado; texto a voz | MediaPipe para Android/iOS, TTS | 🔭 Planeado |

### 5.1 🧠 Hacia el Machine Learning

El clasificador de reglas es explicable, pero escala mal: cada letra nueva exige diseñar reglas a mano. La evolución natural es **aprender las fronteras de decisión a partir de datos**:

```
 Captura de datos        Vector de rasgos                 Modelo                Salida
 ┌──────────────┐       ┌─────────────────────┐       ┌───────────────┐      ┌─────────┐
 │ main.py en   │  ──▶  │ 21 landmarks × (x,y,z)│ ──▶ │ k-NN / RF     │ ──▶  │ letra   │
 │ modo captura │       │ normalizados = 63 val.│      │ (estáticas)   │      │         │
 └──────────────┘       └─────────────────────┘       ├───────────────┤      ├─────────┤
                                                       │ LSTM sobre    │ ──▶  │ seña    │
                        secuencia de T frames ──▶      │ T × 63        │      │ dinámica│
                                                       └───────────────┘      └─────────┘
```

**Por qué LSTM para las señas dinámicas:** una seña en movimiento es una **serie temporal** de poses. Las redes LSTM (*Long Short-Term Memory*) mantienen un estado interno que captura la dependencia entre frames, lo que permite distinguir trayectorias como el trazo de la **J** o la **Z**. Un clasificador de un solo frame no puede hacerlo.

**Ventaja del enfoque basado en landmarks:** el modelo no aprende píxeles, sino **63 números normalizados por frame**. Por eso:
- requiere órdenes de magnitud menos datos que una CNN sobre imágenes;
- entrena en minutos en una laptop;
- se puede desplegar en dispositivos modestos.

### 5.2 🌍 Impacto social

- 📱 **Aplicación móvil:** MediaPipe tiene soporte nativo para Android e iOS, así que el mismo pipeline puede llevarse al teléfono.
- 🔊 **Texto a voz:** cerraría el ciclo de comunicación persona sorda → persona oyente.
- 🏫 **Herramienta educativa:** practicar el alfabeto dactilológico con retroalimentación inmediata.
- 🤝 **Validación con la comunidad:** pruebas con personas usuarias de LSM para corregir la forma de las señas y priorizar el vocabulario.

---

## 6. ⚙️ Guía de instalación y ejecución

### 6.1 Requisitos

| Requisito | Versión / detalle |
|---|---|
| 🪟 Sistema operativo | Windows 10/11 (también funciona en Linux y macOS con rutas equivalentes) |
| 🐍 Python | **3.12** (MediaPipe 0.10.18 **no** es compatible con 3.13) |
| 📷 Hardware | Cámara web integrada o USB |
| 🧰 Editor | VS Code con la extensión **Python** (incluye el depurador `debugpy`) |

### 6.2 Dependencias

| Librería | Versión | Uso |
|---|---|---|
| `mediapipe` | 0.10.18 | Detección de la mano (y, en el futuro, del rostro) |
| `opencv-python` | 4.11 | Captura de video, dibujo y ventanas |
| `numpy` | 1.26 (se instala con MediaPipe) | Lienzo del panel y operaciones con matrices |

### 6.3 Crear el entorno virtual (`venv`)

```powershell
# 1. Crear el entorno con Python 3.12
py -3.12 -m venv venv

# 2. Instalar dependencias
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18 numpy
```

> 💡 La carpeta `venv/` está en `.gitignore`: cada integrante crea la suya localmente y no se sube al repositorio.

### 6.4 ▶️ Ejecución rápida con F5

1. Abre la carpeta `Proyecto/` en VS Code.
2. Pulsa **`F5`**, o el botón ▶️ en *Ejecutar y depurar*, y elige **"Traductor LSM (main.py)"**.

Esto funciona porque la carpeta `.vscode/` ya está configurada:

| Archivo | Qué configura |
|---|---|
| `launch.json` | Ejecuta `main.py` con `venv\Scripts\python.exe` directamente, sin depender de activar el entorno |
| `settings.json` | Fija `venv` como intérprete del proyecto y crea un perfil de terminal **"PowerShell (venv)"** con `ExecutionPolicy Bypass`, para que Windows no bloquee `Activate.ps1` |

**Alternativa desde la terminal:**

```powershell
.\venv\Scripts\python.exe main.py
```

### 6.5 Recomendaciones de uso

- 💡 Iluminación frontal y uniforme; evita tener una ventana detrás.
- ✋ La mano completa dentro del cuadro y con la **palma hacia la cámara**.
- 📏 A una distancia de entre 40 y 80 cm de la cámara.
- 🧘 Sostén cada seña quieta hasta que la barra llegue al 100 %.

> ℹ️ Los mensajes `W0000 … inference_feedback_manager` que aparecen en la consola son avisos internos de MediaPipe, no errores.

---

## 7. 🎛️ Calibración

Todos los umbrales están centralizados al inicio de `main.py`. El **modo diagnóstico** (tecla `D`) muestra en vivo los valores medidos para ajustarlos.

| Constante | Valor | Controla | Si falla… |
|---|---|---|---|
| `SEGUNDOS_PARA_ESCRIBIR` | 1.5 s | Tiempo de retención | Bájalo para escribir más rápido |
| `FRAMES_ESTABLES` | 4 | Consenso temporal | Súbelo si la etiqueta parpadea |
| `R_EXTENDIDO` | 1.45 | Umbral de dedo estirado | Bájalo si un dedo estirado no se detecta |
| `ANG_RECTO` | 150° | Rectitud mínima de un dedo estirado | Bájalo si tus dedos no quedan muy rectos |
| `R_CERRADO` | 1.05 | Umbral de dedo cerrado | Súbelo si la A o BORRAR no entran |
| `T_PULGAR_CRUZADO` | 0.20 | Pulgar sobre la palma | Separa A de BORRAR; afecta a la E |
| `PULGAR_FUERA` | 0.60 | Pulgar abierto | Afecta a L, Y y ESPACIO |
| `ANG_U_MAX` / `ANG_V_MIN` | 10° / 18° | Frontera entre U y V | Ajusta según qué tanto abres los dedos |
| `ANG_ABERTURA_MANO` | 35° | Frontera entre B y ESPACIO | — |
| `O_CONTACTO` | 0.30 | Distancia máxima pulgar–índice para la O | Súbelo si la O no entra |
| `O_T_MAX` | 0.30 | Separa la O de la E por la posición del pulgar | — |
| `L_ANG_MIN` / `L_ANG_MAX` | 50° / 130° | Ángulo válido de la L | — |

**Procedimiento:**
1. Pulsa `D` para activar el diagnóstico.
2. Haz la seña que falla.
3. Lee en pantalla el estado y el valor `r` de cada dedo, y los valores `t` y `d` del pulgar.
4. Compáralos con la tabla de la sección 2.1.4 y ajusta la constante correspondiente.

---

## 8. 🧪 Validación y pruebas

### 8.1 Pruebas unitarias con mano sintética ✅

Se construyó una **mano virtual con cinemática de flexión**: cada dedo tiene 3 articulaciones con ángulos configurables y se proyecta en 2D como lo vería una cámara frontal. Con ella se validó el clasificador:

| Conjunto de prueba | Casos | Resultado |
|---|---|---|
| Señas objetivo (A, B, E, I, L, O, U, V, W, Y, ESPACIO, BORRAR) | 12 | ✅ 12/12 |
| Posturas que **no** deben reconocerse (solo el índice levantado, mano abierta con dedos juntos, garra con el pulgar fuera, U/V ambigua) | 4 | ✅ 4/4 |
| `Escritor`: retención de 1.5 s, anti-repetición, letras dobles, ESPACIO, BORRAR | 6 | ✅ 6/6 |

> 🔍 **Hallazgo durante las pruebas:** la primera versión distinguía U de V por la distancia entre las puntas y fallaba, porque el índice y el medio tienen largos distintos. Se reemplazó por el **ángulo entre vectores**, que no depende del largo de los dedos.

### 8.2 Prueba de integración en vivo ✅

| Verificación | Resultado |
|---|---|
| Inicio de la cámara y de los modelos | ✅ Sin errores |
| Ejecución desde otro directorio de trabajo (rutas) | ✅ Correcta |
| Ventanas lado a lado sin encimarse (pantalla 1920×1080) | ✅ x = 30–686 y 700–1356 |
| Teclas `Espacio` y `Backspace` | ✅ Funcionan |
| Reconocimiento de una seña real (A) | ✅ Registrada |
| Cierre con la `X` de la ventana | ✅ Libera la cámara y los modelos |

### 8.3 Pendiente de validar

- 📊 **Precisión por letra con varios usuarios:** una matriz de confusión con al menos 3 personas y 20 intentos por letra.
- 💡 **Robustez** ante distintas condiciones de iluminación y fondos.

---

## 9. ⚠️ Limitaciones conocidas

| Limitación | Causa | Mitigación prevista |
|---|---|---|
| Solo señas estáticas | El clasificador evalúa un frame a la vez | LSTM sobre secuencias (fase 6) |
| Confusión E ↔ O en algunos ángulos | Se usa una proyección 2D; se pierde la profundidad | Usar la coordenada `z` de MediaPipe |
| Una sola mano | `max_num_hands = 1` | Ampliar para señas bimanuales |
| Umbrales fijos | Reglas diseñadas a mano | Modo diagnóstico ahora; modelo entrenado después |
| Sin componentes no manuales | Módulo facial pendiente | Fase 2 (diseño en la sección 2.2) |
| OpenCV no dibuja acentos ni la ñ | Limitación de las fuentes Hershey | Texto en ASCII en pantalla; usar Pillow en el futuro |

---

## 10. 📚 Referencia del código

### 10.1 Estructura del repositorio

```
Proyecto/
├── .vscode/
│   ├── launch.json       # Ejecución con F5 usando el Python de venv
│   └── settings.json     # Intérprete del proyecto y perfil de PowerShell
├── venv/                 # Entorno virtual (ignorado por Git)
├── .gitignore
├── main.py               # Código fuente del sistema
├── README.md             # Resumen e inicio rápido
└── DOCUMENTACION.md      # Este documento
```

### 10.2 Funciones y clases de `main.py`

| Elemento | Tipo | Responsabilidad |
|---|---|---|
| `dist(a, b)` | función | Distancia euclidiana 2D |
| `angulo(a, b, c)` | función | Ángulo articular en el vértice `b` |
| `angulo_entre(v1, v2)` | función | Ángulo entre dos vectores |
| `a_pixeles(landmarks, ancho, alto)` | función | Desnormalización a píxeles |
| `analizar_mano(p)` | función | Rasgos: `palma`, `estados`, `ratios`, `t_pulgar`, `pulgar_fuera`, `pulgar_cruzado`, `d_pulgar_indice` |
| `angulo_dedos(p, …)` | función | Apertura angular entre dos dedos |
| `clasificar_sena(p, info)` | función | Clasificador de reglas (sección 2.1.4); devuelve la seña o `None` |
| `Estabilizador` | clase | Consenso de N frames consecutivos |
| `Escritor` | clase | Máquina de estados hold-to-type: `actualizar()`, `aplicar()`, `limpiar()` |
| `dibujar_camara(…)` | función | Esqueleto, recuadro, etiqueta, barra y diagnóstico |
| `dibujar_panel(…)` | función | Panel traductor generado con NumPy |
| `colocar_ventanas(ancho)` | función | Coloca las ventanas lado a lado con `cv2.moveWindow` |
| `ventana_cerrada(nombre)` | función | Detecta el cierre de una ventana con la `X` |
| `main()` | función | Bucle principal: captura → detección → clasificación → UI → teclado |

### 10.3 Referencias

- Zhang, F. et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. arXiv:2006.10214.
- Kartynnik, Y. et al. (2019). *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs* (MediaPipe Face Mesh). arXiv:1907.06724.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8).
- Documentación de MediaPipe Solutions: https://ai.google.dev/edge/mediapipe/solutions/guide
- Documentación de OpenCV: https://docs.opencv.org/4.x/
- Consejo Nacional para Prevenir la Discriminación (CONAPRED). *Diccionario de Lengua de Señas Mexicana*.
