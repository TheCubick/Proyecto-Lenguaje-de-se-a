# Sistema Inteligente de Traducción de Lengua de Señas con Visión Artificial y Componentes Multimodales

> **Documentación técnica oficial**
> **Materia:** Inteligencia Artificial
> **Stack:** Python 3.12 · MediaPipe 0.10.18 · OpenCV 4.11 · NumPy · pyttsx3
> **Repositorio:** https://github.com/TheCubick/Proyecto-Lenguaje-de-se-a

### Leyenda de estado

En este documento cada componente lleva una etiqueta que indica su grado de avance real:

| Etiqueta | Significado |
|---|---|
| **Implementado** | Funciona en el código actual (`main.py`) y se validó con grabaciones reales (sección 8) |
| **Diseñado** | Arquitectura y algoritmo definidos; pendiente de construir |
| **Planeado** | Forma parte del roadmap a mediano o largo plazo |

---

## Índice

1. [Descripción general](#1-descripción-general)
2. [Arquitectura del sistema (enfoque multimodal)](#2-arquitectura-del-sistema-enfoque-multimodal)
   - 2.1 [Módulo de manos: alfabeto estático](#21-módulo-de-manos-alfabeto-estático-mediapipe-hands)
   - 2.2 [Señas con movimiento](#22-señas-con-movimiento-buffer-de-trayectoria)
   - 2.3 [Módulo facial](#23-módulo-facial-componentes-no-manuales)
   - 2.4 [Voz asíncrona](#24-voz-asíncrona-text-to-speech)
3. [Flujo de trabajo y experiencia de usuario](#3-flujo-de-trabajo-y-experiencia-de-usuario-ux--ui)
4. [Integración de hardware](#4-integración-de-hardware)
5. [Roadmap y futuro del proyecto](#5-roadmap-y-futuro-del-proyecto)
6. [Guía de instalación y ejecución](#6-guía-de-instalación-y-ejecución)
7. [Calibración](#7-calibración)
8. [Validación con datos reales](#8-validación-con-datos-reales)
9. [Limitaciones conocidas](#9-limitaciones-conocidas)
10. [Decisiones de diseño respaldadas por datos](#10-decisiones-de-diseño-respaldadas-por-datos)
11. [Referencia del código](#11-referencia-del-código)

---

## 1. Descripción general

### 1.1 Propósito

El sistema traduce **en tiempo real** el **alfabeto dactilológico completo de la Lengua de Señas Mexicana (LSM)** a texto y a voz, usando solamente una cámara web. Combina tres canales:

| Canal | Qué aporta | Estado |
|---|---|---|
| **Manual** (manos) | Qué letra o seña se hace: 21 letras estáticas, 6 letras con movimiento y señas de control | Implementado |
| **No manual** (rostro y cabeza) | Si la frase es pregunta, negación o afirmación | Implementado |
| **Voz** | Dice cada letra al registrarla y lee la palabra completa | Implementado |

La LSM no se comunica solo con las manos: los **componentes no manuales** (expresión facial, movimientos de cabeza) tienen función gramatical. Por eso la arquitectura es **multimodal**.

### 1.2 Problema que atiende

La comunidad sorda enfrenta barreras de comunicación cotidianas con personas oyentes que no conocen la lengua de señas. Las soluciones comerciales suelen requerir guantes con sensores o hardware costoso. Este proyecto explora una alternativa **de bajo costo, sin contacto y basada en visión artificial**, que funciona en cualquier computadora con cámara.

### 1.3 Enfoque técnico

1. **Estimación de pose con aprendizaje profundo** (modelos preentrenados de MediaPipe): 21 puntos clave por mano y 468 por rostro.
2. **Clasificadores geométricos basados en reglas**, diseñados por el equipo y **calibrados con grabaciones reales**: distancias normalizadas, ángulos articulares, proyecciones vectoriales y análisis de trayectorias.

| Ventaja del enfoque | Explicación |
|---|---|
| **Explicable** | Cada decisión se justifica con una regla geométrica concreta |
| **Sin dataset previo** | No requiere recolectar ni etiquetar miles de imágenes |
| **Tiempo real** | ≈ 28 FPS en una laptop sin GPU |
| **Invariante** | No depende de la distancia a la cámara, de la posición ni de si se usa la mano izquierda o la derecha |

### 1.4 Alfabeto de referencia

Las señas se definieron a partir de la **lámina oficial "Abecedario en Lengua de Señas Mexicana"** de la SEP y Aprende.mx, que tiene 26 letras y usa flechas rojas para las que llevan movimiento. La **Ñ** se forma como la N con un vaivén lateral. **CH, LL y RR** no están incluidas.

---

## 2. Arquitectura del sistema (enfoque multimodal)

```
                            ┌────────────────────────┐
                            │      Cámara web        │
                            └───────────┬────────────┘
                                        │ frame BGR (≈ 28 FPS)
                            ┌───────────▼────────────┐
                            │ Espejo · BGR → RGB     │
                            └─────┬────────────┬─────┘
                                  │            │
       ┌──────────────────────────▼───┐   ┌────▼───────────────────────────┐
       │ MANOS (MediaPipe Hands)      │   │ ROSTRO (MediaPipe Face Mesh)   │
       │ 21 landmarks                 │   │ 468 landmarks                  │
       ├──────────────────────────────┤   ├────────────────────────────────┤
       │ analizar_mano  → rasgos      │   │ calcular_rasgos_faciales       │
       │ clasificar_pose → letra/pose │   │ AnalizadorFacial               │
       │ Trayectoria (deque, 45 fr.)  │   │  · cejas (línea base personal) │
       │ ReconocedorDinamico          │   │  · negar · asentir             │
       └───────────────┬──────────────┘   └───────────────┬────────────────┘
          letra estable│trazo reconocido                  │ ?  ·  borrar  ·  confirmar
                       └─────────────────┬────────────────┘
                              ┌──────────▼───────────┐
                              │ Escritor             │  texto + eventos
                              │ hold-to-type 1.5 s   │
                              └─────┬─────────┬──────┘
              ┌─────────────────────┤         ├─────────────────────┐
     ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────────────┐
     │ Ventana de      │   │ Panel           │   │ Hilo de voz (pyttsx3)   │
     │ cámara          │   │ traductor       │   │ + enviar_a_hardware()   │
     └─────────────────┘   └─────────────────┘   └─────────────────────────┘
```

---

### 2.1 Módulo de manos: alfabeto estático (MediaPipe Hands)

#### 2.1.1 Detección

| Parámetro | Valor | Justificación |
|---|---|---|
| `static_image_mode` | `False` | Seguimiento temporal: más rápido y estable |
| `max_num_hands` | `1` | Las señas son de una mano |
| `min_detection_confidence` | `0.7` | Reduce detecciones falsas |
| `min_tracking_confidence` | `0.6` | Mantiene el seguimiento con movimiento |

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
| 1. Desnormalización | `(x·ancho, y·alto)` | MediaPipe da coordenadas en [0, 1]; sin esto, una misma distancia mediría distinto en horizontal y en vertical |
| 2. Escala | `escala = max(‖P₀ − P₉‖, 1.55 · ‖P₅ − P₁₇‖)` | Todas las medidas se expresan en "manos". Usa solo puntos rígidos de la palma, así que no cambia con la seña, y el máximo la hace robusta a la rotación |
| 3. Marco de la mano | eje `P₅ → P₁₇` y eje de la palma `P₀ → P₉` | El pulgar y la orientación se miden respecto a la propia mano, por eso funciona igual con la mano izquierda o la derecha |

> **¿Por qué no normalizar por el *bounding box*?** Porque la caja cambia con la propia seña: un puño tiene una caja mucho menor que una mano abierta, y los umbrales dependerían de la letra que se quiere reconocer.

#### 2.1.3 Rasgos que se calculan (`RasgosMano`)

| Rasgo | Cómo se mide | Para qué sirve |
|---|---|---|
| **Estado de cada dedo** (`EXT` / `DOB` / `CER`) | razón `r = ‖punta−muñeca‖ / ‖nudillo−muñeca‖` y ángulo en la articulación media | Base de casi todas las letras |
| **Pulgar abierto** | `‖punta pulgar − nudillo índice‖ / escala ≥ 0.48` y `t < 0` | A, L, G, Y, P. Se mide por **distancia** y no por rectitud, porque el pulgar parpadea entre "recto" y "doblado" |
| **Posición del pulgar `t`** | proyección de la punta sobre el eje índice → meñique | Cruza la palma (E) o queda por fuera (C) |
| **Orientación de la mano** | ángulo del eje de la palma respecto a "arriba": 0° arriba · 90° de lado · 180° abajo | G y H (de lado), M y N (hacia abajo) |
| **Dirección del índice** | ángulo nudillo → punta del índice respecto a "arriba" | Separa Q (cuelga) de X (de lado) |
| **Cruce de índice y medio** | separación **con signo** de las puntas a lo largo del eje de los nudillos | R (entrelazados), P (en "X") |
| **Apertura entre dedos** | ángulo entre vectores nudillo → punta | U, V, B, ESPACIO |
| **Contacto del pulgar** | distancia punta del pulgar ↔ punta de cada dedo | O, F, C |

#### 2.1.4 Clasificador (`clasificar_pose`)

Se decide en este orden: **cuántos dedos están estirados → orientación → detalles del pulgar**. Si ninguna regla coincide devuelve `None`: el sistema prefiere no responder antes que equivocarse.

| Seña | Cómo se hace (lámina SEP) | Regla implementada |
|---|---|---|
| **A** | Puño con el pulgar estirado al lado | 4 dedos cerrados + pulgar abierto |
| **B** | 4 dedos juntos arriba, pulgar sobre la palma | 4 estirados, apertura índice-meñique < 20°, pulgar no abierto |
| **C** | Mano curva en forma de C | ≥ 2 dedos doblados, pulgar por fuera (`t < 0.10`), hueco pulgar-índice entre 0.20 y 1.30 |
| **D** | Índice arriba, los demás cerrados | solo el índice estirado, pulgar no abierto, mano hacia arriba |
| **E** | Dedos doblados sobre el pulgar | ≥ 2 doblados, pulgar cruzado (`t ≥ 0.20`) |
| **F** | Índice toca el pulgar, 3 dedos arriba | medio, anular y meñique estirados, contacto pulgar-índice < 0.35 |
| **G** | "Pistola": índice de lado, pulgar arriba | índice estirado, pulgar abierto a 50°-130°, mano de lado |
| **H** | "Pistola" con dos dedos | índice y medio estirados, mano de lado |
| **I** | Solo el meñique arriba | meñique estirado, pulgar no abierto |
| **L** | Índice arriba y pulgar al lado | índice estirado, pulgar abierto a 50°-130°, mano arriba |
| **M** | 3 dedos hacia abajo | índice, medio y anular estirados, mano hacia abajo |
| **N** | 2 dedos hacia abajo | índice y medio estirados, mano hacia abajo |
| **O** | Círculo índice-pulgar | ningún dedo estirado, contacto pulgar-índice < 0.20 |
| **P** | Índice y medio cruzados en "X" | índice y medio arriba con separación de puntas ≤ −0.18, **o** con el pulgar abierto |
| **R** | Índice y medio entrelazados | índice y medio arriba con las puntas **cruzadas** (separación < 0) |
| **S** | Puño, pulgar sobre los dedos | 4 cerrados, pulgar a menos de 0.20 del nudillo del índice |
| **T** | Puño, pulgar entre los dedos | 4 cerrados, pulgar a 0.20 o más del nudillo del índice |
| **U** | Índice y medio juntos | índice y medio arriba, ángulo < 14° |
| **V** | Índice y medio en V | índice y medio arriba, ángulo ≥ 25° |
| **W** | Índice, medio y anular abiertos | 3 estirados, mano arriba |
| **Y** | Pulgar y meñique estirados | meñique estirado y pulgar abierto |
| **ESPACIO** | Mano abierta, 5 dedos separados | 4 estirados, pulgar recto y lejos (> 0.65), apertura ≥ 48° |

> **La zona intermedia entre B (< 20°) y ESPACIO (≥ 48°) es la "mano relajada" y no escribe nada.** Se midió que una mano abierta y relajada queda en ≈ 40°; con el umbral original (35°) caía en B y habría escrito letras sin querer.

```
                         ¿Cuántos dedos estirados?
     ┌───────┬───────────┬──────────┬──────────┬───────────────┐
     4       3           2          1                          0
     │       │           │          │                          │
  F / M /  F · W · M   ¿orientación? ¿cuál?             O · A · S · T
  ESPACIO / B          abajo → N    índice: L/G/D/(Q/X)  · E · C
  / ABIERTA            lado  → H    meñique: I/Y
                       arriba: P · R · V · U
```

---

### 2.2 Señas con movimiento (buffer de trayectoria)

Ocho gestos necesitan **movimiento** y no se pueden reconocer con un solo fotograma: seis letras (J, K, Ñ, Q, X, Z) y dos señas de control (BORRAR y el `?` dibujado en el aire).

#### 2.2.1 Buffer

`Trayectoria` guarda en una `collections.deque` de **45 fotogramas** (≈ 1.5 s) las posiciones de la muñeca, el centro de la palma, las puntas de los dedos y la configuración vista en cada frame. Todas las posiciones se **normalizan por el tamaño de la mano**, así que el trazo no depende de la distancia a la cámara.

> **Se usan 45 fotogramas y no los 15-20 que se habían propuesto.** Con grabaciones reales, una Z dibujada a ritmo natural dura **1.4-2 s**; en 20 fotogramas (0.7 s) no cabe un trazo de tres movimientos. Cada gesto mira solo la parte del buffer que necesita (columna "Ventana").

#### 2.2.2 Reconocedores

Cada gesto exige **una configuración de mano** durante al menos el 60 % de su ventana (45 % para Q y X) y **un patrón de movimiento**. Todos aceptan el trazo en espejo (zurdos, o quien dibuja "como en el espejo") salvo BORRAR.

| Gesto | Configuración | Punto rastreado | Ventana | Patrón |
|---|---|---|---|---|
| **J** | I o Y (el pulgar parpadea) | punta del meñique | 30 | baja ≥ 0.55 manos y se curva ≥ 0.25 hacia un lado (gancho) |
| **Z** | D | punta del índice | 45 | 3 trazos: barra →, diagonal ↙ (rumbo 225° ± 35°), barra →; barras ≥ 0.40, diagonal ≥ 0.45 |
| **?** | D | punta del índice | 45 | arco que abulta a un lado, regresa al centro y termina en un tallo vertical hacia abajo; altura ≥ 0.80 |
| **K** | P, R o U | punto medio de las puntas índice-medio | 45 | **vaivén** lateral: ≥ 2 cambios de sentido de ≥ 0.15 manos |
| **Ñ** | N (o H si la N queda de lado) | centro de la palma | 30 | vaivén lateral: ≥ 2 cambios de sentido de ≥ 0.10 manos |
| **Q** | índice curvo, resto cerrado, **índice hacia abajo** (≥ 120°) | punta del índice | 24 | vaivén de ≥ 0.07 manos |
| **X** | índice curvo, resto cerrado, **índice de lado o arriba** (< 120°) | punta del índice | 24 | el mismo vaivén |
| **BORRAR** | mano abierta (B, ESPACIO o abierta) | centro de la palma | 15 | barrido **a la izquierda** ≥ 1.10 manos, ≥ 3.0 manos/s, ≥ 85 % monótono |

**Técnicas usadas:** simplificación de trazos con **Ramer-Douglas-Peucker** y análisis por segmentos (rumbo y largo) para la Z; conteo de cambios de sentido con histéresis para los vaivenes; velocidad pico en ventanas de 4 muestras para BORRAR.

#### 2.2.3 Reglas que evitan errores

| Regla | Qué evita |
|---|---|
| Una letra **estática** solo avanza si la palma se mueve a menos de 1.2 manos/s | que moverse mientras se forma la seña escriba letras |
| Tras un trazo, la configuración base queda **bloqueada** hasta que la mano cambie o baje (p. ej. la I tras dibujar una J) | que la I se escriba 1.5 s después de la J |
| Tras un vaivén hace falta una **pausa** de 0.3 s con la palma casi quieta | que agitar seguido escriba `Q Q Q` |
| Enfriamiento de 1.0 s entre trazos | duplicados |
| El buffer se vacía si la mano desaparece más de 5 fotogramas | trazos que mezclan dos intentos |

> **BORRAR cambió de puño a barrido.** Con el alfabeto completo, un puño con el pulgar adentro es la **S**, la **T** o la **A**; borrar con un puño habría borrado letras al escribirlas. Quedan además `Backspace` y negar con la cabeza.

---

### 2.3 Módulo facial (componentes no manuales)

#### 2.3.1 Fundamento lingüístico

| Componente no manual | Función gramatical | Acción en el sistema |
|---|---|---|
| Cejas levantadas | Pregunta cerrada (sí/no) | Agrega `?` |
| Giro lateral repetido de la cabeza | Negación | Borra la última letra |
| Asentir con la cabeza | Afirmación | Confirma la palabra, agrega un espacio y la lee en voz alta |
| Cejas fruncidas | Pregunta abierta (qué, dónde, quién…) | Planeado |

> **¿Existe una seña manual de "pregunta" en la LSM?** No se encontró ninguna fuente que describa un gesto manual con el dedo índice. Lo documentado es la marca facial (cejas levantadas para preguntas cerradas, fruncidas para abiertas). El **"?" dibujado en el aire con el índice** es por tanto un **respaldo propio del sistema**, no una seña oficial de la LSM.

#### 2.3.2 Métrica de las cejas (`calcular_rasgos_faciales`)

Se usan **5 puntos del contorno superior de cada ceja** (Face Mesh 70·63·105·66·107 y 300·293·334·296·336), las **comisuras de los ojos** (33·133 y 362·263) como referencia ósea y el **eje de la cara** (mentón 152 → frente 10):

```
u = (P₁₀ − P₁₅₂) / H          H = ‖P₁₀ − P₁₅₂‖  (alto total del rostro)

E_lado = promedio sobre los 5 puntos de ceja de  (P_ceja − Ō_ojo) · u
E      = ½ (E_der + E_izq) / H            Ō_ojo = punto medio de las comisuras
```

- **Se proyecta sobre el eje de la cara:** no importa si se inclina o gira la cabeza.
- **Se divide entre el alto del rostro:** no importa la distancia a la cámara.
- **Se usa la comisura y no el párpado:** el párpado baja al parpadear o mirar abajo y parecería una ceja levantada.
- **Línea base personal:** percentil 35 de 45 fotogramas con la cara neutra (ignora fotogramas con las cejas ya levantadas); se reajusta despacio mientras la cara está neutra; la tecla `R` recalibra.

#### 2.3.3 Por qué fallaba la versión anterior (diagnóstico con datos reales)

Se grabó una sesión guiada (cara neutra, cejas arriba dos veces, mirar abajo, parpadear, girar, negar, asentir). Razón respecto a la línea base (mediana):

| Métrica | Subida de ceja | Mirar abajo | Parpadeo | Giro | Negar | Asentir | Ruido en reposo |
|---|---|---|---|---|---|---|---|
| **Anterior** (distancia euclidiana / distancia entre ojos) | 1.12 | — | — | 1.01 (p95 **1.19**) | p95 **1.20** | — | 1.6 % |
| **Párpado** (proyectada / alto de rostro) | 1.14 (p95 1.25-1.29) | p95 1.14 | 1.08 (p95 **1.19**) | — | — | — | **4.0 %** |
| **Actual** (comisura, proyectada, / alto de rostro) | **1.10-1.12** | 0.99 (p95 1.03) | 1.01 (p95 1.06) | 0.98 (p95 1.01) | 0.99 (p95 1.03) | 0.99 (p95 1.07) | **1.5 %** |

**Causa del fallo:** una subida real de ceja da solo **+10 a +17 %** en esa métrica, y el umbral era **1.15**: solo el 29-37 % de los fotogramas lo superaba y nunca se sostenía 1.5 s seguidos. Además, la métrica anterior se dejaba engañar por el giro de la cabeza (hasta +19 %).

Con el umbral nuevo (**1.07**, salida **1.04**), la racha continua más larga por encima del umbral fue de **2.1-2.2 s** en las dos subidas (que en total duraron unos 3.6 s) y de **≤ 0.2 s** en todas las demás fases: no hay falsos positivos.

#### 2.3.4 Detección de la pregunta

```
razón = E / E_base
   razón ≥ 1.07                 → empieza a contar (cejas arriba)
   1.04 ≤ razón < 1.07          → histéresis: sigue contando
   razón < 1.04 por más de 0.30 s → se reinicia (un parpadeo no cuenta)
   1.5 s seguidos arriba        → "?"  (solo una por cada vez que se levantan las cejas)
```

Después de negar o asentir, las cejas se ignoran 0.8 s (esos movimientos no deben contar como pregunta). Un **medidor siempre visible** en la ventana de cámara muestra la razón en vivo y la marca del umbral.

#### 2.3.5 Movimientos de cabeza (`DetectorOscilacion`)

Negar y asentir son **patrones temporales**. Una media móvil exponencial (τ = 2.5 s) sigue la postura neutra de la cabeza, así que mirar a un lado y quedarse ahí no cuenta. Se cuentan las **entradas** a zonas con histéresis dentro de una ventana de 1.4 s:

| Gesto | Señal | Amplitud | Condición |
|---|---|---|---|
| Negación | giro (yaw) de la nariz entre los pómulos | 0.08 | ≥ 3 entradas **en ambos lados** |
| Afirmación | inclinación (pitch) de la nariz | 0.035 | ≥ 2 entradas |

Tras un gesto de cabeza hay un enfriamiento de 1.2 s.

---

### 2.4 Voz asíncrona (text-to-speech)

`pyttsx3` (SAPI5 en Windows) corre en **un hilo propio** (`VozAsincrona`), así que hablar **nunca bloquea la cámara**.

| Aspecto | Diseño |
|---|---|
| **Hilo** | El motor se crea **dentro** del hilo (en Windows, SAPI5 queda ligado al hilo que lo crea) y se inicializa COM en él |
| **Voz** | Prefiere español de México (*Microsoft Sabina*); si no, cualquier voz en español; si no, la voz por defecto |
| **Arranque** | Una frase de calentamiento en silencio, para que la primera letra real no se retrase |
| **Cola** | `decir()` regresa al instante. Una **palabra** sustituye a las letras pendientes; si se acumulan más de 2 letras, se descartan las más viejas |
| **Nombres de letras** | Se dice el nombre en español (`jota`, `eñe`, `doble ve`, `equis`…) y no un sonido suelto |
| **Tolerancia a fallos** | Si falta `pyttsx3` o una voz, el programa sigue sin voz y lo indica en el panel (`Voz: no disponible`) |

**Cuándo habla** (`PoliticaVoz`):

| Evento | Qué dice |
|---|---|
| Se registra una letra (barra al 100 % o trazo completo) | el nombre de la letra |
| Se **confirma** la palabra (asentir) | la palabra completa |
| Pasan **4 s sin nuevas letras** | la palabra pendiente (una sola vez) |
| Espacio, borrar, `?` | nada |

Un segundo asentimiento, o la inactividad, **no repiten** una palabra ya leída.

---

## 3. Flujo de trabajo y experiencia de usuario (UX / UI)

### 3.1 Estabilización

`Estabilizador`: una configuración nueva solo se acepta si aparece en **4 fotogramas consecutivos** (≈ 140 ms), lo que elimina el parpadeo de la etiqueta.

### 3.2 Constructor de palabras *Hold-to-Type*

| Característica | Detalle |
|---|---|
| Retención | Una seña **estática** se registra al sostenerla **quieta 1.5 s** (`time.monotonic()`, no depende de los FPS) |
| Barra de progreso | Verde, redondeada y suavizada; **se reinicia al instante** si cambia la seña |
| Anti-repetición | No se repite hasta cambiar de seña o bajar la mano |
| Señas con movimiento | Se registran **al completar el trazo** (la barra salta al 100 %) |
| Texto | Los espacios se ven como espacios y **cualquier guion bajo se reemplaza por un espacio** en pantalla, en la voz y en lo que se envía al hardware (`limpiar_texto`) |

### 3.3 Interfaz dual

**Ventana "Camara IA - Deteccion LSM"**
- Video en espejo con el esqueleto de la mano y los puntos de cejas y ojos.
- Recuadro con esquinas resaltadas (gris: sin seña · azul: cargando · verde: registrada · ámbar: configuración que necesita movimiento).
- Etiqueta con la seña y una pista (`o traza una J`, `o agita para K`…).
- **Estela** del dedo mientras se dibuja un trazo.
- **Medidor de cejas siempre visible** (`CEJAS x1.11`) con la marca del umbral: se ve cómo se "carga" la pregunta.
- Indicador `Modo: Neutro / Pregunta ? / Negacion X / Afirmacion OK / Calibrando...`.
- Barra de progreso inferior y contador de FPS.
- Modo diagnóstico (`D`): estado de cada dedo, métricas del pulgar y de la cabeza.

**Ventana "Panel Traductor LSM"** (lienzo NumPy, sin imágenes externas): la seña actual en grande (con la **Ñ** dibujada a mano, porque las fuentes de OpenCV no la incluyen), cómo se hace, la **guía del alfabeto** en dos columnas (en naranja las que llevan movimiento y resaltada la que se está haciendo), un recordatorio de los gestos del rostro, la caja de texto y el estado de la voz. El fondo del panel se dibuja una sola vez y se copia en cada fotograma.

### 3.4 Teclado

| Tecla | Acción |
|---|---|
| `Espacio` | Agregar un espacio |
| `Backspace` | Borrar la última letra |
| `C` | Limpiar todo el texto |
| `D` | Modo diagnóstico |
| `R` | Recalibrar el rostro |
| `ESC` o la `X` de cualquier ventana | Salir limpiamente |

---

## 4. Integración de hardware

> **Estado:** el **puente de software está implementado**: `enviar_a_hardware(texto_actual)` se llama **solo cuando el texto cambia** (por una seña, por el rostro o por el teclado) e imprime `[SERIAL MOCK] Enviando al ESP32: <texto>`. El **hardware físico** (ESP32 y pantalla) está en fase de **diseño**. Para conectarlo basta con reemplazar el cuerpo de esa función por el envío serial de la sección 4.2.

### 4.1 Arquitectura propuesta

```
┌─────────────────────────────┐                 ┌───────────────────────────────────┐
│     PC (visión artificial)  │                 │    Protoboard                     │
│                             │   USB / UART    │  ┌─────────────┐   I²C   ┌───────┐ │
│  main.py                    │ ──────────────▶ │  │ ESP32 /     │ ──────▶ │ OLED  │ │
│   └─ enviar_a_hardware()    │   115200 baud   │  │ Arduino Uno │         │128×64 │ │
│                             │   ── o bien ──  │  └─────────────┘         └───────┘ │
│   └─ WebSocket (ESP32)      │ ~~~ Wi-Fi ~~~~▶ │        ▲ (solo ESP32)              │
└─────────────────────────────┘                 └───────────────────────────────────┘
```

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

### 4.2 Protocolo de comunicación

Protocolo de texto línea por línea, sencillo de depurar con cualquier monitor serial:

| Mensaje | Significado | Ejemplo |
|---|---|---|
| `T:<texto>\n` | Texto acumulado completo | `T:HOLA LSM` |
| `C\n` | Limpiar la pantalla | `C` |

**Reglas de diseño:**
- Se envía **solo cuando algo cambia**, no en cada frame.
- Se envía el **texto completo** en lugar de diferencias: si se pierde un mensaje, el siguiente corrige el estado.
- Se usa `\n` como delimitador, compatible con `Serial.readStringUntil('\n')` en Arduino.

**Envío desde la PC (propuesta con `pyserial`):**

```python
import serial  # pip install pyserial

puerto = serial.Serial("COM3", 115200, timeout=0)
puerto.write(f"T:{limpiar_texto(texto_actual)}\n".encode("utf-8"))
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

> **Dos notas para la etapa de hardware:**
> 1. La fuente por defecto de Adafruit GFX **no tiene la Ñ**. Con la librería `U8g2` y una fuente Latin-1 (por ejemplo `u8g2_font_6x12_tf`) sí se puede mostrar.
> 2. Un ESP32 **no puede ejecutar MediaPipe**: la visión sigue en la PC (o en una Raspberry Pi o un teléfono) y el microcontrolador solo muestra el texto.

### 4.3 Variante inalámbrica (Wi-Fi, solo ESP32)

| Aspecto | Propuesta |
|---|---|
| Rol del ESP32 | Servidor WebSocket en la red local (o *access point* propio) |
| Rol de la PC | Cliente que envía los mismos mensajes |
| Ventaja | El dispositivo no necesita cable de datos: solo alimentación |
| Riesgo | Latencia variable de la red; conviene usar un *timeout* y reintentos |

### 4.4 Camino hacia la autonomía total

```
 Etapa 1 (actual)         Etapa 2 (próxima)             Etapa 3 (portátil)
 ┌──────────────┐         ┌──────────────────────┐      ┌──────────────────────────┐
 │ PC: visión + │   ──▶   │ PC: visión           │ ──▶  │ SBC (Raspberry Pi 5)     │
 │ pantalla     │         │ ESP32: pantalla OLED │      │ o teléfono: visión       │
 └──────────────┘         └──────────────────────┘      │ ESP32 + batería: pantalla│
                                                        └──────────────────────────┘
```

---

## 5. Roadmap y futuro del proyecto

| Fase | Entregable | Estado |
|---|---|---|
| **1. Prototipo manual** | Letras estáticas, hold-to-type, interfaz dual | Completado |
| **2. Componentes no manuales** | Pregunta, negación y afirmación por rostro y cabeza | Completado |
| **3. Puente de hardware** | `enviar_a_hardware()` (mock) | Completado |
| **4. Alfabeto completo** | 21 letras estáticas + J, K, Ñ, Q, X, Z con movimiento | Completado (varias letras con fiabilidad media o baja, ver sección 8) |
| **5. Voz** | Cada letra y la palabra completa, en un hilo aparte | Completado |
| **6. Hardware físico** | Pantalla OLED en protoboard por USB o Wi-Fi | Diseñado |
| **7. Machine Learning** | Clasificador entrenado con landmarks propios (k-NN, Random Forest, red densa) para las letras ambiguas (F, R, U, P, C, Q, X) | Planeado |
| **8. Señas dinámicas con LSTM / GRU** | Reemplazar las reglas de trazo; añadir CH, LL y RR | Planeado |
| **9. Frases continuas** | Modelos secuenciales / Transformers | Planeado |
| **10. Impacto social** | App móvil o dispositivo cerrado; validación con personas usuarias de LSM | Planeado |

### 5.1 Hacia el aprendizaje automático

El clasificador de reglas es explicable, pero escala mal: cada letra nueva exige diseñar reglas a mano. La evolución natural es **aprender las fronteras de decisión a partir de datos**:

```
 Captura de datos        Vector de rasgos                 Modelo                Salida
 ┌──────────────┐       ┌───────────────────────┐       ┌───────────────┐      ┌─────────┐
 │ sesiones     │  ──▶  │ 21 landmarks × (x,y,z)│ ──▶   │ k-NN / RF     │ ──▶  │ letra   │
 │ guiadas      │       │ normalizados = 63 val.│       │ (estáticas)   │      │         │
 └──────────────┘       └───────────────────────┘       ├───────────────┤      ├─────────┤
                                                        │ LSTM sobre    │ ──▶  │ seña    │
                        secuencia de T frames ──▶       │ T × 63        │      │ dinámica│
                                                        └───────────────┘      └─────────┘
```

El método de las **sesiones guiadas** que se usó para calibrar (la ventana muestra la lámina de cada letra y se graban solo coordenadas de landmarks) ya es una herramienta de **captura de datos**. Con varias personas se podría entrenar un clasificador sobre 63 números normalizados por fotograma, mucho más ligero que una CNN sobre imágenes.

**Por qué LSTM para las señas dinámicas:** una seña en movimiento es una **serie temporal** de poses; las redes LSTM mantienen un estado interno que captura la dependencia entre fotogramas y permitirían reemplazar las reglas de trazo actuales.

### 5.2 Impacto social

- **Aplicación móvil:** MediaPipe tiene soporte nativo para Android e iOS, así que el mismo pipeline puede llevarse al teléfono.
- **Herramienta educativa:** practicar el alfabeto dactilológico con retroalimentación inmediata.
- **Validación con la comunidad:** pruebas con personas usuarias de LSM para corregir la forma de las señas y priorizar el vocabulario.

---

## 6. Guía de instalación y ejecución

### 6.1 Requisitos

| Requisito | Versión / detalle |
|---|---|
| Sistema operativo | Windows 10/11 |
| Python | **3.12** (MediaPipe 0.10.18 **no** es compatible con 3.13) |
| Hardware | Cámara web integrada o USB |
| Voz | Una voz SAPI5 en español (en Windows: *Microsoft Sabina* o *Helena*). Si falta, el programa funciona sin voz |
| Editor | VS Code con la extensión **Python** |

### 6.2 Dependencias

| Librería | Uso |
|---|---|
| `mediapipe==0.10.18` | Detección de la mano (Hands) y del rostro (Face Mesh) |
| `opencv-python` | Captura de video, dibujo y ventanas |
| `numpy` | Panel, trayectorias y operaciones con matrices |
| `pyttsx3` | Voz (en Windows instala también `comtypes` y `pywin32`) |

### 6.3 Crear el entorno virtual (`venv`)

```powershell
# 1. Crear el entorno con Python 3.12
py -3.12 -m venv venv

# 2. Instalar dependencias
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.18 numpy pyttsx3
```

> La carpeta `venv/` está en `.gitignore`: cada integrante crea la suya localmente y no se sube al repositorio.

### 6.4 Ejecución rápida con F5

1. Abre la carpeta `Proyecto/` en VS Code.
2. Pulsa **`F5`**, o el botón en *Ejecutar y depurar*, y elige **"Traductor LSM (main.py)"**.

| Archivo | Qué configura |
|---|---|
| `.vscode/launch.json` | Ejecuta `main.py` con `venv\Scripts\python.exe` directamente, sin depender de activar el entorno |
| `.vscode/settings.json` | Fija `venv` como intérprete del proyecto y crea un perfil de terminal **"PowerShell (venv)"** con `ExecutionPolicy Bypass`, para que Windows no bloquee `Activate.ps1` |

**Alternativa desde la terminal:** `.\venv\Scripts\python.exe main.py`

### 6.5 Recomendaciones de uso

- Iluminación frontal y uniforme; evita tener una ventana detrás.
- Mano completa dentro del cuadro, a 40-80 cm de la cámara.
- Al iniciar, **mantén la cara neutra** unos 1.5 s (`Modo: Calibrando...`).
- **Los trazos (Z, ?, J…) deben ser pequeños, del tamaño de la mano**, sin sacarla del cuadro: si sale, la cámara pierde el seguimiento.
- Sostén cada letra **quieta** hasta que la barra llegue al 100 %.

> Los mensajes `W0000 … inference_feedback_manager` que aparecen en la consola son avisos internos de MediaPipe, no errores.

---

## 7. Calibración

Todos los umbrales están al inicio de `main.py`. El **modo diagnóstico** (tecla `D`) muestra los valores medidos en vivo.

| Constante | Valor | Controla |
|---|---|---|
| `SEGUNDOS_PARA_ESCRIBIR` | 1.5 s | Retención de una seña estática |
| `FRAMES_ESTABLES` | 4 | Consenso para aceptar una seña |
| `V_QUIETO` | 1.2 manos/s | Por encima, la mano "se mueve" y no escribe letras estáticas |
| `PULGAR_ABIERTO` / `PULGAR_FUERA` | 0.48 / 0.65 | Pulgar apartado / pulgar recto y lejos |
| `ANG_B_MAX` / `ANG_ESPACIO_MIN` | 20° / 48° | B y ESPACIO (entre ambos, mano relajada) |
| `ANG_U_MAX` / `ANG_V_MIN` | 14° / 25° | U y V |
| `O_CONTACTO` / `F_CONTACTO` | 0.20 / 0.35 | Contacto pulgar-índice para O y F |
| `ST_DISTANCIA` | 0.20 | Frontera entre S y T |
| `Q_DIR_INDICE_MIN` | 120° | Frontera entre Q y X |
| `TRAYECTORIA_FRAMES` | 45 | Memoria del buffer |
| `VAIVEN_AMP` | Ñ 0.10 · Q/X 0.07 · K 0.15 | Amplitud mínima de cada vaivén |
| `BORRAR_RECORRIDO_MIN` / `BORRAR_VEL_MIN` | 1.10 / 3.0 | Barrido de BORRAR |
| `CEJAS_UMBRAL` / `CEJAS_UMBRAL_SALIDA` | 1.07 / 1.04 | Cejas arriba / de vuelta a neutro |
| `CEJAS_SEGUNDOS` | 1.5 s | Tiempo con las cejas arriba para agregar `?` |
| `NEG_AMPLITUD` / `AFIRM_AMPLITUD` | 0.08 / 0.035 | Negar y asentir |
| `ACCION_NEGACION` | `"LETRA"` | Qué borra negar: `"LETRA"`, `"PALABRA"` o `"TODO"` |
| `VOZ_INACTIVIDAD` | 4.0 s | Tiempo sin letras para leer la palabra pendiente |
| `VOZ_HABILITADA` | `True` | Activar o desactivar la voz |

**Procedimiento:** pulsa `D`, haz la seña que falla, lee el estado de cada dedo y los valores del pulgar, compáralos con la tabla de la sección 2.1.4 y ajusta la constante correspondiente.

---

## 8. Validación con datos reales

### 8.1 Metodología

Se grabaron **cuatro sesiones guiadas** con la cámara real del autor: una ventana mostraba la **lámina oficial** de cada letra (con una animación del trazo para los gestos con movimiento) y se guardaron **solo coordenadas de landmarks**, sin imágenes. Después cada grabación se **reprodujo a través del mismo código de `main.py`** (`Estabilizador`, `Trayectoria`, `Escritor`…), de modo que los resultados son los del programa real.

| Sesión | Duración | Fotogramas | Fotogramas con mano o cara |
|---|---|---|---|
| Cara (cejas, mirar, parpadear, girar, negar, asentir) | 39 s | 793 | 100 % cara |
| Letras estáticas (23 fases) | 103 s | 2 978 | 97 % mano |
| Trazos 1 | 67 s | 1 950 | 67 % mano |
| Trazos 2 (con animación del recorrido) | 72 s | 2 127 | 91 % mano |

> **Limitación de la validación:** son datos de **una sola persona**. Los umbrales de la mano se calibraron con la sesión de letras estáticas y con la de trazos 1, así que esos resultados son de entrenamiento. Solo la sesión de trazos 2 es, en parte, una prueba con datos nuevos. Falta validar con más personas.

### 8.2 Letras estáticas (sesión real, 23 fases)

Precisión por fotograma (en los últimos 2.5 s de cada fase):

| Fiabilidad | Letras |
|---|---|
| **Alta (≥ 96 %)** | A 99 · B 100 · D 96 · E 100 · G 97 · H 97 · I 100 · L 100 · M 100 · N 100 · O 100 · S 100 · T 100 · V 100 · W 100 · Y 100 · ESPACIO 100 |
| **Media (78-91 %)** | C 91 · P 88 · U 78 |
| **Baja (< 70 %)** | R 68 · F 39 |
| Mano relajada | no escribió **ninguna** letra |

**De extremo a extremo** (estabilizador + mano quieta + hold-to-type) se escribió `ABCDEGHILMNOPKSTUVWY␣` frente a lo esperado `ABCDEFGHILMNOPRSTUVWY␣`: **20 de los 22 símbolos esperados salieron correctos y en orden**. Faltaron la **F** (solo hubo ≈ 0.4 s de F limpia; después la mano se movió rápido y la cámara llegó a perderla) y la **R** (se sostuvo quieta 1.4 s, menos de los 1.5 s necesarios). Hubo **una K falsa** (ver sección 9).

### 8.3 Señas con movimiento

| Gesto | Sesión nueva (trazos 2) | Sesión anterior (trazos 1) |
|---|---|---|
| **Z** | 3 de 4 | sin datos (la mano salía del cuadro) |
| **?** | **4 de 4** | sin datos |
| **J** | 2 de 3 | 1 |
| **K** | 1 | 0 en su fase (hizo un solo barrido; ya no es el patrón) |
| **Ñ** | 1 | 1 |
| **Q** | 1 | 2 |
| **X** | 1 | 2 |
| **BORRAR** | **3 de 3** | 1 |
| **Movimiento libre** (sin hacer letras) | **0 falsos** | **0 falsos** |

Algunas detecciones aparecen en los primeros segundos de la fase siguiente (por ejemplo una K al inicio de la fase de la J): son del gesto anterior, que se seguía haciendo cuando sonó el aviso del cambio de fase, y no se cuentan como aciertos. En la sesión nueva, **Q y X se distinguieron correctamente** gracias a la dirección del índice; antes de usar ese criterio la X se leía como Q.

### 8.4 Rostro

Con el detector **nuevo** sobre la grabación real: **2 de 2 preguntas** detectadas (a los 8.4 s y 15.4 s, ≈ 1.5 s después de subir las cejas), **1 negación**, **2 asentimientos** (un solo asentimiento de 4 s, con el enfriamiento de 1.2 s) y **cero eventos falsos** al mirar abajo, parpadear, girar la cabeza o quedarse mirando de lado. **El detector anterior detectaba 0 de 2 preguntas.**

### 8.5 Invarianza a escala, posición y mano (2 890 fotogramas reales transformados)

| Transformación | Fotogramas con la misma clasificación |
|---|---|
| Escala 0.4×, 0.7×, 1.8×, 2.5× | **100 %** |
| Desplazada | **100 %** |
| Espejo (la otra mano) | **100 %** |
| Escala 0.5× + espejo + desplazada | **100 %** |
| Giro en el plano de 5°, 10°, 20° | 99.7 %, 99.7 %, 99.6 % |
| Ruido gaussiano del 1.5 % de la escala | 95.1 % |

### 8.6 Voz y rendimiento

| Prueba | Resultado |
|---|---|
| **38 comprobaciones de lógica** (escritor, política de voz, cola de voz, guiones bajos, Ñ) | todas correctas |
| `decir()` | regresa en **0.00 ms** |
| Tiempos por fotograma con la carga real (cámara + manos + rostro + dibujo), **sin voz** y **con voz continua** (8 frases) | mediana **47.0 ms en ambos casos**, p95 62 vs 63 ms, **ningún fotograma mayor que 2× la mediana** mientras habla |
| Costo de `dibujar_panel` | de 8.5 ms a **1.4 ms** (fondo estático en caché; idéntico píxel a píxel) |
| FPS finales | **27-29 FPS** (MediaPipe Hands ≈ 26 ms, Face Mesh ≈ 4 ms, el resto ≈ 7 ms) |
| Prueba en vivo de `main.py` desde otra carpeta | ventanas lado a lado, 3 teclas → 3 envíos al hardware, cierra limpio con la `X` |

---

## 9. Limitaciones conocidas

| Limitación | Detalle | Mitigación |
|---|---|---|
| **K con falsos positivos** | Un vaivén de una mano con dos dedos arriba es justo lo que define a la K. En 103 s de escritura estática hubo **una K falsa**: se sacudía la mano mientras se formaba la R. Un barrido de un solo trazo era **indistinguible** de acomodar la mano (se comparó ventana por ventana) | K exige un vaivén (≥ 2 cambios de sentido). Evitar agitar la mano al escribir U, R o P |
| **F, R, U, P y C poco fiables** | Los dedos cruzados o en círculo ocluyen puntos y MediaPipe confunde las puntas | Ver sección 8.2; candidatas a un clasificador entrenado |
| **Q y X casi iguales** | Misma mano (índice curvo, resto cerrado); solo la dirección del índice las separa | Regla por dirección del índice (Q ≥ 120°); baja confianza |
| **S y T casi iguales** | En 2D el pulgar queda sobre el nudillo del índice en ambas | Se separan por la distancia del pulgar (0.20), calibrada con una sola mano |
| **Vista 2D** | No se usa la coordenada `z` | Posible mejora |
| **Una sola persona** | Umbrales calibrados con una sola mano | Repetir las sesiones con más personas |
| **Trazos pequeños** | Si la mano sale del cuadro, la cámara la pierde | Dibujar del tamaño de la mano |
| **CH, LL y RR** | No incluidas | Roadmap |
| **Fuentes de OpenCV** | No dibujan acentos; la Ñ se compone a mano | Texto sin acentos en pantalla y Ñ propia |
| **Voz del sistema** | Requiere una voz SAPI5 en español | Si falta, el programa sigue sin voz |

---

## 10. Decisiones de diseño respaldadas por datos

| Decisión | Evidencia |
|---|---|
| **Umbral de cejas 1.07 y no 1.15** | Una subida real da +10-17 %; con 1.15 solo el 29-37 % de los fotogramas lo superaba |
| **Cejas medidas en la comisura, proyectadas y entre el alto de rostro** | Es la métrica con menos ruido (1.5 %) y la única que no se mueve con la cabeza (≤ 7 % frente a 20 %); la del párpado tiene 2.7× más ruido |
| **Pulgar "abierto" por distancia, no por rectitud** | El pulgar parpadea entre recto y doblado (la G caía en X en el 29 % de los fotogramas); la distancia es estable |
| **Mano relajada sin letra (B < 20°, ESPACIO ≥ 48°)** | Una mano abierta relajada mide ≈ 40° y con el umbral de 35° caía en B |
| **U: puntas casi juntas, R: puntas cruzadas** | En la U las puntas quedan a +0.07 y en la R a −0.11; la regla "puntas casi juntas = R" convertía las U en R |
| **Buffer de 45 fotogramas y ventana por gesto** | Una Z a ritmo natural dura 1.4-2 s; en 20 fotogramas no cabe |
| **Q y X separadas por la dirección del índice** | Q 135-177° frente a X 23-113° en dos sesiones distintas; la apertura del pulgar y la orientación de la palma confundían |
| **BORRAR como barrido de mano abierta** | El puño es S, T o A; borrar con un puño borraría letras al escribirlas |
| **Velocidad de "mano quieta" 1.2 manos/s** | Sostener una seña produce 0.5-1.1 de temblor natural; con 0.9 se rompía la retención |
| **Pausa tras un vaivén** | Sin ella, agitar seguido escribía una Q cada ≈ 1.1 s |
| **K como vaivén y no como barrido** | Los falsos positivos eran barridos únicos |

---

## 11. Referencia del código

### 11.1 Estructura del repositorio

```
Proyecto/
├── .vscode/
│   ├── launch.json       # Ejecución con F5 usando el Python de venv
│   └── settings.json     # Intérprete del proyecto y perfil de PowerShell
├── venv/                 # Entorno virtual (ignorado por Git)
├── .gitignore
├── main.py               # Código fuente del sistema (10 secciones)
├── README.md             # Resumen e inicio rápido
└── DOCUMENTACION.md      # Este documento
```

### 11.2 `main.py` por secciones

| Sección | Elementos |
|---|---|
| **1. Configuración** | constantes, `GUIA` del alfabeto, estados faciales |
| **2. Texto y voz** | `limpiar_texto`, `nombre_hablado`, `ultima_palabra`, `_a_ascii` |
| **3. Geometría** | `dist`, `angulo`, `angulo_entre`, `a_pixeles`, `restar`, `unitario` |
| **4. Manos (rasgos)** | `escala_mano`, `RasgosMano`, `analizar_mano`, `angulo_dedos` |
| **5. Clasificación** | `clasificar_pose`, `decidir_q_o_x`, `Trayectoria`, `simplificar_rdp`, `segmentos`, `_trazo_j`, `_trazo_z`, `_trazo_interrogacion`, `_vaiven`, `_trazo_borrar`, `contar_reversos`, `pico_velocidad`, `ReconocedorDinamico` |
| **6. Rostro** | `calcular_rasgos_faciales`, `DetectorOscilacion`, `AnalizadorFacial` |
| **7. Escritura** | `Estabilizador`, `Escritor`, `ProgresoSuave` |
| **8. Voz y hardware** | `VozAsincrona`, `PoliticaVoz`, `_elegir_voz`, `enviar_a_hardware` |
| **9. Interfaz** | `EstadoUI`, `poner_texto` (con Ñ), `dibujar_camara`, `dibujar_panel` (fondo en caché), `medidor_cejas`, `dibujar_estela`, `colocar_ventanas` |
| **10. Principal** | `detectar_mano`, `detectar_rostro`, `main` |

### 11.3 Referencias

- SEP y Aprende.mx. *Abecedario en Lengua de Señas Mexicana* (lámina). https://nuevaescuelamexicana.sep.gob.mx/contenido/recurso/abecedario-en-lengua-de-senas-mexicana/
- Zhang, F. et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. arXiv:2006.10214.
- Kartynnik, Y. et al. (2019). *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs* (MediaPipe Face Mesh). arXiv:1907.06724.
- Ramer, U. (1972). *An iterative procedure for the polygonal approximation of plane curves*. Computer Graphics and Image Processing, 1(3).
- Hochreiter, S. y Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8).
- Documentación de MediaPipe Solutions: https://ai.google.dev/edge/mediapipe/solutions/guide
- Documentación de OpenCV: https://docs.opencv.org/4.x/
- Documentación de pyttsx3: https://pyttsx3.readthedocs.io/
