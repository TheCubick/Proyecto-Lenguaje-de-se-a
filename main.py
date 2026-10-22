"""Sistema Inteligente de Traducción de Lengua de Señas Mexicana (LSM) — Manos + Rostro + Voz.

Canal manual    (MediaPipe Hands):     alfabeto LSM completo.
                                       · Estáticas:  A B C D E F G H I L M N O P R S T U V W Y + ESPACIO
                                       · Con movimiento: J K Ñ Q X Z + BORRAR + "?"  (buffer de trayectoria)
Canal no manual (MediaPipe Face Mesh): cejas arriba = "?", negar con la cabeza = borrar,
                                       asentir = confirmar palabra (la lee en voz alta).
Voz             (pyttsx3, hilo propio): dice cada letra al registrarla y lee la palabra al confirmar
                                       o tras un tiempo de inactividad. Nunca bloquea la cámara.

Escritura "hold-to-type": una seña estática se registra al sostenerla quieta 1.5 s; las señas con
movimiento se registran al completar el trazo.
Teclas: Espacio, Backspace, C (limpiar), D (diagnóstico), R (recalibrar rostro), ESC (salir).

Estructura del archivo:
    1. Configuración              6. Canal no manual (rostro)
    2. Utilidades de texto/voz    7. Escritura (estabilizador + hold-to-type)
    3. Utilidades geométricas     8. Voz asíncrona y capa de hardware
    4. Canal manual (rasgos)      9. Interfaz gráfica
    5. Clasificación de señas    10. Programa principal
       (estáticas y con movimiento)
"""
import math
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
VENTANA_CAMARA = "Camara IA - Deteccion LSM"
VENTANA_PANEL = "Panel Traductor LSM"
ANCHO_PANEL = 640
ALTO_PANEL = 740
MARGEN_VENTANAS = 30

# --- Escritura ---
SEGUNDOS_PARA_ESCRIBIR = 1.5   # Tiempo que hay que sostener una seña estática QUIETA para registrarla
FRAMES_ESTABLES = 4            # Frames seguidos con la misma seña antes de aceptarla (anti-parpadeo)
V_QUIETO = 1.2                 # Velocidad de la palma (escalas/s) por encima de la cual la mano "se mueve".
                               # Sostener una seña produce 0.5-1.1 de temblor natural; un trazo, más de 2.
DURACION_DESTELLO = 1.2        # Segundos que se muestra en grande una seña con movimiento recién registrada

# --- Mano: umbrales geométricos (todos adimensionales => invariantes a la escala) ---
# r = dist(punta, muñeca) / dist(nudillo, muñeca)
R_EXTENDIDO = 1.45        # r mayor => dedo estirado (además de estar recto)
ANG_RECTO = 150           # Ángulo mínimo (grados) en la articulación media para considerarlo recto
R_CERRADO = 1.05          # r menor => punta metida en la palma (puño)
FACTOR_ANCHO_PALMA = 1.55 # Largo de palma ≈ 1.55 × ancho de nudillos (para la escala robusta)
T_PULGAR_CRUZADO = 0.20   # Proyección del pulgar sobre el eje índice->meñique (cruza la palma)
PULGAR_FUERA = 0.65       # Pulgar recto y lejos del nudillo del índice (mano abierta de ESPACIO)
PULGAR_ABIERTO = 0.48     # Pulgar apartado de la mano (A, L, G, Y...): se mide por DISTANCIA, no por rectitud,
                          # porque el pulgar parpadea entre "recto" y "doblado" y la distancia es estable
ANG_U_MAX = 14            # U: índice y medio casi paralelos
ANG_V_MIN = 25            # V: índice y medio abiertos
ANG_B_MAX = 20            # B: 4 dedos juntos (apertura índice-meñique menor que esto)
ANG_ESPACIO_MIN = 48      # ESPACIO: dedos bien separados. Entre ambos queda la "mano relajada" (sin letra)
O_CONTACTO = 0.20         # dist(punta pulgar, punta índice) / escala para la O
O_T_MAX = 0.30            # En la O el pulgar se queda del lado del índice
L_ANG_MIN, L_ANG_MAX = 50, 130  # Ángulo pulgar-índice de la L y de la G ("pistola")
ORIENT_ARRIBA = 55        # Orientación de la mano: 0° = dedos arriba, 90° = de lado, 180° = abajo
ORIENT_LADO = (65, 115)
ORIENT_ABAJO = 125
F_CONTACTO = 0.35         # F: la punta del índice toca la del pulgar
ST_DISTANCIA = 0.20       # Puño con el pulgar sobre el nudillo del índice: más cerca = S, más lejos = T
C_GAP = (0.20, 1.30)      # C: hueco entre la punta del pulgar y la del índice
T_C_MAX = 0.10            # C: el pulgar queda por fuera de la mano (en la E cruza la palma)
SEP_R_MAX = 0.0           # R: las puntas de índice y medio están cruzadas (separación con signo < 0)
P_CRUCE = 0.18            # P/K: cruzados en "X" (tijera) al menos esto, en escalas
INDICE_CURVO_R = 1.18     # Q/X: el índice sale claramente curvo (su razón r) con el resto de la mano cerrada
Q_DIR_INDICE_MIN = 120    # Q vs X: en la Q el índice cuelga (medido: Q 135-177°, X 23-113°, contados desde "arriba")

# --- Señas con movimiento (buffer de trayectoria) ---
TRAYECTORIA_FRAMES = 45   # Frames recordados (≈ 1.5 s a 29 FPS). Con datos reales, una Z a ritmo natural dura 1.4-2 s:
                          # 20 frames (0.7 s) no alcanzan para un trazo de tres movimientos. Cada gesto usa su propia
                          # ventana (GESTOS_DINAMICOS): los vaivenes y el barrido de BORRAR miran solo los últimos 15-24 frames
DIN_MIN_MUESTRAS = 10     # Muestras mínimas para evaluar un trazo
REARME_VEL = 0.8          # Tras un gesto de vaivén, la palma debe quedar por debajo de esta velocidad (escalas/s)...
REARME_SEGUNDOS = 0.3     # ... durante este tiempo antes de aceptar otro (si no, agitar seguido escribiría Q Q Q)
DIN_FRACCION_POSE = 0.60  # Fracción de la ventana en que debe verse la configuración de la mano
DIN_ENFRIAMIENTO = 1.0    # Segundos sin evaluar trazos tras reconocer uno (evita duplicados)
DIN_PERDIDOS_MAX = 5      # Frames sin mano antes de vaciar el buffer
VEL_SUAVIZADO = 0.25      # Suavizado (EMA) de la velocidad de la mano: ignora picos de 1-2 frames
J_DESCENSO_MIN = 0.55     # J: cuánto baja el meñique (en escalas)
J_GANCHO_MIN = 0.25       # J: cuánto se curva hacia un lado después de bajar
Z_BARRA_MIN = 0.40        # Z: largo mínimo de las dos barras
Z_DIAGONAL_MIN = 0.45     # Z: largo mínimo de la diagonal
INTERROG_ALTO_MIN = 0.80  # "?": altura mínima del trazo
VAIVEN_REVERSOS = 2       # Ñ, Q, X: cambios de sentido necesarios (2 = ida, vuelta e ida)
VAIVEN_AMP = {"Ñ": 0.10, "QX": 0.07, "K": 0.15}  # Amplitud mínima de cada oscilación (en escalas), medida en vaivenes reales
BORRAR_RECORRIDO_MIN = 1.10  # BORRAR: barrido a la izquierda, recorrido mínimo (en escalas)
BORRAR_VEL_MIN = 3.0         # BORRAR: velocidad pico mínima (escalas/s); evita borrar al reacomodar la mano

# --- Rostro ---
FRAMES_CALIBRACION = 45   # Frames con cara neutra para medir la línea base de las cejas
CALIB_PERCENTIL = 35      # Percentil de la calibración (ignora frames con cejas ya levantadas)
CEJAS_UMBRAL = 1.07       # Cejas "arriba" si su altura supera en 7 % la línea base. Medido con grabaciones reales:
                          # una subida de ceja da +10-17 %; parpadear, mirar abajo y girar o asentir la cabeza, < 7 %
CEJAS_UMBRAL_SALIDA = 1.04  # Histéresis: deben bajar de esto para volver a neutro
CEJAS_SEGUNDOS = 1.5      # Tiempo con las cejas arriba para agregar "?"
CEJAS_TOLERANCIA = 0.30   # Caídas más cortas que esto (parpadeo, ruido) no reinician el conteo
CEJAS_REARME = 0.30       # Tiempo con las cejas abajo para poder volver a preguntar
CEJAS_ADAPTACION = 0.005  # Qué tan rápido se reajusta la línea base mientras la cara está neutra
CEJAS_ZONA_ADAPT = 0.03   # Solo se reajusta si la razón está a menos de 3 % de la línea base
CEJAS_PAUSA_TRAS_CABEZA = 0.8  # Segundos sin evaluar cejas tras negar o asentir
NEG_AMPLITUD = 0.08       # Desviación mínima del giro (yaw) para contar un movimiento lateral
NEG_ENTRADAS_MIN = 3      # izquierda-derecha-izquierda (o al revés)
AFIRM_AMPLITUD = 0.035    # Desviación mínima de la inclinación (pitch) para contar un movimiento vertical
AFIRM_ENTRADAS_MIN = 2    # abajo-arriba o abajo-abajo
OSC_VENTANA = 1.4         # Segundos en los que debe ocurrir el movimiento de cabeza
OSC_TAU_CENTRO = 2.5      # Constante de tiempo (s) de la posición neutra de la cabeza
ENFRIAMIENTO_CABEZA = 1.2 # Segundos sin detectar otro gesto de cabeza tras uno reconocido
DURACION_AVISO = 1.2      # Segundos que se muestra en pantalla el último gesto facial
ACCION_NEGACION = "LETRA" # Qué borra "No" con la cabeza: "LETRA", "PALABRA" o "TODO"
PROCESAR_ROSTRO_CADA = 1  # 1 = cada frame; 2 = frames alternos (si la PC va lenta)

# --- Voz ---
VOZ_HABILITADA = True
VOZ_VELOCIDAD = 170       # Palabras por minuto
VOZ_VOLUMEN = 1.0
VOZ_INACTIVIDAD = 4.0     # Segundos sin nuevas letras para leer en voz alta la palabra pendiente
VOZ_MAX_LETRAS_EN_COLA = 2

# --- Colores (BGR) ---
FONDO = (28, 24, 22)
TARJETA = (48, 42, 38)
BORDE = (80, 72, 66)
TEXTO = (235, 235, 235)
TEXTO_SUAVE = (150, 145, 140)
ACENTO = (255, 190, 60)     # Azul claro
EXITO = (110, 220, 80)      # Verde
ALERTA = (80, 90, 255)      # Rojo
AMBAR = (60, 190, 255)      # Ámbar

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
FUENTE_TITULO = cv2.FONT_HERSHEY_DUPLEX

# Alfabeto: (clave, descripción corta, movimiento). Sin acentos: OpenCV no los dibuja (la Ñ se compone a mano).
GUIA = [
    ("A", "Puno, pulgar estirado al lado", None),
    ("B", "4 dedos juntos, pulgar cruzado", None),
    ("C", "Mano curva en forma de C", None),
    ("D", "Indice arriba, otros cerrados", None),
    ("E", "Dedos doblados sobre el pulgar", None),
    ("F", "Indice-pulgar, 3 dedos arriba", None),
    ("G", "Pistola: indice de lado", None),
    ("H", "Pistola: indice y medio de lado", None),
    ("I", "Solo el menique arriba", None),
    ("J", "I + gancho (baja y curva)", "Mueve el menique: baja y haz un gancho"),
    ("K", "P + agitar de lado a lado", "Agita la mano (grande) de lado a lado"),
    ("L", "Indice arriba y pulgar al lado", None),
    ("M", "3 dedos hacia abajo", None),
    ("N", "2 dedos hacia abajo", None),
    ("Ñ", "N + vaiven de lado a lado", "Agita la N de lado a lado"),
    ("O", "Circulo indice-pulgar", None),
    ("P", "Indice y medio cruzados en X", None),
    ("Q", "Pinza abajo + balanceo lateral", "Balancea la pinza de lado a lado"),
    ("R", "Indice y medio entrelazados", None),
    ("S", "Puno, pulgar sobre los dedos", None),
    ("T", "Puno, pulgar entre los dedos", None),
    ("U", "Indice y medio juntos", None),
    ("V", "Indice y medio en V", None),
    ("W", "Indice, medio y anular abiertos", None),
    ("X", "Indice de lado + vaiven", "Mueve el indice de lado a lado"),
    ("Y", "Pulgar y menique estirados", None),
    ("Z", "D + trazo de Z en el aire", "Dibuja una Z en el aire con el indice"),
    ("ESPACIO", "Mano abierta, 5 dedos", None),
    ("BORRAR", "Mano abierta barre a la izq.", "Barre la mano abierta a la izquierda"),
    ("?", "Indice dibuja ? en el aire", "Dibuja un ? en el aire con el indice"),
]
DESCRIPCIONES = {clave: (desc, mov) for clave, desc, mov in GUIA}

# Pistas para las configuraciones que sirven de base a un trazo
PISTA_TRAZO = {"I": "o traza una J", "Y": "o traza una J", "D": "o traza Z / ?", "P": "o agita para K",
               "R": "o agita para K", "U": "o agita para K", "N": "o agita para Ñ", "Q": "balancea para Q",
               "X": "mueve para X", "B": "o barre a la izq.", "ESPACIO": "o barre a la izq."}

# Texto y color de cada estado facial
MODOS_FACIALES = {
    "SIN_ROSTRO": ("Sin rostro", TEXTO_SUAVE),
    "CALIBRANDO": ("Calibrando...", AMBAR),
    "NEUTRO": ("Neutro", TEXTO),
    "CEJAS": ("Pregunta ?", ACENTO),       # Cejas arriba, cargando
    "PREGUNTA": ("Pregunta ?", EXITO),     # "?" agregado
    "NEGACION": ("Negacion X", ALERTA),
    "AFIRMACION": ("Afirmacion OK", EXITO),
}

# Gesto facial -> acción del escritor
ACCIONES_FACIALES = {"PREGUNTA": "PREGUNTA", "NEGACION": "NEGACION", "AFIRMACION": "CONFIRMAR"}


# =============================================================================
# 2. UTILIDADES DE TEXTO Y VOZ
# =============================================================================
# Nombre en español de cada letra: así la voz dice "jota" y no un sonido suelto
NOMBRES_VOZ = {
    "A": "a", "B": "be", "C": "ce", "D": "de", "E": "e", "F": "efe", "G": "ge", "H": "hache", "I": "i",
    "J": "jota", "K": "ka", "L": "ele", "M": "eme", "N": "ene", "Ñ": "eñe", "O": "o", "P": "pe", "Q": "cu",
    "R": "erre", "S": "ese", "T": "te", "U": "u", "V": "uve", "W": "doble ve", "X": "equis", "Y": "ye",
    "Z": "zeta",
}


def limpiar_texto(texto):
    """Reemplaza cualquier guion bajo por un espacio. Se aplica a TODO lo que se dibuja, se habla o se envía."""
    return str(texto).replace("_", " ")


def nombre_hablado(letra):
    return NOMBRES_VOZ.get(str(letra).upper(), str(letra).lower())


def ultima_palabra(texto):
    """Última palabra del texto (sin espacios ni guiones bajos alrededor); '' si no hay."""
    partes = limpiar_texto(texto).split()
    return partes[-1] if partes else ""


def _a_ascii(texto):
    """Quita acentos y signos que las fuentes de OpenCV no dibujan; conserva Ñ/ñ (se componen aparte)."""
    salida = []
    for ch in texto.replace("¿", "?").replace("¡", "!"):
        if ch in "Ññ":
            salida.append(ch)
            continue
        base = "".join(c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn")
        salida.append(base if base.isascii() else "?")
    return "".join(salida)


_avisos_mostrados = set()


def avisar_una_vez(mensaje):
    """Imprime un aviso solo la primera vez (evita llenar la consola en cada frame)."""
    if mensaje not in _avisos_mostrados:
        _avisos_mostrados.add(mensaje)
        print(f"[AVISO] {mensaje}")


# =============================================================================
# 3. UTILIDADES GEOMÉTRICAS
# =============================================================================
def dist(a, b):
    """Distancia euclidiana 2D."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def punto_medio(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def restar(a, b):
    return (a[0] - b[0], a[1] - b[1])


def unitario(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / n, v[1] / n)


def angulo(a, b, c):
    """Ángulo en grados en el vértice b (180 = recto)."""
    v1, v2 = restar(a, b), restar(c, b)
    n = math.hypot(*v1) * math.hypot(*v2)
    if n == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


def angulo_entre(v1, v2):
    """Ángulo en grados entre dos vectores."""
    return angulo((v1[0], v1[1]), (0, 0), (v2[0], v2[1]))


def a_pixeles(landmarks, ancho, alto, indices=None):
    """Normalizado [0,1] -> píxeles, para que las distancias no se deformen por el aspecto de la cámara.

    Con `indices` devuelve un dict {índice: (x, y)} solo con esos puntos (más rápido para Face Mesh).
    """
    if indices is None:
        return [(lm.x * ancho, lm.y * alto) for lm in landmarks]
    return {i: (landmarks[i].x * ancho, landmarks[i].y * alto) for i in indices}


# =============================================================================
# 4. CANAL MANUAL — rasgos geométricos de la mano
# =============================================================================
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
PINKY_MCP, PINKY_TIP = 17, 20
# (MCP, PIP, DIP, TIP) de índice, medio, anular y meñique
DEDOS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]

EXT, DOBLADO, CERRADO = "EXT", "DOB", "CER"


def escala_mano(p):
    """Tamaño de referencia de la mano, robusto a la rotación.

    Solo usa puntos rígidos de la palma (no dedos), así no cambia con la seña. Se toma el máximo
    entre el largo de la palma y el ancho de nudillos (escalado) porque, al rotar la mano, uno de
    los dos se acorta en la proyección 2D pero el otro no.
    """
    return max(dist(p[WRIST], p[MIDDLE_MCP]), FACTOR_ANCHO_PALMA * dist(p[INDEX_MCP], p[PINKY_MCP])) or 1.0


@dataclass
class RasgosMano:
    """Resumen adimensional de la mano (invariante a escala, posición y mano izquierda/derecha)."""
    p: list                  # 21 puntos en píxeles
    escala: float
    centro: tuple            # centro de la palma
    estados: list            # [índice, medio, anular, meñique] -> EXT / DOB / CER
    ratios: list
    t_pulgar: float          # proyección del pulgar sobre el eje índice->meñique (<0 fuera, >0 cruza)
    pulgar_fuera: bool       # recto y bien separado (mano abierta)
    pulgar_abierto: bool     # apartado de la mano (A, L, G, Y...)
    pulgar_cruzado: bool
    d_pulgar_indice: float   # punta del pulgar al nudillo del índice
    d_pt: list               # punta del pulgar a la punta de cada dedo
    orient: float            # 0° dedos arriba · 90° de lado · 180° abajo
    dir_indice: float        # hacia dónde apunta el dedo índice (nudillo -> punta): 0° arriba · 90° de lado · 180° abajo
    ang_im: float            # apertura índice-medio
    ang_abertura: float      # apertura índice-meñique
    sep_puntas: float        # separación CON SIGNO de las puntas índice/medio (<0 = cruzadas)
    sep_base: float          # separación de los nudillos índice/medio
    ang_pulgar_indice: float


def angulo_dedos(p, mcp_a, tip_a, mcp_b, tip_b):
    """Ángulo de apertura entre dos dedos (vectores nudillo -> punta)."""
    return angulo_entre(restar(p[tip_a], p[mcp_a]), restar(p[tip_b], p[mcp_b]))


def analizar_mano(p):
    """Calcula los rasgos geométricos de la mano a partir de los 21 puntos en píxeles."""
    escala = escala_mano(p)

    # Estado de cada dedo combinando una proporción de distancias y un ángulo articular
    estados, ratios = [], []
    for mcp, pip, _, tip in DEDOS:
        r = dist(p[tip], p[WRIST]) / (dist(p[mcp], p[WRIST]) or 1.0)
        recto = angulo(p[mcp], p[pip], p[tip]) > ANG_RECTO
        if recto and r > R_EXTENDIDO:
            estados.append(EXT)
        elif r < R_CERRADO:
            estados.append(CERRADO)
        else:
            estados.append(DOBLADO)
        ratios.append(r)

    # Pulgar: proyección sobre el eje nudillo índice -> nudillo meñique (t < 0 fuera, t > 0 cruza la palma)
    eje = restar(p[PINKY_MCP], p[INDEX_MCP])
    rel = restar(p[THUMB_TIP], p[INDEX_MCP])
    t_pulgar = (rel[0] * eje[0] + rel[1] * eje[1]) / ((eje[0] ** 2 + eje[1] ** 2) or 1.0)
    pulgar_recto = angulo(p[THUMB_MCP], p[THUMB_IP], p[THUMB_TIP]) > 140
    d_pulgar_indice = dist(p[THUMB_TIP], p[INDEX_MCP]) / escala

    # Orientación de la mano: ángulo entre el eje de la palma (muñeca -> nudillo medio) y "arriba" de la imagen
    palma = restar(p[MIDDLE_MCP], p[WRIST])
    orient = math.degrees(math.acos(max(-1.0, min(1.0, -palma[1] / (math.hypot(*palma) or 1.0)))))

    # Cruce de índice y medio: separación de las puntas medida a lo largo del eje de los nudillos
    w = unitario(eje)
    sep_puntas = (restar(p[MIDDLE_TIP], p[INDEX_TIP])[0] * w[0] + restar(p[MIDDLE_TIP], p[INDEX_TIP])[1] * w[1]) / escala
    sep_base = (restar(p[MIDDLE_MCP], p[INDEX_MCP])[0] * w[0] + restar(p[MIDDLE_MCP], p[INDEX_MCP])[1] * w[1]) / escala

    centro = (sum(p[i][0] for i in (0, 5, 9, 13, 17)) / 5, sum(p[i][1] for i in (0, 5, 9, 13, 17)) / 5)
    return RasgosMano(
        p=p, escala=escala, centro=centro, estados=estados, ratios=ratios, t_pulgar=t_pulgar,
        pulgar_fuera=pulgar_recto and d_pulgar_indice > PULGAR_FUERA and t_pulgar < 0,
        pulgar_abierto=d_pulgar_indice >= PULGAR_ABIERTO and t_pulgar < 0,
        pulgar_cruzado=t_pulgar > T_PULGAR_CRUZADO,
        d_pulgar_indice=d_pulgar_indice,
        d_pt=[dist(p[THUMB_TIP], p[tip]) / escala for _, _, _, tip in DEDOS],
        orient=orient,
        dir_indice=math.degrees(math.acos(max(-1.0, min(1.0, -(p[INDEX_TIP][1] - p[INDEX_MCP][1]) /
                                                       (dist(p[INDEX_TIP], p[INDEX_MCP]) or 1.0))))),
        ang_im=angulo_dedos(p, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP),
        ang_abertura=angulo_dedos(p, INDEX_MCP, INDEX_TIP, PINKY_MCP, PINKY_TIP),
        sep_puntas=sep_puntas, sep_base=sep_base,
        ang_pulgar_indice=angulo_entre(restar(p[THUMB_TIP], p[THUMB_MCP]), restar(p[INDEX_TIP], p[INDEX_MCP])),
    )


# =============================================================================
# 5. CLASIFICACIÓN DE SEÑAS
# =============================================================================
# Letras que se escriben sosteniendo la mano quieta
LETRAS_TIPEABLES = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "O", "P", "R", "S", "T",
                    "U", "V", "W", "Y", "ESPACIO"}
# Configuraciones que no se muestran ni escriben pero sirven de base a un trazo (mano abierta -> BORRAR)
POSES_SILENCIOSAS = {"ABIERTA"}


def _q_o_x(r):
    """Índice curvo o en pinza con el resto de la mano cerrada: Q (el índice cuelga hacia abajo) o X (el índice
    apunta de lado o arriba). Ambas se escriben con un vaivén."""
    return decidir_q_o_x(r.dir_indice)


def decidir_q_o_x(dir_indice):
    """Q si el dedo índice apunta hacia abajo; si no, X."""
    return "Q" if dir_indice >= Q_DIR_INDICE_MIN else "X"


def clasificar_pose(r):
    """Devuelve la configuración de la mano: una letra estática, "ESPACIO", "Q" o "X" (configuraciones que
    solo valen con movimiento), "ABIERTA" (mano abierta sin seña) o None si no hay certeza.
    Se basa en el alfabeto oficial de la LSM (SEP) y está calibrada con grabaciones reales.

    Orden de decisión: número de dedos estirados -> orientación de la mano -> detalles del pulgar.
    """
    est = r.estados
    indice, medio, anular, menique = (e == EXT for e in est)
    n_ext = sum((indice, medio, anular, menique))
    arriba = r.orient <= ORIENT_ARRIBA
    lado = ORIENT_LADO[0] <= r.orient <= ORIENT_LADO[1]
    abajo = r.orient >= ORIENT_ABAJO

    # --- Cuatro dedos estirados: F (índice casi recto), M (hacia abajo), ESPACIO, B o mano abierta ---
    if n_ext == 4:
        if r.d_pt[0] < F_CONTACTO:                  # el pulgar toca la punta del índice
            return "F"
        if abajo:                                   # 3 dedos hacia abajo con el meñique "dudoso"
            return "M"
        if r.pulgar_fuera and r.ang_abertura >= ANG_ESPACIO_MIN:
            return "ESPACIO"
        if not r.pulgar_fuera and r.ang_abertura < ANG_B_MAX:
            return "B"
        return "ABIERTA"                            # mano relajada: no escribe, pero vale para barrer (BORRAR)

    # --- Tres dedos estirados: F, W (arriba) o M (abajo) ---
    if n_ext == 3:
        if medio and anular and menique:            # F: el índice toca el pulgar
            return "F" if r.d_pt[0] < F_CONTACTO else None
        if indice and medio and anular:
            if abajo:
                return "M"
            if arriba:
                return "W"
        return None

    # --- Dos dedos (índice y medio): N, H, P, R, V, U según orientación y cruce ---
    if n_ext == 2 and indice and medio:
        if abajo:
            return "N"
        if lado:
            return "H"
        if not arriba:
            return None
        if r.pulgar_abierto:                        # P con el pulgar abierto (como se mide en la práctica);
            return "P"                              # la V y la U llevan el pulgar doblado sobre la palma
        if r.sep_puntas <= -P_CRUCE:                # cruzados en "X" (tijera de la lámina; con barrido es K)
            return "P"
        if r.sep_puntas < SEP_R_MAX:                # puntas entrelazadas
            return "R"
        if r.ang_im >= ANG_V_MIN:
            return "V"
        if r.ang_im < ANG_U_MAX:
            return "U"
        return None

    # --- Un solo dedo ---
    if n_ext == 1:
        if indice:
            if r.pulgar_abierto:
                if r.ang_pulgar_indice < L_ANG_MIN:  # pinza (índice y pulgar casi juntos): Q o X (con movimiento)
                    return _q_o_x(r)
                if L_ANG_MIN <= r.ang_pulgar_indice <= L_ANG_MAX:   # L (arriba), G (de lado) o Q (hacia abajo)
                    return "L" if arriba else ("G" if lado else ("Q" if abajo else None))
                return None
            if arriba:
                return "D"
            return "X" if (lado or abajo) else None  # índice de lado: X (con movimiento)
        if menique:
            return "Y" if r.pulgar_abierto else "I"
        return None

    # --- Ningún dedo estirado: O, A, S, T, E, C ---
    if n_ext == 0:
        hueco = r.d_pt[0]                           # punta del pulgar a punta del índice
        if hueco < O_CONTACTO and est[0] != CERRADO and r.t_pulgar < O_T_MAX:
            return "O"
        if est[0] == DOBLADO and est[1:] == [CERRADO] * 3 and r.ratios[0] >= INDICE_CURVO_R                 and not (arriba and not r.pulgar_abierto):
            return _q_o_x(r)                        # índice en gancho con el resto cerrado: Q o X
        if est.count(CERRADO) == 4:                 # puño: depende de dónde queda el pulgar
            if r.pulgar_abierto:
                return "A"
            return "S" if r.d_pulgar_indice < ST_DISTANCIA else "T"
        if est.count(DOBLADO) >= 2:                 # dedos curvos: E (pulgar cruzado) o C (pulgar por fuera)
            if r.t_pulgar >= T_PULGAR_CRUZADO:
                return "E"
            if C_GAP[0] <= hueco <= C_GAP[1] and r.t_pulgar < T_C_MAX:
                return "C"
        return None

    return None


# --- Trayectoria de la mano -------------------------------------------------
class Trayectoria:
    """Buffer circular (deque) con las últimas posiciones de la muñeca / los dedos.

    Cada muestra guarda: tiempo, posiciones de los puntos rastreados, escala de la mano y la
    configuración vista en ese frame. También calcula la velocidad de la palma (EMA).
    """

    def __init__(self, n=TRAYECTORIA_FRAMES):
        self.muestras = deque(maxlen=n)
        self.perdidos = 0
        self.vel = 0.0                 # escalas por segundo

    def agregar(self, t, r, pose):
        p = r.p
        puntos = {"muneca": p[WRIST], "indice": p[INDEX_TIP], "medio": p[MIDDLE_TIP],
                  "menique": p[PINKY_TIP], "centro": r.centro,
                  "puntas": punto_medio(p[INDEX_TIP], p[MIDDLE_TIP])}   # no se mueve al separar los dedos
        if self.muestras:
            t0, puntos0 = self.muestras[-1][:2]
            if t - t0 > 1e-3:
                v = dist(puntos["centro"], puntos0["centro"]) / (r.escala or 1.0) / (t - t0)
                self.vel += (v - self.vel) * VEL_SUAVIZADO
        self.muestras.append((t, puntos, r.escala, pose, r.dir_indice))
        self.perdidos = 0

    def perder(self):
        """Un frame sin mano: tras unos cuantos seguidos se vacía el buffer."""
        self.perdidos += 1
        self.vel *= 0.5
        if self.perdidos > DIN_PERDIDOS_MAX:
            self.muestras.clear()

    def limpiar(self):
        self.muestras.clear()

    def _ultimas(self, n=None):
        """Las últimas `n` muestras (todas si n es None)."""
        muestras = list(self.muestras)
        return muestras[-n:] if n else muestras

    def serie(self, nombre, n=None):
        """Trayectoria de las últimas `n` muestras normalizada por el tamaño de la mano (y hacia abajo)."""
        ms = self._ultimas(n)
        escala = float(np.median([m[2] for m in ms])) or 1.0
        return np.array([m[1][nombre] for m in ms], dtype=float) / escala

    def serie_px(self, nombre):
        return [m[1][nombre] for m in self.muestras]

    def tiempos(self, n=None):
        return np.array([m[0] for m in self._ultimas(n)], dtype=float)

    def fraccion_pose(self, permitidas, n=None):
        ms = self._ultimas(n)
        return sum(m[3] in permitidas for m in ms) / max(len(ms), 1)


# --- Reconocedores de trazos --------------------------------------------------
def simplificar_rdp(pts, eps):
    """Ramer-Douglas-Peucker: índices de los vértices que describen el trazo con tolerancia `eps`."""
    mantener = {0, len(pts) - 1}
    pila = [(0, len(pts) - 1)]
    while pila:
        i, j = pila.pop()
        a, ab = pts[i], pts[j] - pts[i]
        largo = float(np.hypot(*ab))
        lejos, idx = 0.0, -1
        for k in range(i + 1, j):
            v = pts[k] - a
            d = abs(ab[0] * v[1] - ab[1] * v[0]) / largo if largo > 1e-9 else float(np.hypot(*v))
            if d > lejos:
                lejos, idx = d, k
        if lejos > eps:
            mantener.add(idx)
            pila += [(i, idx), (idx, j)]
    return sorted(mantener)


def segmentos(pts, eps=0.10, largo_min=0.12, fusion=35):
    """Trazo -> lista de (rumbo en grados, largo). Rumbo: 0 = derecha, 90 = arriba, 270 = abajo (y hacia arriba)."""
    idx = simplificar_rdp(pts, eps)
    segs = []
    for i, j in zip(idx[:-1], idx[1:]):
        v = pts[j] - pts[i]
        segs.append([(math.degrees(math.atan2(-v[1], v[0]))) % 360, float(np.hypot(*v)), v[0], v[1]])
    fusionado = []
    for s in segs:
        if fusionado:
            ant = fusionado[-1]
            dif = abs((s[0] - ant[0] + 180) % 360 - 180)
            if s[1] < largo_min or dif <= fusion:   # tramo diminuto o casi en la misma dirección: se une
                ant[2] += s[2]
                ant[3] += s[3]
                ant[0] = math.degrees(math.atan2(-ant[3], ant[2])) % 360
                ant[1] = float(np.hypot(ant[2], ant[3]))
                continue
        fusionado.append(s)
    return [(s[0], s[1]) for s in fusionado]


def en_arco(rumbo, centro, semiancho):
    return abs((rumbo - centro + 180) % 360 - 180) <= semiancho


def _simetrico(fn):
    """Acepta el trazo y su imagen en espejo: sirve para zurdos y para quien dibuja "como en el espejo"."""
    def g(p, ts):
        return fn(p, ts) or fn(p * np.array([-1.0, 1.0]), ts)
    return g


def _trazo_j(p, ts):
    """J: el meñique baja en vertical y se curva hacia un lado al final (gancho)."""
    n = len(p)
    k = int(np.argmax(p[:, 1]))                       # punto más bajo
    if k < 3 or n - 1 - k < 2:
        return False
    i0 = int(np.argmin(p[:k + 1, 1]))                 # donde empezó a bajar
    descenso = p[k, 1] - p[i0, 1]
    deriva = abs(p[k, 0] - p[i0, 0])
    gancho = p[k, 0] - p[k:, 0].min()                 # cuánto se fue a la izquierda después de bajar
    return descenso >= J_DESCENSO_MIN and deriva <= 0.6 * descenso and gancho >= J_GANCHO_MIN


def _trazo_z(p, ts):
    """Z: barra a la derecha, diagonal hacia abajo-izquierda y otra barra a la derecha."""
    segs = segmentos(p)
    if len(segs) < 3:
        return False
    (h1, l1), (h2, l2), (h3, l3) = segs[-3:]
    return (en_arco(h1, 0, 40) and l1 >= Z_BARRA_MIN and en_arco(h2, 225, 35) and l2 >= Z_DIAGONAL_MIN
            and en_arco(h3, 0, 40) and l3 >= Z_BARRA_MIN)


def _trazo_interrogacion(p, ts):
    """"?": arco que abulta a la derecha, regresa hacia el centro y termina en un tallo vertical."""
    n = len(p)
    if p[:, 1].max() - p[:, 1].min() < INTERROG_ALTO_MIN:
        return False
    k = int(np.argmax(p[:, 0]))                       # punto más a la derecha (el abultamiento)
    if not (0.15 * n <= k <= 0.75 * n) or int(np.argmin(p[:, 1])) > 0.6 * n:
        return False
    if p[k, 0] - p[0, 0] < 0.20 or p[k, 0] - p[-1, 0] < 0.20 or p[-1, 1] - p[k, 1] < 0.40:
        return False
    dx_final, dy_final = p[-1, 0] - p[-4, 0], p[-1, 1] - p[-4, 1]   # el tallo termina casi vertical
    return dy_final >= 0.25 and abs(dx_final) <= 0.6 * dy_final


def pico_velocidad(p, ts, k=4):
    """Velocidad máxima (escalas/s) del trazo, medida en ventanas de `k` muestras."""
    if len(p) <= k:
        return 0.0
    return max(float(np.hypot(*(p[i + k] - p[i]))) / max(ts[i + k] - ts[i], 1e-3) for i in range(len(p) - k))


def contar_reversos(xs, amp):
    """Cambios de sentido de una señal 1D, con histéresis `amp` para ignorar el temblor."""
    ultimo, sentido, reversos = xs[0], 0, 0
    for x in xs[1:]:
        if sentido == 0:
            if x - ultimo > amp:
                sentido, ultimo = 1, x
            elif ultimo - x > amp:
                sentido, ultimo = -1, x
        elif sentido == 1:
            if x > ultimo:
                ultimo = x
            elif ultimo - x > amp:
                sentido, ultimo, reversos = -1, x, reversos + 1
        else:
            if x < ultimo:
                ultimo = x
            elif x - ultimo > amp:
                sentido, ultimo, reversos = 1, x, reversos + 1
    return reversos


def _vaiven(amp, horizontal=True):
    """Ñ, K, Q, X: oscilación de al menos `amp` con VAIVEN_REVERSOS cambios de sentido en x.
    Con `horizontal` el movimiento debe ser casi lateral; Q y X (balanceo del índice curvo) no lo exigen."""
    def f(p, ts):
        ancho = p[:, 0].max() - p[:, 0].min()
        alto = p[:, 1].max() - p[:, 1].min()
        if horizontal and alto > 0.8 * ancho:
            return False
        return ancho >= 1.8 * amp and contar_reversos(p[:, 0], amp) >= VAIVEN_REVERSOS
    return f


def _trazo_borrar(p, ts):
    """BORRAR: barrido rápido de la mano abierta hacia la izquierda."""
    dx, dy = p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1]
    if dx > -BORRAR_RECORRIDO_MIN or abs(dy) > 0.6 * abs(dx):
        return False
    pasos = np.diff(p[:, 0])
    total = np.abs(pasos).sum()
    if total <= 0 or -pasos[pasos < 0].sum() / total < 0.85:
        return False
    k = 4
    if len(p) <= k:
        return False
    return max((p[i, 0] - p[i + k, 0]) / max(ts[i + k] - ts[i], 1e-3) for i in range(len(p) - k)) >= BORRAR_VEL_MIN


# (gesto, configuraciones válidas, punto que se rastrea, reconocedor, fracción mínima con esa configuración, ventana)
# La ventana es cuántos de los últimos frames mira cada gesto: un trazo largo (Z, ?) necesita más que un vaivén.
GESTOS_DINAMICOS = [
    ("J", {"I", "Y"}, "menique", _simetrico(_trazo_j), DIN_FRACCION_POSE, 30),      # el pulgar parpadea entre I e Y
    ("Z", {"D"}, "indice", _simetrico(_trazo_z), DIN_FRACCION_POSE, 45),
    ("?", {"D"}, "indice", _simetrico(_trazo_interrogacion), DIN_FRACCION_POSE, 45),
    ("K", {"P", "R", "U"}, "puntas", _vaiven(VAIVEN_AMP["K"]), DIN_FRACCION_POSE, 45),   # índice y medio que se agitan
    ("Ñ", {"N", "H"}, "centro", _vaiven(VAIVEN_AMP["Ñ"]), DIN_FRACCION_POSE, 30),       # una N de lado se lee como H
    # Q y X son el mismo vaivén; con el índice curvo y la mano de lado la configuración baila (Q/X/F), por eso
    # se les pide menos: se decide Q o X con el promedio de la ventana
    ("QX", {"Q", "X"}, "indice", _vaiven(VAIVEN_AMP["QX"], horizontal=False), 0.45, 24),
    ("BORRAR", {"B", "ESPACIO", "ABIERTA"}, "centro", _trazo_borrar, DIN_FRACCION_POSE, 15),
]
GESTOS_OSCILANTES = {"K", "Ñ", "QX"}       # tras uno de estos hace falta una pausa antes de aceptar otro
PUNTO_DE_POSE = {}
for _g, _poses, _punto, _, _, _ in GESTOS_DINAMICOS:
    for _pose in _poses:
        PUNTO_DE_POSE.setdefault(_pose, _punto)


class ReconocedorDinamico:
    """Evalúa en cada frame si el buffer contiene un trazo completo con la configuración de mano adecuada."""

    def __init__(self):
        self.libre_desde = 0.0
        self.pausa_pendiente = False    # tras un vaivén hay que soltar la mano un momento
        self.t_quieto = None

    def _actualizar_pausa(self, tray, ahora):
        if not self.pausa_pendiente:
            return
        if tray.vel < REARME_VEL:
            self.t_quieto = ahora if self.t_quieto is None else self.t_quieto
            if ahora - self.t_quieto >= REARME_SEGUNDOS:
                self.pausa_pendiente = False
        else:
            self.t_quieto = None

    def evaluar(self, tray, pose, ahora):
        self._actualizar_pausa(tray, ahora)
        if ahora < self.libre_desde or len(tray.muestras) < DIN_MIN_MUESTRAS:
            return None
        for gesto, poses, punto, reconocer, fraccion, ventana in GESTOS_DINAMICOS:
            n = min(ventana, len(tray.muestras))
            if pose not in poses or n < DIN_MIN_MUESTRAS or tray.fraccion_pose(poses, n) < fraccion:
                continue
            if gesto in GESTOS_OSCILANTES and self.pausa_pendiente:
                continue
            if reconocer(tray.serie(punto, n), tray.tiempos(n)):
                if gesto == "QX":                   # Q y X son el mismo vaivén: se decide con el promedio de la ventana
                    ultimas = tray._ultimas(n)
                    gesto = decidir_q_o_x(float(np.mean([m[4] for m in ultimas])))
                    self.pausa_pendiente, self.t_quieto = True, None
                elif gesto in GESTOS_OSCILANTES:
                    self.pausa_pendiente, self.t_quieto = True, None
                self.libre_desde = ahora + DIN_ENFRIAMIENTO
                tray.limpiar()
                return gesto
        return None


# =============================================================================
# 6. CANAL NO MANUAL — rostro y movimientos de cabeza (MediaPipe Face Mesh)
# =============================================================================
# Índices de Face Mesh (468 puntos). "Der./izq." son del sujeto.
CEJA_SUP_DER = (70, 63, 105, 66, 107)       # Contorno superior de la ceja derecha
CEJA_SUP_IZQ = (300, 293, 334, 296, 336)    # Contorno superior de la ceja izquierda
CANTO_DER = (33, 133)                       # Comisuras del ojo derecho (óseas: no se mueven al parpadear)
CANTO_IZQ = (362, 263)
FRENTE, MENTON, NARIZ = 10, 152, 1
POMULO_DER, POMULO_IZQ = 234, 454           # Contorno lateral de la cara
INDICES_ROSTRO = (*CEJA_SUP_DER, *CEJA_SUP_IZQ, *CANTO_DER, *CANTO_IZQ, FRENTE, MENTON, NARIZ,
                  POMULO_DER, POMULO_IZQ)


def calcular_rasgos_faciales(f):
    """Rasgos faciales adimensionales a partir de un dict {índice: (x, y)}.

    - cejas:       altura media de las cejas sobre la línea de las comisuras de los ojos, MEDIDA A LO LARGO
                   DEL EJE DE LA CARA (mentón -> frente) y dividida por el alto total del rostro. Así no
                   depende de la distancia a la cámara, ni de inclinar o girar la cabeza (se verificó con
                   datos reales: con la cabeza en movimiento varía menos de 7 %, la medida anterior 20 %).
                   Se usa la comisura y no el párpado: el párpado sube y baja al parpadear o mirar abajo.
    - giro:        (yaw) posición horizontal de la nariz entre los pómulos, de -1 a 1.
    - inclinacion: (pitch) altura de la nariz respecto a la línea de los pómulos.
    Devuelve None si la geometría es degenerada (cara muy pequeña).
    """
    alto_cara = dist(f[FRENTE], f[MENTON])
    ancho_cara = dist(f[POMULO_DER], f[POMULO_IZQ])
    if alto_cara < 1 or ancho_cara < 1:
        return None
    eje = unitario(restar(f[FRENTE], f[MENTON]))

    def altura(ref, ids):
        return sum((f[i][0] - ref[0]) * eje[0] + (f[i][1] - ref[1]) * eje[1] for i in ids) / len(ids)

    ref_der = punto_medio(f[CANTO_DER[0]], f[CANTO_DER[1]])
    ref_izq = punto_medio(f[CANTO_IZQ[0]], f[CANTO_IZQ[1]])
    centro_pomulos = punto_medio(f[POMULO_DER], f[POMULO_IZQ])
    return {
        "cejas": (altura(ref_der, CEJA_SUP_DER) + altura(ref_izq, CEJA_SUP_IZQ)) / 2 / alto_cara,
        "giro": (dist(f[NARIZ], f[POMULO_DER]) - dist(f[NARIZ], f[POMULO_IZQ])) / ancho_cara,
        "inclinacion": (f[NARIZ][1] - centro_pomulos[1]) / ancho_cara,
    }


class DetectorOscilacion:
    """Detecta un movimiento de vaivén (negar o asentir) en una señal 1D.

    Mantiene una "posición neutra" que se adapta lentamente (EMA), así que mirar a un lado y
    quedarse ahí no cuenta. Cuenta las veces que la señal ENTRA en la zona positiva o negativa
    (con histéresis para ignorar el temblor) dentro de una ventana de tiempo.
    """

    def __init__(self, amplitud, entradas_min, requiere_ambos_lados):
        self.amplitud = amplitud
        self.entradas_min = entradas_min
        self.requiere_ambos_lados = requiere_ambos_lados
        self.reiniciar()

    def reiniciar(self):
        self.centro = None
        self.t_anterior = None
        self.zona = 0
        self.entradas = deque()       # (tiempo, lado) de cada entrada a una zona
        self.desviacion = 0.0

    def actualizar(self, t, valor):
        """Agrega una muestra. Devuelve True si se completó el movimiento."""
        if self.centro is None:
            self.centro = valor
        else:
            dt = max(0.0, t - self.t_anterior)
            self.centro += (valor - self.centro) * min(1.0, dt / OSC_TAU_CENTRO)
        self.t_anterior = t
        d = valor - self.centro
        self.desviacion = d

        # Máquina de zonas con histéresis: se entra con |d| > amplitud y se sale con |d| < amplitud/2
        if self.zona == 0:
            nueva = 1 if d > self.amplitud else (-1 if d < -self.amplitud else 0)
        elif self.zona * d < self.amplitud / 2:
            nueva = -self.zona if abs(d) > self.amplitud else 0
        else:
            nueva = self.zona
        if nueva != 0 and nueva != self.zona:
            self.entradas.append((t, nueva))
        self.zona = nueva

        while self.entradas and t - self.entradas[0][0] > OSC_VENTANA:
            self.entradas.popleft()

        lados = {lado for _, lado in self.entradas}
        if len(self.entradas) >= self.entradas_min and (not self.requiere_ambos_lados or len(lados) == 2):
            self.entradas.clear()
            return True
        return False


class AnalizadorFacial:
    """Interpreta los componentes no manuales: cejas (pregunta), negar y asentir con la cabeza."""

    def __init__(self):
        self.negacion = DetectorOscilacion(NEG_AMPLITUD, NEG_ENTRADAS_MIN, requiere_ambos_lados=True)
        self.afirmacion = DetectorOscilacion(AFIRM_AMPLITUD, AFIRM_ENTRADAS_MIN, requiere_ambos_lados=False)
        self.ultimo_evento, self.t_evento = None, -10.0
        self.enfriamiento_hasta = 0.0
        self.cejas_pausa_hasta = 0.0
        self.rasgos = None
        self.modo = "SIN_ROSTRO"
        self.recalibrar()

    def recalibrar(self):
        """Vuelve a medir la línea base de las cejas (tecla R)."""
        self.muestras_calibracion = []
        self.base_cejas = None
        self.ratio_cejas = 1.0
        self.pregunta_hecha = False
        self._reiniciar_cejas()

    def _reiniciar_cejas(self):
        self.cejas_inicio = None
        self.ultima_arriba = -10.0
        self.progreso_pregunta = 0.0

    @property
    def progreso_calibracion(self):
        return 1.0 if self.base_cejas is not None else len(self.muestras_calibracion) / FRAMES_CALIBRACION

    def _perder_rostro(self):
        self.rasgos = None
        self.negacion.reiniciar()
        self.afirmacion.reiniciar()
        self._reiniciar_cejas()

    def actualizar(self, puntos, t):
        """Procesa un frame. `puntos` es {índice: (x, y)} o None si no hay cara.

        Devuelve el gesto reconocido en este frame ("PREGUNTA", "NEGACION", "AFIRMACION") o None.
        """
        rasgos = calcular_rasgos_faciales(puntos) if puntos is not None else None
        if rasgos is None or rasgos["cejas"] <= 0:
            self._perder_rostro()
            self.modo = "SIN_ROSTRO"
            return None
        self.rasgos = rasgos

        # 1) Calibración: percentil bajo de la altura de cejas con cara neutra (ignora frames con cejas arriba)
        if self.base_cejas is None:
            self.muestras_calibracion.append(rasgos["cejas"])
            if len(self.muestras_calibracion) >= FRAMES_CALIBRACION:
                self.base_cejas = float(np.percentile(self.muestras_calibracion, CALIB_PERCENTIL))
            self.modo = "CALIBRANDO"
            return None

        evento = None

        # 2) Movimientos de cabeza (negar / asentir)
        negar = self.negacion.actualizar(t, rasgos["giro"])
        asentir = self.afirmacion.actualizar(t, rasgos["inclinacion"])
        if t >= self.enfriamiento_hasta and (negar or asentir):
            evento = "NEGACION" if negar else "AFIRMACION"
            self.enfriamiento_hasta = t + ENFRIAMIENTO_CABEZA
            self.negacion.entradas.clear()
            self.afirmacion.entradas.clear()
            self.cejas_pausa_hasta = t + CEJAS_PAUSA_TRAS_CABEZA   # negar/asentir no deben contar como pregunta
            self._reiniciar_cejas()

        # 3) Cejas (pregunta): razón respecto a la línea base, con histéresis y tolerancia a caídas breves
        self.ratio_cejas = rasgos["cejas"] / self.base_cejas
        if t >= self.cejas_pausa_hasta:
            if self.ratio_cejas >= CEJAS_UMBRAL:
                self.ultima_arriba = t
                if self.cejas_inicio is None:
                    self.cejas_inicio = t
            elif self.ratio_cejas >= CEJAS_UMBRAL_SALIDA and self.cejas_inicio is not None:
                self.ultima_arriba = t       # banda de histéresis: sigue contando
            if self.cejas_inicio is not None and t - self.ultima_arriba > CEJAS_TOLERANCIA:
                self._reiniciar_cejas()      # las cejas bajaron de verdad
            if self.cejas_inicio is not None:
                self.progreso_pregunta = min(1.0, (t - self.cejas_inicio) / CEJAS_SEGUNDOS)
                if self.progreso_pregunta >= 1.0 and not self.pregunta_hecha and evento is None:
                    evento = "PREGUNTA"
                    self.pregunta_hecha = True   # Una sola "?" por cada vez que se levantan las cejas
            elif self.pregunta_hecha and self.ratio_cejas < CEJAS_UMBRAL_SALIDA and t - self.ultima_arriba > CEJAS_REARME:
                self.pregunta_hecha = False
            # Reajuste lento de la línea base mientras la cara está neutra (luz, postura...)
            if self.cejas_inicio is None and abs(self.ratio_cejas - 1.0) < CEJAS_ZONA_ADAPT:
                self.base_cejas += (rasgos["cejas"] - self.base_cejas) * CEJAS_ADAPTACION

        # 4) Estado que se muestra en pantalla
        if evento:
            self.ultimo_evento, self.t_evento = evento, t
        if self.ultimo_evento and t - self.t_evento < DURACION_AVISO:
            self.modo = self.ultimo_evento
        elif self.cejas_inicio is not None and not self.pregunta_hecha:
            self.modo = "CEJAS"
        else:
            self.modo = "NEUTRO"
        return evento


# =============================================================================
# 7. ESCRITURA — estabilizador + hold-to-type
# =============================================================================
class Estabilizador:
    """Solo acepta una seña nueva cuando se repite N frames seguidos (filtro anti-parpadeo)."""

    def __init__(self, n):
        self.n = n
        self.estable = None
        self.candidato = None
        self.conteo = 0

    def actualizar(self, sena):
        if sena == self.estable:
            self.candidato, self.conteo = sena, 0
        elif sena == self.candidato:
            self.conteo += 1
            if self.conteo >= self.n:
                self.estable, self.conteo = sena, 0
        else:
            self.candidato, self.conteo = sena, 1
        return self.estable


class Escritor:
    """Mantiene el texto, el hold-to-type y la lista de eventos (letra, espacio, borrar, confirmar...).

    Una seña estática se registra al sostenerla `segundos` quieta. Para repetir la misma letra hay que
    cambiar de seña (o bajar la mano) y volver a hacerla; así se evitan letras duplicadas sin querer.
    Una seña con movimiento se registra al instante y "bloquea" la configuración de mano con la que se
    hizo (p. ej. la I tras dibujar una J) hasta que la mano cambie.
    """

    def __init__(self, segundos):
        self.segundos = segundos
        self.texto = ""
        self.sena = None
        self.inicio = 0.0
        self.registrada = False
        self.ultimo_registro = -10.0
        self.bloqueo = None
        self.destello = (None, -10.0)       # (letra con movimiento, instante) para mostrarla en grande
        self.eventos = []

    def sacar_eventos(self):
        eventos, self.eventos = self.eventos, []
        return eventos

    def ejecutar(self, accion, ahora):
        """Aplica una letra o una acción de edición al texto y emite el evento correspondiente."""
        if accion == "ESPACIO":
            self.texto += " "
            self.eventos.append(("espacio",))
        elif accion == "BORRAR":
            self.texto = self.texto[:-1]
            self.eventos.append(("borrar",))
        elif accion == "LIMPIAR":
            self.texto = ""
            self.eventos.append(("limpiar",))
        elif accion == "CONFIRMAR":            # Asentir: cierra la palabra con un espacio
            if self.texto.strip():
                if not self.texto.endswith(" "):
                    self.texto += " "
                self.eventos.append(("confirmar",))
        elif accion == "PREGUNTA":             # Cejas arriba o "?" en el aire: "?" al final de la palabra
            base = self.texto.rstrip()
            if base and not base.endswith("?"):
                self.texto = base + "?"
                self.eventos.append(("pregunta",))
        elif accion == "NEGACION":             # Negar con la cabeza
            if ACCION_NEGACION == "TODO":
                self.texto = ""
            elif ACCION_NEGACION == "PALABRA":
                base = self.texto.rstrip()
                self.texto = base[:base.rfind(" ") + 1] if " " in base else ""
            else:
                self.texto = self.texto[:-1]
            self.eventos.append(("negacion",))
        else:                                  # Letra
            self.texto += accion
            self.eventos.append(("letra", accion))
        self.ultimo_registro = ahora

    def registrar_dinamico(self, gesto, ahora, pose):
        """Registra una seña con movimiento (J, K, Ñ, Q, X, Z, BORRAR, ?) en cuanto se completa el trazo."""
        self.ejecutar({"?": "PREGUNTA"}.get(gesto, gesto), ahora)
        self.sena, self.inicio, self.registrada = gesto, ahora, True
        self.bloqueo = pose if pose in LETRAS_TIPEABLES else None
        self.destello = (gesto, ahora)

    def actualizar(self, sena, ahora):
        """Avanza el hold-to-type. Devuelve el progreso real de 0.0 a 1.0 de la seña actual."""
        if self.bloqueo is not None:
            if sena == self.bloqueo:           # sigue la misma mano con la que se hizo el trazo: no escribir
                return 1.0
            self.bloqueo = None
        if sena != self.sena:                  # Cambió la seña: el progreso se reinicia
            self.sena, self.inicio, self.registrada = sena, ahora, False
        if sena is None:
            return 0.0
        if self.registrada:
            return 1.0
        progreso = (ahora - self.inicio) / self.segundos
        if progreso >= 1.0:
            self.ejecutar(sena, ahora)
            self.registrada = True
            return 1.0
        return progreso


class ProgresoSuave:
    """Suaviza la barra al subir y la reinicia al instante si la seña cambia."""

    def __init__(self, suavizado=0.35):
        self.valor = 0.0
        self.suavizado = suavizado

    def actualizar(self, objetivo):
        if objetivo < self.valor - 1e-3:
            self.valor = objetivo
        else:
            self.valor += (objetivo - self.valor) * self.suavizado
        return self.valor


# =============================================================================
# 8. VOZ ASÍNCRONA Y CAPA DE HARDWARE
# =============================================================================
def _elegir_voz(motor):
    """Prefiere español de México; si no hay, cualquier voz en español; si no, None (voz por defecto)."""
    mejor, puntaje = None, 0
    for v in motor.getProperty("voices"):
        idiomas = [str(i).lower() for i in (v.languages or [])]
        ficha = f"{v.name} {v.id}".lower()
        if any("es-mx" in i or "es_mx" in i for i in idiomas) or "es-mx" in ficha or "mexico" in ficha:
            s = 3
        elif any("es" == i[:2] or "es" == i[1:3] for i in idiomas) or "spanish" in ficha or "es-es" in ficha:
            s = 2
        else:
            s = 0
        if s > puntaje:
            mejor, puntaje = v, s
    return mejor


class VozAsincrona:
    """Motor de voz (pyttsx3) en un hilo propio: hablar NUNCA bloquea el bucle de la cámara.

    El motor se crea DENTRO del hilo (en Windows, SAPI5 queda ligado al hilo que lo crea) y las frases
    esperan en una cola. Las palabras sustituyen a las letras pendientes; si se acumulan letras se
    descartan las más viejas, para que la voz nunca se quede atrasada respecto a lo que se ve.
    """

    def __init__(self, habilitada=True):
        self.estado = "desactivada"
        self._cola = deque()              # (texto, tipo)
        self._cond = threading.Condition()
        self._cerrando = False
        self._hilo = None
        if habilitada:
            self.estado = "iniciando"
            self._hilo = threading.Thread(target=self._bucle, name="voz-lsm", daemon=True)
            self._hilo.start()

    def decir(self, texto, tipo="letra"):
        """Encola una frase y regresa al instante. `tipo`: "letra" o "palabra"."""
        texto = limpiar_texto(texto).strip()
        if not texto or self.estado in ("desactivada", "no disponible"):
            return
        with self._cond:
            if tipo == "palabra":
                self._cola = deque(x for x in self._cola if x[1] != "letra")
            else:
                letras = [x for x in self._cola if x[1] == "letra"]
                while len(letras) >= VOZ_MAX_LETRAS_EN_COLA:
                    self._cola.remove(letras.pop(0))
            self._cola.append((texto, tipo))
            self._cond.notify()

    def cerrar(self):
        with self._cond:
            self._cerrando = True
            self._cola.clear()
            self._cond.notify()
        if self._hilo is not None:
            self._hilo.join(timeout=1.5)

    def _bucle(self):
        motor = None
        try:
            try:                                  # Windows: cada hilo debe inicializar COM
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass
            import pyttsx3
            motor = pyttsx3.init()
            voz = _elegir_voz(motor)
            if voz is not None:
                motor.setProperty("voice", voz.id)
            else:
                avisar_una_vez("No hay una voz en español instalada; se usará la voz por defecto de Windows.")
            motor.setProperty("rate", VOZ_VELOCIDAD)
            motor.setProperty("volume", 0.0)      # calentamiento en silencio: la primera frase real no se retrasa
            motor.say("a")
            motor.runAndWait()
            motor.setProperty("volume", VOZ_VOLUMEN)
            self.estado = "lista"
        except Exception as e:
            self.estado = "no disponible"
            avisar_una_vez(f"Voz no disponible ({type(e).__name__}: {e}). Instala con: pip install pyttsx3")
            return
        while True:
            with self._cond:
                while not self._cola and not self._cerrando:
                    self._cond.wait()
                if self._cerrando:
                    break
                texto, _ = self._cola.popleft()
            try:
                motor.say(texto)
                motor.runAndWait()
            except Exception as e:                # Un fallo al hablar no debe matar el hilo
                avisar_una_vez(f"Error al hablar ({type(e).__name__}: {e}).")
        try:
            motor.stop()
        except Exception:
            pass


class PoliticaVoz:
    """Decide QUÉ se dice y CUÁNDO:
      · al registrarse una letra (barra al 100 % o trazo completo)  -> dice el nombre de la letra
      · al confirmar la palabra (asentir)                           -> lee la palabra completa
      · tras `inactividad` segundos sin nuevas letras               -> lee la palabra pendiente
    """

    def __init__(self, voz, inactividad=VOZ_INACTIVIDAD):
        self.voz = voz
        self.inactividad = inactividad
        self.pendiente = False            # hay letras nuevas que aún no se han leído como palabra
        self.t_ultimo = 0.0

    def evento(self, ev, texto, ahora):
        tipo = ev[0]
        if tipo == "letra":
            self.voz.decir(nombre_hablado(ev[1]), "letra")
            self.pendiente = True
            self.t_ultimo = ahora
        elif tipo == "confirmar":
            self.leer(texto)
        elif tipo == "limpiar":
            self.pendiente = False
        else:                             # espacio, borrar, negación, pregunta: reinician el contador
            self.t_ultimo = ahora
            if not texto.strip():
                self.pendiente = False

    def tick(self, texto, ahora):
        if self.pendiente and ahora - self.t_ultimo >= self.inactividad:
            self.leer(texto)

    def leer(self, texto):
        if self.pendiente:
            palabra = ultima_palabra(texto)
            if palabra:
                self.voz.decir(palabra.lower(), "palabra")
        self.pendiente = False


def enviar_a_hardware(texto_actual):
    """Envía el texto actual al microcontrolador.

    Por ahora es un MOCK. En la etapa de hardware se reemplaza por algo como:
        puerto = serial.Serial("COM3", 115200)          # pip install pyserial
        puerto.write(f"T:{texto_actual}\\n".encode("ascii", "replace"))
    Se llama solo cuando el texto cambia, no en cada frame.
    """
    print(f"[SERIAL MOCK] Enviando al ESP32: {limpiar_texto(texto_actual)}")


# =============================================================================
# 9. INTERFAZ GRÁFICA
# =============================================================================
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class EstadoUI:
    """Todo lo que necesitan las ventanas para dibujarse en un frame."""
    sena: Optional[str]         # letra o configuración que se muestra en grande
    estado_sena: str            # "ninguna" | "sostener" | "mover" | "registrada" | "pose"
    progreso: float
    texto: str
    ahora: float
    ultimo_registro: float
    modo_facial: str
    progreso_pregunta: float
    ratio_cejas: float
    calib: float                # 0..1: progreso de la calibración del rostro (1 = lista)
    fps: float
    voz: str


# --- Texto (con soporte para la Ñ y sin guiones bajos) ---------------------------
def medir_texto(texto, escala, grosor=1, fuente=FUENTE):
    limpio = _a_ascii(limpiar_texto(texto)).replace("Ñ", "N").replace("ñ", "n")
    (w, h), base = cv2.getTextSize(limpio, fuente, escala, grosor)
    return w, h


def poner_texto(img, texto, org, escala=0.5, color=TEXTO, grosor=1, fuente=FUENTE):
    """cv2.putText que reemplaza "_" por " " y dibuja la Ñ (las fuentes Hershey no la incluyen)."""
    texto = _a_ascii(limpiar_texto(texto))
    x, y = org
    trozo = ""

    def volcar():
        nonlocal x, trozo
        if trozo:
            cv2.putText(img, trozo, (int(x), int(y)), fuente, escala, color, grosor, cv2.LINE_AA)
            x += cv2.getTextSize(trozo, fuente, escala, grosor)[0][0]
            trozo = ""

    for ch in texto:
        if ch in "Ññ":
            volcar()
            base = "N" if ch == "Ñ" else "n"
            (w, h), _ = cv2.getTextSize(base, fuente, escala, grosor)
            cv2.putText(img, base, (int(x), int(y)), fuente, escala, color, grosor, cv2.LINE_AA)
            ancho_t = max(int(w * 0.72), 4)       # tilde: un periodo de onda seno, centrado sobre la letra
            cx, cy = int(x + w / 2), int(y - h - max(4, int(h * 0.20)))
            amplitud = max(int(h * 0.10), 1)
            u = np.linspace(0.0, 1.0, 16)
            onda = np.stack([cx - ancho_t / 2 + ancho_t * u, cy - amplitud * np.sin(2 * np.pi * u)], axis=1)
            cv2.polylines(img, [onda.astype(np.int32).reshape(-1, 1, 2)], False, color,
                          max(1, int(grosor * 0.7)), cv2.LINE_AA)
            x += w
        else:
            trozo += ch
    volcar()


def texto_centrado(img, texto, cx, cy, escala, color, grosor, fuente=FUENTE):
    w, h = medir_texto(texto, escala, grosor, fuente)
    poner_texto(img, texto, (cx - w / 2, cy + h / 2), escala, color, grosor, fuente)


def escala_para_ancho(texto, ancho_max, escala_max, grosor, fuente=FUENTE):
    w, _ = medir_texto(texto, escala_max, grosor, fuente)
    return escala_max if w <= ancho_max else escala_max * ancho_max / w


def ajustar_lineas(texto, ancho_max, escala, fuente=FUENTE):
    """Parte un texto en líneas que quepan en `ancho_max` píxeles."""
    lineas, actual = [], ""
    for palabra in limpiar_texto(texto).split():
        prueba = (actual + " " + palabra).strip()
        if medir_texto(prueba, escala, 1, fuente)[0] > ancho_max and actual:
            lineas.append(actual)
            actual = palabra
        else:
            actual = prueba
    return lineas + [actual] if actual else lineas


# --- Formas ---------------------------------------------------------------------
def rect_redondeado(img, x0, y0, x1, y1, radio, color):
    """Rectángulo relleno con esquinas redondeadas."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    radio = int(max(0, min(radio, (x1 - x0) // 2, (y1 - y0) // 2)))
    if x1 <= x0 or y1 <= y0:
        return
    cv2.rectangle(img, (x0 + radio, y0), (x1 - radio, y1), color, -1)
    cv2.rectangle(img, (x0, y0 + radio), (x1, y1 - radio), color, -1)
    for cx, cy in ((x0 + radio, y0 + radio), (x1 - radio, y0 + radio),
                   (x0 + radio, y1 - radio), (x1 - radio, y1 - radio)):
        cv2.circle(img, (cx, cy), radio, color, -1, cv2.LINE_AA)


def aclarar(color, factor=0.45):
    return tuple(int(c + (255 - c) * factor) for c in color)


def barra_progreso(img, x0, y0, x1, y1, progreso, color):
    """Barra moderna: pista oscura redondeada, relleno de color con brillo superior."""
    alto = y1 - y0
    rect_redondeado(img, x0, y0, x1, y1, alto // 2, (55, 52, 50))
    progreso = max(0.0, min(1.0, progreso))
    x_fin = x0 + (x1 - x0) * progreso
    if x_fin - x0 >= 2:
        rect_redondeado(img, x0, y0, x_fin, y1, alto // 2, color)
        rect_redondeado(img, x0 + 3, y0 + 2, x_fin - 3, y0 + alto * 0.4, alto // 4, aclarar(color))


def tarjeta(img, x0, y0, x1, y1, color=TARJETA, borde=BORDE):
    rect_redondeado(img, x0, y0, x1, y1, 10, borde)
    rect_redondeado(img, x0 + 1, y0 + 1, x1 - 1, y1 - 1, 9, color)


def esquinas(img, x0, y0, x1, y1, color, largo=22, grosor=3):
    """Recuadro elegante: solo las esquinas marcadas."""
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (x, y), (x + dx * largo, y), color, grosor, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * largo), color, grosor, cv2.LINE_AA)


def pastilla_modo(img, x_derecha, y, modo, progreso=0.0):
    """Indicador "Modo: ..." alineado a la derecha, con punto de color y mini-barra de carga."""
    nombre, color = MODOS_FACIALES[modo]
    etiqueta = f"Modo: {nombre}"
    w, h = medir_texto(etiqueta, 0.5)
    x0, x1 = x_derecha - w - 44, x_derecha
    rect_redondeado(img, x0, y, x1, y + 28, 14, (35, 32, 30))
    cv2.circle(img, (x0 + 16, y + 14), 6, color, -1, cv2.LINE_AA)
    poner_texto(img, etiqueta, (x0 + 30, y + 14 + h // 2), 0.5, color, 1)
    if modo == "CEJAS":
        barra_progreso(img, x0 + 10, y + 32, x1 - 10, y + 38, progreso, ACENTO)


def medidor_cejas(img, x, y, ui):
    """Medidor SIEMPRE visible de las cejas: muestra cuánto se levantan respecto a tu línea base."""
    ancho = 150
    rect_redondeado(img, x, y, x + ancho + 20, y + 44, 8, (35, 32, 30))
    poner_texto(img, "CEJAS", (x + 10, y + 16), 0.42, TEXTO_SUAVE, 1)
    if ui.calib < 1.0:
        poner_texto(img, f"calibrando {int(ui.calib * 100)}%", (x + 66, y + 16), 0.42, AMBAR, 1)
        barra_progreso(img, x + 10, y + 26, x + 10 + ancho, y + 36, ui.calib, AMBAR)
        return
    minimo, maximo = 0.95, 1.20
    frac = (ui.ratio_cejas - minimo) / (maximo - minimo)
    umbral = (CEJAS_UMBRAL - minimo) / (maximo - minimo)
    color = EXITO if ui.modo_facial == "PREGUNTA" else (ACENTO if ui.ratio_cejas >= CEJAS_UMBRAL else TEXTO_SUAVE)
    poner_texto(img, f"x{ui.ratio_cejas:.2f}", (x + 66, y + 16), 0.42, color, 1)
    barra_progreso(img, x + 10, y + 26, x + 10 + ancho, y + 36, frac, color)
    tx = int(x + 10 + ancho * max(0.0, min(1.0, umbral)))
    cv2.line(img, (tx, y + 22), (tx, y + 40), (255, 255, 255), 1, cv2.LINE_AA)   # marca del umbral


def dibujar_estela(frame, tray, punto):
    """Dibuja el recorrido reciente del punto rastreado (se ve cómo se traza la J, la Z, el ?...)."""
    pts = tray.serie_px(punto)
    for i in range(1, len(pts)):
        k = i / len(pts)
        color = (int(60 + 195 * k), int(180 * k + 40), int(255 * (1 - k * 0.4)))
        cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])), color,
                 2 + int(3 * k), cv2.LINE_AA)
    if pts:
        cv2.circle(frame, (int(pts[-1][0]), int(pts[-1][1])), 6, (255, 255, 255), -1, cv2.LINE_AA)


def dibujar_camara(frame, ui, mano, rasgos, rostro, analizador, tray, diagnostico):
    """Ventana de cámara: esqueleto de la mano, recuadro, etiqueta, estela, medidor de cejas, modo facial y barra."""
    alto, ancho = frame.shape[:2]
    color_estado = {"registrada": EXITO, "sostener": ACENTO, "mover": TEXTO_SUAVE, "pose": AMBAR}.get(
        ui.estado_sena, TEXTO_SUAVE)

    # --- Puntos faciales clave (cejas y comisuras) ---
    if rostro is not None:
        for i in (*CEJA_SUP_DER, *CEJA_SUP_IZQ, *CANTO_DER, *CANTO_IZQ):
            cv2.circle(frame, (int(rostro[i][0]), int(rostro[i][1])), 2, (255, 220, 120), -1, cv2.LINE_AA)

    # --- Mano ---
    if mano is not None:
        landmarks, puntos = mano
        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                               mp_styles.get_default_hand_landmarks_style(),
                               mp_styles.get_default_hand_connections_style())
        xs = [x for x, _ in puntos]
        ys = [y for _, y in puntos]
        pad = 20
        x0, y0 = max(int(min(xs)) - pad, 0), max(int(min(ys)) - pad, 0)
        x1, y1 = min(int(max(xs)) + pad, ancho - 1), min(int(max(ys)) + pad, alto - 1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color_estado, 1)
        esquinas(frame, x0, y0, x1, y1, color_estado)

        etiqueta = ui.sena if ui.sena else "?"
        if ui.estado_sena == "pose" and etiqueta in PISTA_TRAZO:
            etiqueta += " (mueve)"
        tw, th = medir_texto(etiqueta, 0.8, 2, FUENTE_TITULO)
        ey = max(y0 - 10, th + 14)
        rect_redondeado(frame, x0, ey - th - 12, x0 + tw + 20, ey + 4, 6, color_estado)
        poner_texto(frame, etiqueta, (x0 + 10, ey - 4), 0.8, (20, 20, 20), 2, FUENTE_TITULO)
        pista = PISTA_TRAZO.get(ui.sena) if ui.estado_sena in ("sostener", "registrada", "mover") else None
        if pista:
            poner_texto(frame, pista, (x0 + 4, min(y1 + 18, alto - 60)), 0.45, AMBAR, 1)
        if ui.sena in PUNTO_DE_POSE and tray.vel > 0.6:   # la estela solo se dibuja mientras la mano se mueve
            dibujar_estela(frame, tray, PUNTO_DE_POSE[ui.sena])

    # --- Medidor de cejas (arriba a la izquierda) e indicador de estado facial (arriba a la derecha) ---
    medidor_cejas(frame, 12, 12, ui)
    pastilla_modo(frame, ancho - 12, 12, ui.modo_facial, ui.progreso_pregunta)

    # --- Diagnóstico (tecla D) ---
    if diagnostico:
        lineas = []
        if rasgos is not None:
            lineas += [f"{n}: {e} r={r:.2f}" for n, e, r in zip(("Ind", "Med", "Anu", "Men"), rasgos.estados, rasgos.ratios)]
            lineas.append(f"Pulgar t={rasgos.t_pulgar:.2f} d={rasgos.d_pulgar_indice:.2f} fuera={rasgos.pulgar_fuera}")
            lineas.append(f"orient={rasgos.orient:.0f} cruce={rasgos.sep_puntas:+.2f} vel={tray.vel:.1f}")
        if analizador.rasgos is not None:
            lineas.append(f"Giro {analizador.negacion.desviacion:+.2f} (+-{NEG_AMPLITUD})")
            lineas.append(f"Incl {analizador.afirmacion.desviacion:+.2f} (+-{AFIRM_AMPLITUD})")
        for i, linea in enumerate(lineas):
            poner_texto(frame, linea, (10, 78 + i * 18), 0.46, (0, 255, 255), 1)

    # --- Barra inferior de progreso ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, alto - 52), (ancho, alto), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    if ui.estado_sena == "registrada":
        estado = "REGISTRADO"
    elif ui.estado_sena == "sostener":
        estado = f"Sosten la sena {ui.sena} quieta... {int(ui.progreso * 100)}%"
    elif ui.estado_sena == "mover":
        estado = "Mano en movimiento: haz el trazo o sostenla quieta"
    elif ui.estado_sena == "pose":
        estado = f"Mueve la mano para escribir {ui.sena}"
    else:
        estado = "Muestra una sena a la camara"
    poner_texto(frame, estado, (16, alto - 30), 0.5, TEXTO if ui.estado_sena != "ninguna" else TEXTO_SUAVE, 1)
    poner_texto(frame, f"{ui.fps:4.1f} FPS", (ancho - 86, alto - 30), 0.45, TEXTO_SUAVE, 1)
    barra_progreso(frame, 16, alto - 20, ancho - 16, alto - 8, ui.progreso, EXITO)


Y_TOP, Y_FILA = 60, 250            # tarjetas "seña actual" y "cómo se hace"
Y_GUIA0, Y_GUIA1 = 258, 584        # tarjeta de la guía del alfabeto
_panel_base = None                 # fondo del panel (lo que no cambia), se dibuja una sola vez
_cache_lineas = {}                 # descripciones ya partidas en líneas
_cache_texto = ("", "")            # (texto completo, parte visible en la caja)


def _dibujar_entrada_guia(panel, i, activa):
    """Una fila de la guía del alfabeto, con o sin resaltado."""
    clave_g, desc, mov = GUIA[i]
    col, fila = divmod(i, (len(GUIA) + 1) // 2)
    x = 22 + col * 304
    y = Y_GUIA0 + 50 + fila * 19
    if activa:
        rect_redondeado(panel, x - 4, y - 15, x + 296, y + 5, 4, (70, 60, 40))
    color_clave = EXITO if activa else (AMBAR if mov else ACENTO)
    etiqueta = clave_g if len(clave_g) == 1 or clave_g == "?" else clave_g[:3] + "."
    poner_texto(panel, etiqueta, (x + 4, y), 0.45, color_clave, 1)
    poner_texto(panel, desc, (x + 46, y), 0.4, TEXTO if activa else (200, 200, 200), 1)


def _panel_estatico():
    """Todo lo que no cambia entre fotogramas: encabezado, tarjetas, etiquetas, guía y pie."""
    panel = np.full((ALTO_PANEL, ANCHO_PANEL, 3), FONDO, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (ANCHO_PANEL, 50), (40, 34, 30), -1)
    poner_texto(panel, "TRADUCTOR LSM", (16, 33), 0.8, ACENTO, 1, FUENTE_TITULO)
    tarjeta(panel, 14, Y_TOP, 300, Y_FILA)
    poner_texto(panel, "SENA ACTUAL", (28, Y_TOP + 26), 0.5, TEXTO_SUAVE, 1)
    tarjeta(panel, 308, Y_TOP, 626, Y_FILA)
    poner_texto(panel, "COMO SE HACE", (322, Y_TOP + 26), 0.5, TEXTO_SUAVE, 1)
    tarjeta(panel, 14, Y_GUIA0, 626, Y_GUIA1)
    poner_texto(panel, "GUIA RAPIDA  (naranja = lleva movimiento)", (28, Y_GUIA0 + 24), 0.5, TEXTO_SUAVE, 1)
    for i in range(len(GUIA)):
        _dibujar_entrada_guia(panel, i, False)
    tarjeta(panel, 14, 590, 626, 620)
    poner_texto(panel, "ROSTRO:  cejas arriba 1.5 s = ?   |   negar = borrar   |   asentir = confirmar y leer",
                (24, 610), 0.42, AMBAR, 1)
    tarjeta(panel, 14, 628, 626, 704, (18, 16, 14), ACENTO)
    poner_texto(panel, "TEXTO", (28, 650), 0.45, TEXTO_SUAVE, 1)
    poner_texto(panel, "Espacio | Backspace | C: limpiar | D: diagnostico | R: recalibrar | ESC", (14, 726), 0.38,
                TEXTO_SUAVE, 1)
    return panel


def _lineas(texto, ancho, escala):
    clave = (texto, ancho, escala)
    if clave not in _cache_lineas:
        _cache_lineas[clave] = ajustar_lineas(texto, ancho, escala)
    return _cache_lineas[clave]


def _texto_visible(texto):
    """La parte final del texto que cabe en la caja (los espacios se ven como espacios)."""
    global _cache_texto
    if _cache_texto[0] != texto:
        visible = limpiar_texto(texto)
        while visible and medir_texto(visible, 1.3, 2, FUENTE_TITULO)[0] > ANCHO_PANEL - 70:
            visible = visible[1:]  # Si no cabe, se muestra el final
        _cache_texto = (texto, visible)
    return _cache_texto[1]


def dibujar_panel(ui):
    """Panel traductor generado con NumPy: seña actual, cómo se hace, guía del alfabeto y texto.

    El fondo (guía, tarjetas, etiquetas) se dibuja una sola vez y se copia en cada fotograma; solo se pinta
    lo que cambia. Así el panel cuesta ≈ 3 ms en lugar de ≈ 9 ms."""
    global _panel_base
    if _panel_base is None:
        _panel_base = _panel_estatico()
    panel = _panel_base.copy()
    pastilla_modo(panel, ANCHO_PANEL - 14, 10, ui.modo_facial, ui.progreso_pregunta)

    # --- Seña actual (izquierda) ---
    cx, cy = 157, 150
    if ui.sena:
        color = {"registrada": EXITO, "pose": AMBAR, "mover": TEXTO_SUAVE}.get(ui.estado_sena, TEXTO)
        largo = len(ui.sena)
        grosor = 7 if largo == 1 else 3
        # La letra gigante debe caber entre el título de la tarjeta y la barra; la Ñ lleva tilde y necesita más aire
        escala_max = (3.3 if ui.sena == "Ñ" else 4.4) if largo == 1 else 1.6
        escala = escala_para_ancho(ui.sena, 240, escala_max, grosor, FUENTE_TITULO)
        texto_centrado(panel, ui.sena, cx, cy, escala, color, grosor, FUENTE_TITULO)
    else:
        texto_centrado(panel, "-", cx, cy, 4.0, BORDE, 6, FUENTE_TITULO)
    barra_progreso(panel, 30, Y_FILA - 36, 284, Y_FILA - 24, ui.progreso, EXITO)
    porcentaje = {"registrada": "LISTO", "pose": "MUEVE LA MANO", "mover": "EN MOVIMIENTO"}.get(
        ui.estado_sena, f"{int(ui.progreso * 100)}%")
    texto_centrado(panel, porcentaje, cx, Y_FILA - 48, 0.45, TEXTO_SUAVE, 1)

    # --- Cómo se hace (derecha) ---
    clave = ui.sena if ui.sena in DESCRIPCIONES else None
    if clave:
        desc, mov = DESCRIPCIONES[clave]
        poner_texto(panel, clave, (322, Y_TOP + 62), 0.9, ACENTO, 2, FUENTE_TITULO)
        y = Y_TOP + 90
        for linea in _lineas(desc, 290, 0.55):
            poner_texto(panel, linea, (322, y), 0.55, TEXTO, 1)
            y += 22
        if mov:
            for linea in _lineas("Movimiento: " + mov, 290, 0.5):
                poner_texto(panel, linea, (322, y + 6), 0.5, AMBAR, 1)
                y += 20
    elif ui.sena in ("Q", "X"):
        poner_texto(panel, ui.sena, (322, Y_TOP + 62), 0.9, AMBAR, 2, FUENTE_TITULO)
    else:
        for i, linea in enumerate(_lineas("Muestra una sena a la camara. La guia de abajo explica cada letra.", 290, 0.5)):
            poner_texto(panel, linea, (322, Y_TOP + 60 + i * 22), 0.5, TEXTO_SUAVE, 1)

    # --- Fila resaltada de la guía ---
    for i, (clave_g, _, _) in enumerate(GUIA):
        if clave_g == ui.sena:
            _dibujar_entrada_guia(panel, i, True)
            break

    # --- Caja de texto acumulado (los espacios se ven como espacios) ---
    cursor = "|" if int(ui.ahora * 2) % 2 == 0 else " "
    poner_texto(panel, _texto_visible(ui.texto) + cursor, (28, 692), 1.3, TEXTO, 2, FUENTE_TITULO)

    # --- Estado de la voz (pie) ---
    color_voz = EXITO if ui.voz == "lista" else (AMBAR if ui.voz == "iniciando" else TEXTO_SUAVE)
    ancho_voz = medir_texto(f"Voz: {ui.voz}", 0.4)[0]
    poner_texto(panel, f"Voz: {ui.voz}", (ANCHO_PANEL - ancho_voz - 14, 726), 0.4, color_voz, 1)
    return panel


def colocar_ventanas(ancho_camara):
    """Cámara a la izquierda y panel a la derecha, sin encimarse."""
    cv2.namedWindow(VENTANA_CAMARA, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(VENTANA_PANEL, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(VENTANA_CAMARA, MARGEN_VENTANAS, MARGEN_VENTANAS)
    cv2.moveWindow(VENTANA_PANEL, MARGEN_VENTANAS * 2 + ancho_camara, MARGEN_VENTANAS)


def ventana_cerrada(nombre):
    """True si el usuario cerró la ventana con la X."""
    try:
        return cv2.getWindowProperty(nombre, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# =============================================================================
# 10. PROGRAMA PRINCIPAL
# =============================================================================
def detectar_mano(hands, rgb, ancho, alto):
    """Devuelve ((landmarks, puntos_px), rasgos, configuración) o (None, None, None) si no hay mano o hay error."""
    try:
        resultado = hands.process(rgb)
        if not resultado.multi_hand_landmarks:
            return None, None, None
        landmarks = resultado.multi_hand_landmarks[0]
        puntos = a_pixeles(landmarks.landmark, ancho, alto)
        rasgos = analizar_mano(puntos)
        return (landmarks, puntos), rasgos, clasificar_pose(rasgos)
    except Exception as e:  # Un frame defectuoso no debe tirar el programa
        avisar_una_vez(f"Error procesando la mano ({type(e).__name__}: {e}); se omite el frame.")
        return None, None, None


def detectar_rostro(face_mesh, rgb, ancho, alto):
    """Devuelve {índice: (x, y)} con los puntos faciales necesarios, o None si no hay cara o hay error."""
    try:
        resultado = face_mesh.process(rgb)
        if not resultado.multi_face_landmarks:
            return None
        return a_pixeles(resultado.multi_face_landmarks[0].landmark, ancho, alto, INDICES_ROSTRO)
    except Exception as e:
        avisar_una_vez(f"Error procesando el rostro ({type(e).__name__}: {e}); se omite el frame.")
        return None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara. Verifica que esté conectada y disponible.")

    estabilizador = Estabilizador(FRAMES_ESTABLES)
    escritor = Escritor(SEGUNDOS_PARA_ESCRIBIR)
    analizador = AnalizadorFacial()
    tray = Trayectoria()
    reconocedor = ReconocedorDinamico()
    barra = ProgresoSuave()
    voz = VozAsincrona(VOZ_HABILITADA)
    politica = PoliticaVoz(voz)
    texto_enviado = escritor.texto
    diagnostico = False
    ventanas_listas = False
    rostro = None
    fps, t_frame_anterior = 0.0, time.monotonic()
    n_frame = 0

    print("Traductor LSM iniciado. Mantén la cara neutra un momento para calibrar. ESC para salir.")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("[AVISO] No se pudo leer un frame de la cámara.")
                    break

                frame = cv2.flip(frame, 1)  # Espejo
                alto, ancho = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                ahora = time.monotonic()
                n_frame += 1

                # --- Percepción (cada canal tolera fallos por separado) ---
                mano, rasgos, pose_cruda = detectar_mano(hands, rgb, ancho, alto)
                if n_frame % PROCESAR_ROSTRO_CADA == 0:
                    rostro = detectar_rostro(face_mesh, rgb, ancho, alto)
                    evento_facial = analizador.actualizar(rostro, ahora)
                else:
                    evento_facial = None

                # --- Manos: configuración estable, trayectoria y señas con movimiento ---
                pose = estabilizador.actualizar(pose_cruda)
                gesto = None
                if rasgos is not None:
                    tray.agregar(ahora, rasgos, pose_cruda)
                    gesto = reconocedor.evaluar(tray, pose_cruda, ahora)
                else:
                    tray.perder()
                moviendo = rasgos is not None and tray.vel > V_QUIETO
                if gesto:
                    escritor.registrar_dinamico(gesto, ahora, pose)

                # --- Escritura: las letras estáticas solo avanzan con la mano quieta ---
                letra = pose if (pose in LETRAS_TIPEABLES and not moviendo) else None
                progreso = escritor.actualizar(letra, ahora)
                if evento_facial:
                    escritor.ejecutar(ACCIONES_FACIALES[evento_facial], ahora)

                # --- Voz y hardware: reaccionan a lo que ocurrió en este frame ---
                for ev in escritor.sacar_eventos():
                    politica.evento(ev, escritor.texto, ahora)
                politica.tick(escritor.texto, ahora)
                if escritor.texto != texto_enviado:
                    enviar_a_hardware(escritor.texto)
                    texto_enviado = escritor.texto

                # --- Qué se muestra en grande ---
                gesto_mostrado, t_gesto = escritor.destello
                if gesto_mostrado and ahora - t_gesto < DURACION_DESTELLO:
                    sena_ui, estado_ui, progreso = gesto_mostrado, "registrada", 1.0
                elif pose is not None and rasgos is not None and pose not in POSES_SILENCIOSAS:
                    sena_ui = pose
                    if pose not in LETRAS_TIPEABLES:
                        estado_ui = "pose"
                    elif moviendo:
                        estado_ui = "mover"
                    else:
                        estado_ui = "registrada" if escritor.registrada else "sostener"
                else:
                    sena_ui, estado_ui = None, "ninguna"

                # --- Interfaz ---
                dt = ahora - t_frame_anterior
                t_frame_anterior = ahora
                if dt > 0:
                    fps = 1.0 / dt if fps == 0 else fps * 0.9 + (1.0 / dt) * 0.1
                ui = EstadoUI(sena_ui, estado_ui, barra.actualizar(progreso), escritor.texto, ahora,
                              escritor.ultimo_registro, analizador.modo, analizador.progreso_pregunta,
                              analizador.ratio_cejas, analizador.progreso_calibracion, fps, voz.estado)
                dibujar_camara(frame, ui, mano, rasgos, rostro, analizador, tray, diagnostico)
                panel = dibujar_panel(ui)

                if not ventanas_listas:
                    colocar_ventanas(ancho)
                    ventanas_listas = True
                cv2.imshow(VENTANA_CAMARA, frame)
                cv2.imshow(VENTANA_PANEL, panel)

                # --- Teclado ---
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == 27:                          # ESC
                    break
                elif tecla == 32:                        # Espacio
                    escritor.ejecutar("ESPACIO", ahora)
                elif tecla == 8:                         # Backspace
                    escritor.ejecutar("BORRAR", ahora)
                elif tecla in (ord("c"), ord("C")):
                    escritor.ejecutar("LIMPIAR", ahora)
                elif tecla in (ord("d"), ord("D")):
                    diagnostico = not diagnostico
                elif tecla in (ord("r"), ord("R")):
                    analizador.recalibrar()

                if ventana_cerrada(VENTANA_CAMARA) or ventana_cerrada(VENTANA_PANEL):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            voz.cerrar()

    print(f"Texto final: {limpiar_texto(escritor.texto)!r}")
    print("Programa cerrado correctamente.")


if __name__ == "__main__":
    main()
