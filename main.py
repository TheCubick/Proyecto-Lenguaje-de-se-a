"""Sistema de Reconocimiento de Lengua de Señas Mexicana (LSM) con MediaPipe + OpenCV.

Reconoce letras estáticas del alfabeto dactilológico y las escribe manteniendo la seña
1.5 s ("hold-to-type"). Teclas: Espacio, Backspace, C (limpiar), D (diagnóstico), ESC (salir).
"""
import math
import time

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
VENTANA_CAMARA = "Camara IA - Deteccion LSM"
VENTANA_PANEL = "Panel Traductor LSM"
SEGUNDOS_PARA_ESCRIBIR = 1.5   # Tiempo que hay que sostener una seña para registrarla
FRAMES_ESTABLES = 4            # Frames seguidos con la misma seña antes de aceptarla (anti-parpadeo)
MARGEN_VENTANAS = 30
ANCHO_PANEL = 640

# ---------------------------------------------------------------------------
# Umbrales geométricos (ajustables; activa el diagnóstico con la tecla D para calibrarlos)
# ---------------------------------------------------------------------------
# r = dist(punta, muñeca) / dist(nudillo, muñeca)
R_EXTENDIDO = 1.45        # r mayor => dedo estirado (además de estar recto)
ANG_RECTO = 150           # Ángulo mínimo en la articulación media para considerarlo recto
R_CERRADO = 1.05          # r menor => punta metida en la palma (puño)
# Pulgar
T_PULGAR_CRUZADO = 0.20   # Proyección de la punta del pulgar sobre el eje índice->meñique
PULGAR_FUERA = 0.60       # dist(punta pulgar, nudillo índice) / palma para considerarlo abierto
# Separaciones como ángulo entre dedos (no depende del largo de cada dedo)
ANG_U_MAX = 10            # U: índice y medio casi paralelos
ANG_V_MIN = 18            # V: índice y medio abiertos
ANG_ABERTURA_MANO = 35    # Índice-meñique: >= mano abierta (ESPACIO), < dedos juntos (B)
# Contacto índice-pulgar para la O
O_CONTACTO = 0.30         # dist(punta pulgar, punta índice) / palma
O_T_MAX = 0.30            # En la O el pulgar se queda del lado del índice (en la E cruza la palma)
# Ángulo pulgar-índice para la L
L_ANG_MIN, L_ANG_MAX = 50, 130

# Colores (BGR)
FONDO = (28, 24, 22)
TARJETA = (48, 42, 38)
BORDE = (80, 72, 66)
TEXTO = (235, 235, 235)
TEXTO_SUAVE = (150, 145, 140)
ACENTO = (255, 190, 60)     # Azul claro
EXITO = (120, 220, 90)      # Verde
ALERTA = (80, 120, 255)     # Rojo/naranja

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
FUENTE_TITULO = cv2.FONT_HERSHEY_DUPLEX

# Guía rápida (sin acentos/ñ: OpenCV no las dibuja)
GUIA = [
    ("A", "Puno + pulgar al costado"),
    ("B", "4 dedos juntos, pulgar dentro"),
    ("E", "Dedos en garra sobre pulgar"),
    ("I", "Solo menique arriba"),
    ("L", "Indice + pulgar en L"),
    ("O", "Circulo indice-pulgar"),
    ("U", "Indice + medio juntos"),
    ("V", "Indice + medio en V"),
    ("W", "Indice + medio + anular"),
    ("Y", "Pulgar + menique"),
    ("ESPACIO", "Mano abierta, 5 dedos"),
    ("BORRAR", "Puno, pulgar adentro"),
]

# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# Landmarks
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
PINKY_MCP, PINKY_TIP = 17, 20
# (MCP, PIP, DIP, TIP) de índice, medio, anular y meñique
DEDOS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]

EXT, DOBLADO, CERRADO = "EXT", "DOB", "CER"


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angulo(a, b, c):
    """Ángulo en grados en el vértice b (180 = recto)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


def angulo_entre(v1, v2):
    """Ángulo en grados entre dos vectores."""
    return angulo((v1[0], v1[1]), (0, 0), (v2[0], v2[1]))


def a_pixeles(landmarks, ancho, alto):
    """Normalizado -> píxeles, para que las distancias no se deformen por el aspecto de la cámara."""
    return [(lm.x * ancho, lm.y * alto) for lm in landmarks]


def analizar_mano(p):
    """Calcula los rasgos geométricos de la mano. No depende de si es mano izquierda o derecha."""
    palma = dist(p[WRIST], p[MIDDLE_MCP]) or 1.0

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

    # Posición del pulgar proyectada sobre el eje nudillo índice -> nudillo meñique:
    # t < 0 queda por fuera del índice, t > 0 cruza por delante de la palma.
    eje = (p[PINKY_MCP][0] - p[INDEX_MCP][0], p[PINKY_MCP][1] - p[INDEX_MCP][1])
    rel = (p[THUMB_TIP][0] - p[INDEX_MCP][0], p[THUMB_TIP][1] - p[INDEX_MCP][1])
    t_pulgar = (rel[0] * eje[0] + rel[1] * eje[1]) / ((eje[0] ** 2 + eje[1] ** 2) or 1.0)

    pulgar_recto = angulo(p[THUMB_MCP], p[THUMB_IP], p[THUMB_TIP]) > 140
    d_pulgar_indice = dist(p[THUMB_TIP], p[INDEX_MCP]) / palma

    return {
        "palma": palma,
        "estados": estados,
        "ratios": ratios,
        "t_pulgar": t_pulgar,
        "pulgar_fuera": pulgar_recto and d_pulgar_indice > PULGAR_FUERA and t_pulgar < 0,
        "pulgar_cruzado": t_pulgar > T_PULGAR_CRUZADO,
        "d_pulgar_indice": d_pulgar_indice,
    }


def angulo_dedos(p, mcp_a, tip_a, mcp_b, tip_b):
    """Ángulo de apertura entre dos dedos (nudillo -> punta)."""
    v_a = (p[tip_a][0] - p[mcp_a][0], p[tip_a][1] - p[mcp_a][1])
    v_b = (p[tip_b][0] - p[mcp_b][0], p[tip_b][1] - p[mcp_b][1])
    return angulo_entre(v_a, v_b)


def clasificar_sena(p, info):
    """Devuelve la seña reconocida ("A", "B", ..., "ESPACIO", "BORRAR") o None."""
    est = info["estados"]
    ext = [e == EXT for e in est]
    indice, medio, anular, menique = ext
    n_ext = sum(ext)

    # --- Cuatro dedos estirados: B (juntos, pulgar dentro) o ESPACIO (mano abierta) ---
    if n_ext == 4:
        abertura = angulo_dedos(p, INDEX_MCP, INDEX_TIP, PINKY_MCP, PINKY_TIP)
        if info["pulgar_fuera"] and abertura >= ANG_ABERTURA_MANO:
            return "ESPACIO"
        if not info["pulgar_fuera"] and abertura < ANG_ABERTURA_MANO:
            return "B"
        return None

    # --- W: índice, medio y anular ---
    if indice and medio and anular and not menique:
        return "W"

    # --- U / V: índice y medio ---
    if indice and medio and not anular and not menique:
        apertura = angulo_dedos(p, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP)
        if apertura < ANG_U_MAX:
            return "U"
        if apertura > ANG_V_MIN:
            return "V"
        return None  # Zona ambigua: mejor no adivinar

    # --- I / Y: meñique solo (con o sin pulgar) ---
    if menique and not (indice or medio or anular):
        return "Y" if info["pulgar_fuera"] else "I"

    # --- L: índice y pulgar a ~90 grados ---
    if indice and not (medio or anular or menique):
        if info["pulgar_fuera"]:
            v_pulgar = (p[THUMB_TIP][0] - p[THUMB_MCP][0], p[THUMB_TIP][1] - p[THUMB_MCP][1])
            v_indice = (p[INDEX_TIP][0] - p[INDEX_MCP][0], p[INDEX_TIP][1] - p[INDEX_MCP][1])
            if L_ANG_MIN <= angulo_entre(v_pulgar, v_indice) <= L_ANG_MAX:
                return "L"
        return None

    if n_ext > 0:
        return None

    # --- Ningún dedo estirado: O, A, BORRAR o E ---
    contacto_o = dist(p[THUMB_TIP], p[INDEX_TIP]) / info["palma"]
    if contacto_o < O_CONTACTO and est[0] != CERRADO and info["t_pulgar"] < O_T_MAX:
        return "O"

    if all(e == CERRADO for e in est):
        return "BORRAR" if info["pulgar_cruzado"] else "A"

    if sum(e == DOBLADO for e in est) >= 2 and info["pulgar_cruzado"]:
        return "E"

    return None


# ---------------------------------------------------------------------------
# Estabilidad y escritura
# ---------------------------------------------------------------------------
class Estabilizador:
    """Solo acepta una seña nueva cuando se repite N frames seguidos."""

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
    """Hold-to-type: registra la seña tras sostenerla SEGUNDOS_PARA_ESCRIBIR.

    Para repetir la misma letra hay que cambiar de seña (o bajar la mano) y volver a hacerla.
    """

    def __init__(self, segundos):
        self.segundos = segundos
        self.texto = ""
        self.sena = None
        self.inicio = 0.0
        self.registrada = False
        self.ultimo_registro = -10.0

    def aplicar(self, sena, ahora):
        if sena == "ESPACIO":
            self.texto += " "
        elif sena == "BORRAR":
            self.texto = self.texto[:-1]
        else:
            self.texto += sena
        self.ultimo_registro = ahora

    def limpiar(self):
        self.texto = ""

    def actualizar(self, sena, ahora):
        """Devuelve el progreso de 0.0 a 1.0 de la seña actual."""
        if sena != self.sena:
            self.sena, self.inicio, self.registrada = sena, ahora, False
        if sena is None:
            return 0.0
        if self.registrada:
            return 1.0
        progreso = (ahora - self.inicio) / self.segundos
        if progreso >= 1.0:
            self.aplicar(sena, ahora)
            self.registrada = True
            return 1.0
        return progreso


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def texto_centrado(img, texto, cx, cy, escala, color, grosor, fuente=FUENTE):
    (w, h), _ = cv2.getTextSize(texto, fuente, escala, grosor)
    cv2.putText(img, texto, (int(cx - w / 2), int(cy + h / 2)), fuente, escala, color, grosor, cv2.LINE_AA)


def escala_para_ancho(texto, ancho_max, escala_max, grosor, fuente=FUENTE):
    (w, _), _ = cv2.getTextSize(texto, fuente, escala_max, grosor)
    return escala_max if w <= ancho_max else escala_max * ancho_max / w


def tarjeta(img, x0, y0, x1, y1, color=TARJETA, borde=BORDE):
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), borde, 1)


def barra_progreso(img, x0, y0, x1, y1, progreso, color):
    cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), -1)
    relleno = int(x0 + (x1 - x0) * max(0.0, min(1.0, progreso)))
    if relleno > x0:
        cv2.rectangle(img, (x0, y0), (relleno, y1), color, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)


def esquinas(img, x0, y0, x1, y1, color, largo=22, grosor=3):
    """Recuadro elegante: solo las esquinas marcadas."""
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (x, y), (x + dx * largo, y), color, grosor, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * largo), color, grosor, cv2.LINE_AA)


def dibujar_camara(frame, puntos, landmarks, sena, progreso, registrada, info, diagnostico):
    alto, ancho = frame.shape[:2]

    if landmarks is not None:
        mp_draw.draw_landmarks(
            frame, landmarks, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )
        xs = [x for x, _ in puntos]
        ys = [y for _, y in puntos]
        pad = 20
        x0, y0 = max(int(min(xs)) - pad, 0), max(int(min(ys)) - pad, 0)
        x1, y1 = min(int(max(xs)) + pad, ancho - 1), min(int(max(ys)) + pad, alto - 1)
        color = EXITO if registrada else (ACENTO if sena else TEXTO_SUAVE)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
        esquinas(frame, x0, y0, x1, y1, color)

        etiqueta = sena if sena else "?"
        (tw, th), _ = cv2.getTextSize(etiqueta, FUENTE_TITULO, 0.8, 2)
        ey = max(y0 - 10, th + 14)
        cv2.rectangle(frame, (x0, ey - th - 12), (x0 + tw + 20, ey + 4), color, -1)
        cv2.putText(frame, etiqueta, (x0 + 10, ey - 4), FUENTE_TITULO, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

        if diagnostico and info is not None:
            lineas = [f"{n}: {e} r={r:.2f}" for n, e, r in
                      zip(("Ind", "Med", "Anu", "Men"), info["estados"], info["ratios"])]
            lineas.append(f"Pulgar t={info['t_pulgar']:.2f} d={info['d_pulgar_indice']:.2f}")
            lineas.append(f"fuera={info['pulgar_fuera']} cruzado={info['pulgar_cruzado']}")
            for i, linea in enumerate(lineas):
                cv2.putText(frame, linea, (10, 30 + i * 20), FUENTE, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Barra de progreso de escritura en la parte inferior
    cv2.rectangle(frame, (0, alto - 46), (ancho, alto), (20, 20, 20), -1)
    if sena:
        estado = "REGISTRADO" if registrada else f"Sosten la sena... {int(progreso * 100)}%"
        cv2.putText(frame, estado, (14, alto - 28), FUENTE, 0.5, TEXTO, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Muestra una sena a la camara", (14, alto - 28), FUENTE, 0.5, TEXTO_SUAVE, 1, cv2.LINE_AA)
    barra_progreso(frame, 14, alto - 18, ancho - 14, alto - 8, progreso, EXITO if registrada else ACENTO)


def dibujar_panel(alto, sena, progreso, registrada, texto, ahora, ultimo_registro):
    panel = np.full((alto, ANCHO_PANEL, 3), FONDO, dtype=np.uint8)

    # Encabezado
    cv2.rectangle(panel, (0, 0), (ANCHO_PANEL, 44), (40, 34, 30), -1)
    cv2.putText(panel, "TRADUCTOR LSM", (16, 30), FUENTE_TITULO, 0.8, ACENTO, 1, cv2.LINE_AA)
    cv2.putText(panel, "MediaPipe + OpenCV", (ANCHO_PANEL - 190, 29), FUENTE, 0.5, TEXTO_SUAVE, 1, cv2.LINE_AA)

    # Distribución: seña actual (izquierda), guía (derecha), caja de texto (abajo)
    caja_alto = 120
    y_caja = alto - caja_alto - 14
    x_div = 290

    # --- Seña actual ---
    tarjeta(panel, 14, 58, x_div - 8, y_caja - 12)
    cv2.putText(panel, "SENA ACTUAL", (28, 84), FUENTE, 0.5, TEXTO_SUAVE, 1, cv2.LINE_AA)
    cx, cy = (14 + x_div - 8) // 2, (58 + y_caja - 12) // 2 + 6
    recien = ahora - ultimo_registro < 0.4  # Destello al registrar
    if sena:
        color = EXITO if (registrada or recien) else TEXTO
        grosor = 8 if len(sena) == 1 else 3
        escala = escala_para_ancho(sena, x_div - 70, 6.0 if len(sena) == 1 else 1.6, grosor, FUENTE_TITULO)
        texto_centrado(panel, sena, cx, cy, escala, color, grosor, FUENTE_TITULO)
    else:
        texto_centrado(panel, "-", cx, cy, 4.0, BORDE, 6, FUENTE_TITULO)
    barra_progreso(panel, 30, y_caja - 40, x_div - 24, y_caja - 30, progreso, EXITO if registrada else ACENTO)

    # --- Guía rápida ---
    tarjeta(panel, x_div, 58, ANCHO_PANEL - 14, y_caja - 12)
    cv2.putText(panel, "GUIA RAPIDA", (x_div + 14, 84), FUENTE, 0.5, TEXTO_SUAVE, 1, cv2.LINE_AA)
    alto_disp = (y_caja - 12) - 100
    paso = max(14, min(22, alto_disp // len(GUIA)))
    for i, (letra, desc) in enumerate(GUIA):
        y = 104 + i * paso
        activa = letra == sena
        if activa:
            cv2.rectangle(panel, (x_div + 6, y - paso + 6), (ANCHO_PANEL - 20, y + 5), (70, 60, 40), -1)
        color_letra = EXITO if activa else ACENTO
        etiqueta = letra if len(letra) == 1 else letra[:3] + "."
        cv2.putText(panel, etiqueta, (x_div + 14, y), FUENTE, 0.45, color_letra, 1, cv2.LINE_AA)
        cv2.putText(panel, desc, (x_div + 64, y), FUENTE, 0.42, TEXTO if activa else (200, 200, 200), 1, cv2.LINE_AA)

    # --- Caja de texto acumulado ---
    tarjeta(panel, 14, y_caja, ANCHO_PANEL - 14, alto - 14, (18, 16, 14), ACENTO)
    cv2.putText(panel, "TEXTO", (28, y_caja + 24), FUENTE, 0.5, TEXTO_SUAVE, 1, cv2.LINE_AA)
    visible = texto.replace(" ", "_")
    max_ancho = ANCHO_PANEL - 70
    while visible and cv2.getTextSize(visible, FUENTE_TITULO, 1.4, 2)[0][0] > max_ancho:
        visible = visible[1:]  # Muestra el final si no cabe
    cursor = "|" if int(ahora * 2) % 2 == 0 else ""
    cv2.putText(panel, visible + cursor, (28, y_caja + 72), FUENTE_TITULO, 1.4, TEXTO, 2, cv2.LINE_AA)
    cv2.putText(panel, "Espacio: espacio | Backspace: borrar | C: limpiar | D: diagnostico | ESC: salir",
                (28, alto - 26), FUENTE, 0.38, TEXTO_SUAVE, 1, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# Ventanas
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara. Verifica que esté conectada y disponible.")

    estabilizador = Estabilizador(FRAMES_ESTABLES)
    escritor = Escritor(SEGUNDOS_PARA_ESCRIBIR)
    diagnostico = False
    ventanas_listas = False

    print("Traductor LSM iniciado. Sosten una sena 1.5 s para escribirla. ESC para salir.")

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
            resultado = hands.process(rgb)

            landmarks, puntos, info, sena_cruda = None, None, None, None
            if resultado.multi_hand_landmarks:
                landmarks = resultado.multi_hand_landmarks[0]
                puntos = a_pixeles(landmarks.landmark, ancho, alto)
                info = analizar_mano(puntos)
                sena_cruda = clasificar_sena(puntos, info)

            ahora = time.monotonic()
            sena = estabilizador.actualizar(sena_cruda)
            progreso = escritor.actualizar(sena, ahora)

            dibujar_camara(frame, puntos, landmarks, sena, progreso, escritor.registrada, info, diagnostico)
            panel = dibujar_panel(alto, sena, progreso, escritor.registrada, escritor.texto,
                                  ahora, escritor.ultimo_registro)

            if not ventanas_listas:
                colocar_ventanas(ancho)
                ventanas_listas = True
            cv2.imshow(VENTANA_CAMARA, frame)
            cv2.imshow(VENTANA_PANEL, panel)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:                      # ESC
                break
            elif tecla == 32:                    # Espacio
                escritor.aplicar("ESPACIO", ahora)
            elif tecla == 8:                     # Backspace
                escritor.aplicar("BORRAR", ahora)
            elif tecla in (ord("c"), ord("C")):
                escritor.limpiar()
            elif tecla in (ord("d"), ord("D")):
                diagnostico = not diagnostico

            if ventana_cerrada(VENTANA_CAMARA) or ventana_cerrada(VENTANA_PANEL):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    print(f"Texto final: {escritor.texto!r}")
    print("Programa cerrado correctamente.")


if __name__ == "__main__":
    main()
