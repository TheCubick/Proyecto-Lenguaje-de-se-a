import cv2
import mediapipe as mp
import os
import math
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMES_DIR = os.path.join(BASE_DIR, "memes")
TAM_MEME = 380

# Frames consecutivos que un gesto debe mantenerse para mostrarse (anti-parpadeo)
FRAMES_ESTABLES = 6
# Frames que se conserva la última cara vista (la mano puede taparla al rascarse)
FRAMES_MEMORIA_CARA = 20

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 1. Carga de imágenes con respaldo visual 🖼️
def cargar_imagen(nombre):
    ruta = os.path.join(MEMES_DIR, nombre)
    img = cv2.imread(ruta) if os.path.exists(ruta) else None
    if img is not None:
        return cv2.resize(img, (TAM_MEME, TAM_MEME))
    print(f"[AVISO] No se encontró la imagen: {ruta}")
    negra = np.zeros((TAM_MEME, TAM_MEME, 3), dtype=np.uint8)
    cv2.putText(negra, f"Falta {nombre}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return negra

# Gesto -> (imagen, color del texto en BGR)
MEMES = {
    "DUDA":    (cargar_imagen("chango_pensando.png"), (0, 255, 0)),    # Chango rascándose 🐵
    "LENTES":  (cargar_imagen("gato_lentes.png"),     (0, 255, 255)),  # Gatito con lentes ☝️🤓
    "PUNO":    (cargar_imagen("gato_puno.png"),       (0, 0, 255)),    # Gatito tirando golpe 👊
    "NINGUNO": (np.zeros((TAM_MEME, TAM_MEME, 3), dtype=np.uint8), (150, 150, 150)),  # Negro ⬛
}

# Índices de landmarks de la mano
WRIST = 0
THUMB_TIP = 4
# (MCP, PIP, TIP) para índice, medio, anular y meñique
DEDOS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]


def a_pixeles(landmarks, ancho, alto):
    """Convierte landmarks normalizados a píxeles para que las distancias no se deformen por el aspecto 16:9."""
    return [(lm.x * ancho, lm.y * alto) for lm in landmarks]


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angulo(a, b, c):
    """Ángulo en grados en el vértice b (180 = recto)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


def dedo_extendido(p, mcp, pip, tip):
    # Recto en la articulación media y la punta claramente más lejos de la muñeca que el PIP
    return angulo(p[mcp], p[pip], p[tip]) > 160 and dist(p[tip], p[WRIST]) > dist(p[pip], p[WRIST]) * 1.15


def dedo_cerrado(p, mcp, pip, tip):
    # La punta se dobla de regreso: queda más cerca de la muñeca que su articulación media
    return dist(p[tip], p[WRIST]) < dist(p[pip], p[WRIST])


def mano_en_cabeza(p, cara):
    """True si al menos 2 puntas de dedos están en la zona de la cabeza (frente, coronilla o sienes)."""
    if cara is None:
        return False
    x, y, w, h = cara
    # Zona: desde ~0.8 caras por encima de la frente hasta la altura de los ojos,
    # ensanchada a los lados para cubrir sienes/orejas.
    x0, x1 = x - 0.5 * w, x + 1.5 * w
    y0, y1 = y - 0.8 * h, y + 0.45 * h
    puntas = [p[t] for _, _, t in DEDOS] + [p[THUMB_TIP]]
    dentro = sum(1 for (px, py) in puntas if x0 <= px <= x1 and y0 <= py <= y1)
    return dentro >= 2


def clasificar_gesto(p, cara):
    extendidos = [dedo_extendido(p, *d) for d in DEDOS]
    cerrados = [dedo_cerrado(p, *d) for d in DEDOS]
    tam_palma = dist(p[WRIST], p[9]) or 1.0

    # Índice apuntando hacia arriba: punta por encima del PIP y del MCP, más vertical que horizontal
    mcp, pip, tip = (p[i] for i in DEDOS[0])
    indice_arriba = (
        extendidos[0]
        and tip[1] < pip[1] < mcp[1]
        and abs(tip[1] - mcp[1]) > abs(tip[0] - mcp[0])
    )
    gesto_lentes = indice_arriba and all(cerrados[1:])

    # 1. Gatito con lentes: solo el índice levantado ☝️🤓 (tiene prioridad aunque esté frente a la cara)
    if gesto_lentes:
        return "LENTES"

    # 2. Chango rascándose: la mano está sobre la cabeza/frente/sien 🐵
    if mano_en_cabeza(p, cara):
        return "DUDA"

    # 3. Gatito puñetazo: los 4 dedos cerrados y el pulgar recogido sobre el puño 👊
    pulgar_recogido = min(dist(p[THUMB_TIP], p[i]) for i in (5, 6, 9, 10)) < tam_palma * 0.8
    if all(cerrados) and pulgar_recogido:
        return "PUNO"

    # Mano relajada / posición neutral
    return "NINGUNO"


class Estabilizador:
    """Solo cambia el gesto mostrado cuando el nuevo se repite N frames seguidos."""

    def __init__(self, n):
        self.n = n
        self.mostrado = "NINGUNO"
        self.candidato = "NINGUNO"
        self.conteo = 0

    def actualizar(self, gesto):
        if gesto == self.mostrado:
            self.candidato, self.conteo = gesto, 0
        elif gesto == self.candidato:
            self.conteo += 1
            if self.conteo >= self.n:
                self.mostrado, self.conteo = gesto, 0
        else:
            self.candidato, self.conteo = gesto, 1
        return self.mostrado


def detectar_cara(rgb_frame, ancho, alto):
    res = face_detector.process(rgb_frame)
    if not res.detections:
        return None
    # La cara más grande (la más cercana a la cámara)
    caja = max((d.location_data.relative_bounding_box for d in res.detections),
               key=lambda b: b.width * b.height)
    return (caja.xmin * ancho, caja.ymin * alto, caja.width * ancho, caja.height * alto)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo acceder a la cámara. Verifica que esté conectada y disponible.")

VENTANA_CAMARA = "Camara IA"
VENTANA_MEME = "Meme Reaccion"

estabilizador = Estabilizador(FRAMES_ESTABLES)
ultima_cara, frames_sin_cara = None, 0

print("Iniciando IA... Presiona ESC para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Espejo 🪞
    alto, ancho = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)

    cara = detectar_cara(rgb_frame, ancho, alto)
    if cara is not None:
        ultima_cara, frames_sin_cara = cara, 0
    else:
        frames_sin_cara += 1
        if frames_sin_cara <= FRAMES_MEMORIA_CARA:
            cara = ultima_cara

    gesto_crudo = "NINGUNO"
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        puntos = a_pixeles(hand_landmarks.landmark, ancho, alto)
        gesto_crudo = clasificar_gesto(puntos, cara)

    gesto_actual = estabilizador.actualizar(gesto_crudo)
    meme_actual, color = MEMES[gesto_actual]

    if cara is not None:
        x, y, w, h = (int(v) for v in cara)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 1)

    # Retroalimentación en pantalla
    cv2.putText(frame, f"Gesto: {gesto_actual}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow(VENTANA_CAMARA, frame)
    cv2.imshow(VENTANA_MEME, meme_actual)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
face_detector.close()
