import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # ← protobuf crash fix

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import datetime
from collections import deque

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
SAVE_DIR        = "air_drawings"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_UNDO        = 20
BRUSH_MIN       = 2
BRUSH_MAX       = 30
GLOW_SIGMA      = 4
AUTO_SAVE_SEC   = 30
MAX_PARTICLES   = 150
CONSTELLATION_R = 160
MAX_CONST_PTS   = 60
ERASE_RADIUS    = 28
FIST_HOLD_SEC   = 1.0
VELOCITY_SMOOTH = 0.25

# ══════════════════════════════════════════════
#  MEDIAPIPE
# ══════════════════════════════════════════════
mp_hands     = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
face_mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ══════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════
class Kalman1D:
    def __init__(self, q=0.008, r=0.5):
        self.x = 0.0
        self.P = 1.0
        self.Q = q
        self.R = r

    def reset(self, value):
        self.x = float(value)
        self.P = 1.0

    def update(self, z):
        P_ = self.P + self.Q
        K  = P_ / (P_ + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * P_
        return self.x

kx = Kalman1D()
ky = Kalman1D()

# ══════════════════════════════════════════════
#  PARTICLE SYSTEM
# ══════════════════════════════════════════════
class Particle:
    def __init__(self, x, y, color):
        self.x     = x + np.random.uniform(-8, 8)
        self.y     = y + np.random.uniform(-8, 8)
        angle      = np.random.uniform(0, 2 * math.pi)
        speed      = np.random.uniform(0.5, 3.5)
        self.vx    = speed * math.cos(angle)
        self.vy    = speed * math.sin(angle)
        self.life  = 1.0
        self.decay = np.random.uniform(0.04, 0.13)
        self.color = color
        self.size  = np.random.randint(2, 5)

    def step(self):
        self.x   += self.vx
        self.y   += self.vy
        self.vy  += 0.12
        self.vx  *= 0.97
        self.life -= self.decay
        return self.life > 0

    def draw(self, img):
        a = self.life
        c = tuple(int(ch * a) for ch in self.color)
        cv2.circle(img, (int(self.x), int(self.y)), self.size, c, -1)

particles = []

# ══════════════════════════════════════════════
#  COLOR PALETTE (12 colors)
# ══════════════════════════════════════════════
PALETTE = [
    ("WHITE",   (255, 255, 255)),
    ("CYAN",    (255, 255,   0)),
    ("MAGENTA", (255,   0, 255)),
    ("YELLOW",  (  0, 255, 255)),
    ("GREEN",   (  0, 255,   0)),
    ("RED",     (  0,   0, 255)),
    ("BLUE",    (255,  80,   0)),
    ("ORANGE",  (  0, 165, 255)),
    ("PINK",    (147,  20, 255)),
    ("LIME",    (  0, 255, 128)),
    ("GOLD",    (  0, 215, 255)),
    ("VIOLET",  (211,   0, 148)),
]
color_idx    = 0
color_names  = [p[0] for p in PALETTE]
color_values = [p[1] for p in PALETTE]

def rainbow_color(t):
    hue = int((t * 60) % 180)
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

# ══════════════════════════════════════════════
#  DRAWING MODES
# ══════════════════════════════════════════════
MODES    = ["FREEHAND", "SYMMETRY", "RADIAL", "CONSTELLATION", "GLOW", "RAINBOW", "SPEED_ART"]
mode_idx = 0

# ══════════════════════════════════════════════
#  GESTURE DETECTION
# ══════════════════════════════════════════════
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

def finger_states(lms, hand_label="Right"):
    ext = []
    if hand_label == "Right":
        ext.append(1 if lms[4].x < lms[3].x else 0)
    else:
        ext.append(1 if lms[4].x > lms[3].x else 0)
    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        ext.append(1 if lms[tip].y < lms[pip].y else 0)
    return ext

GESTURE_MAP = {
    (0,1,0,0,0): "DRAW",
    (0,1,1,0,0): "ERASE",
    (0,0,0,0,0): "FIST",
    (1,1,1,1,1): "OPEN_PALM",
    (0,1,1,1,1): "OPEN_PALM",
    (1,0,0,0,0): "THUMBS_UP",
    (0,1,0,0,1): "ROCK",
    (0,1,1,1,0): "THREE",
}

def classify(fingers):
    return GESTURE_MAP.get(tuple(fingers), f"OTHER_{sum(fingers)}")

# ══════════════════════════════════════════════
#  EFFECTS
# ══════════════════════════════════════════════
def apply_glow(canvas, sigma=GLOW_SIGMA):
    blur = cv2.GaussianBlur(canvas, (0, 0), sigma)
    return cv2.addWeighted(canvas, 1.0, blur, 0.6, 0)

def detect_emotion(lms):
    try:
        mt = lms[13]; mb = lms[14]
        ml = lms[61]; mr = lms[291]
        mw = abs(mr.x - ml.x) + 1e-6
        mh = abs(mb.y - mt.y)
        open_ratio   = mh / mw
        avg_corner_y = (ml.y + mr.y) / 2
        center_y     = (mt.y + mb.y) / 2
        if open_ratio > 0.20:
            return "SURPRISED", (0, 255, 255)
        elif avg_corner_y < center_y - 0.004:
            return "HAPPY",    (0, 255, 0)
        elif avg_corner_y > center_y + 0.004:
            return "SAD",      (255, 50, 50)
        else:
            return "NEUTRAL",  (200, 200, 200)
    except Exception:
        return "NEUTRAL", (200, 200, 200)

def draw_constellation(canvas, pts, color):
    for i, p1 in enumerate(pts):
        cv2.circle(canvas, p1, 3, color, -1)
        for p2 in pts[i+1:]:
            d = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if d < CONSTELLATION_R:
                intensity = int(180 * (1 - d / CONSTELLATION_R))
                lc = tuple(min(255, int(c * intensity / 180)) for c in color)
                cv2.line(canvas, p1, p2, lc, 1)

def radial_draw(canvas, px, py, ppx, ppy, cx, cy, color, thickness):
    pts = [
        ((px, py),          (ppx, ppy)),
        ((2*cx-px, py),     (2*cx-ppx, ppy)),
        ((px, 2*cy-py),     (ppx, 2*cy-ppy)),
        ((2*cx-px,2*cy-py), (2*cx-ppx,2*cy-ppy)),
    ]
    for (a, b) in pts:
        cv2.line(canvas, a, b, color, thickness)

# ══════════════════════════════════════════════
#  FACE WINDOW
# ══════════════════════════════════════════════
def draw_face_window(frame, face_result, fw=150, fh=150):
    h, w = frame.shape[:2]
    if not face_result or not face_result.multi_face_landmarks:
        return None
    lms = face_result.multi_face_landmarks[0].landmark
    xs  = [int(l.x * w) for l in lms]
    ys  = [int(l.y * h) for l in lms]
    x1  = max(0, min(xs)-20); x2 = min(w, max(xs)+20)
    y1  = max(0, min(ys)-20); y2 = min(h, max(ys)+20)
    if x2 <= x1 or y2 <= y1:
        return None
    crop  = frame[y1:y2, x1:x2].copy()
    small = cv2.resize(crop, (fw, fh))
    for lm in lms:
        px = int((lm.x*w - x1) * fw / (x2-x1))
        py = int((lm.y*h - y1) * fh / (y2-y1))
        if 0 <= px < fw and 0 <= py < fh:
            cv2.circle(small, (px, py), 1, (0, 255, 100), -1)
    frame[10:10+fh, 10:10+fw] = small
    cv2.rectangle(frame, (8, 8), (12+fw, 12+fh), (0, 255, 255), 2)
    cv2.putText(frame, "FACE MESH", (10, fh+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    return lms

# ══════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════
def draw_hud(frame, gesture, fps, emotion, em_color, brush,
             mode_name, color_name, draw_color, undo_cnt, paused, vel):
    h, w = frame.shape[:2]
    px   = w - 195

    ovl = frame.copy()
    cv2.rectangle(ovl, (px-8, 0), (w, 320), (5, 5, 15), -1)
    cv2.addWeighted(ovl, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (px-8, 0), (w, 320), (0,200,200), 1)

    def put(text, y, color=(210,210,210), scale=0.5, bold=1):
        cv2.putText(frame, text, (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

    fps_c = (0,255,0) if fps>25 else (0,165,255) if fps>15 else (0,0,255)
    put(f"FPS  {fps:4.0f}",           24,  fps_c, 0.55, 2)
    put(f"GESTURE: {gesture}",        52,  (0,255,255), 0.45)
    put(f"MODE   : {mode_name}",      76,  (255,200,0), 0.45)
    put(f"VEL    : {vel:4.0f}",       100, (180,180,180), 0.42)
    put("COLOR",                      128, (180,180,180), 0.42)
    cv2.rectangle(frame, (px,133), (px+55,153), draw_color, -1)
    cv2.rectangle(frame, (px,133), (px+55,153), (255,255,255), 1)
    put(color_name,                   168, (220,220,220), 0.42)
    put(f"BRUSH {brush}px",           193, (180,180,180), 0.42)
    bar_w = int(185 * brush / BRUSH_MAX)
    cv2.rectangle(frame, (px,198), (px+185,206), (40,40,40), -1)
    cv2.rectangle(frame, (px,198), (px+bar_w,206), draw_color, -1)
    put(f"MOOD: {emotion}",           228, em_color, 0.42)
    put(f"UNDO: {undo_cnt}/{MAX_UNDO}", 254, (150,150,150), 0.42)
    if paused:
        put("PAUSED", 286, (0,200,255), 0.6, 2)

    bh = 58
    ovl2 = frame.copy()
    cv2.rectangle(ovl2, (0, h-bh), (px-8, h), (5,5,15), -1)
    cv2.addWeighted(ovl2, 0.5, frame, 0.5, 0, frame)
    cv2.line(frame, (0,h-bh), (px-8,h-bh), (0,200,200), 1)

    shortcuts = [
        ("INDEX","Draw"),("PEACE","Erase"),("FIST 1s","Clear"),
        ("PALM","Pause"),("THUMB","Save"),("ROCK","Mode"),
        ("3 FIN","Color"),("Z","Undo"),("Q","Quit"),
    ]
    slot = (px - 8) // len(shortcuts)
    for i, (k, v) in enumerate(shortcuts):
        x = i*slot + 5
        cv2.putText(frame, k, (x,h-38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, v, (x,h-18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (255,255,255), 1, cv2.LINE_AA)

    cv2.putText(frame, "  AIR CANVAS PRO  v3.0", (170,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

def draw_skeleton(frame, lms, w, h):
    for conn in mp_hands.HAND_CONNECTIONS:
        a, b = conn
        cv2.line(frame,
                 (int(lms[a].x*w), int(lms[a].y*h)),
                 (int(lms[b].x*w), int(lms[b].y*h)),
                 (40,40,60), 1)
    for lm in lms:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0,200,200), -1)

def save_canvas(canvas, prefix="save"):
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.png")
    cv2.imwrite(path, canvas)
    return path

# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════
cap    = cv2.VideoCapture(0)
canvas = None

prev_x, prev_y   = None, None
undo_stack        = deque(maxlen=MAX_UNDO)
gesture           = "NONE"
prev_gesture      = "NONE"
paused            = False
fist_t            = None
mode_cd           = 0
color_cd          = 0
saved_flash       = 0
emotion_str       = "NEUTRAL"
emotion_color     = (200, 200, 200)
vel_smooth        = 0.0
brush_size        = 8
frame_idx         = 0
fps               = 30.0
fps_t0            = time.time()
last_auto_save    = time.time()
cx, cy            = 0, 0
constellation_pts = []

print("╔══════════════════════════════════╗")
print("║   AIR CANVAS PRO  v3.0 ULTRA    ║")
print("╚══════════════════════════════════╝")
print(f"Drawings saved to: {os.path.abspath(SAVE_DIR)}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame_idx += 1
    if frame_idx % 10 == 0:
        fps    = 10.0 / max(time.time() - fps_t0, 1e-6)
        fps_t0 = time.time()

    rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_mesh_detector.process(rgb)
    hand_result = hands.process(rgb)

    face_lms = draw_face_window(frame, face_result)
    if face_lms:
        emotion_str, emotion_color = detect_emotion(face_lms)

    if hand_result.multi_hand_landmarks:
        lms_list   = hand_result.multi_hand_landmarks[0].landmark
        hand_label = hand_result.multi_handedness[0].classification[0].label

        raw_x = int(lms_list[8].x * w)
        raw_y = int(lms_list[8].y * h)
        sx    = int(kx.update(raw_x))
        sy    = int(ky.update(raw_y))

        if prev_x is not None:
            raw_vel    = math.hypot(sx-prev_x, sy-prev_y)
            vel_smooth = VELOCITY_SMOOTH*raw_vel + (1-VELOCITY_SMOOTH)*vel_smooth
        else:
            vel_smooth = 0.0

        fingers = finger_states(lms_list, hand_label)
        gesture = classify(fingers)

        if gesture != prev_gesture:
            fist_t = None

        draw_skeleton(frame, lms_list, w, h)

        if not paused or gesture == "OPEN_PALM":

            if gesture == "DRAW":
                wx   = lms_list[0].x; wy = lms_list[0].y
                ix   = lms_list[8].x; iy = lms_list[8].y
                span = math.hypot(wx-ix, wy-iy)

                if MODES[mode_idx] == "SPEED_ART":
                    brush_size = int(np.clip(BRUSH_MAX - vel_smooth*0.8, BRUSH_MIN, BRUSH_MAX))
                else:
                    brush_size = int(np.clip(span*90, BRUSH_MIN, BRUSH_MAX))

                draw_color = color_values[color_idx] if MODES[mode_idx] != "RAINBOW" \
                             else rainbow_color(time.time())

                cv2.circle(frame, (sx,sy), brush_size, draw_color, 2)
                cv2.circle(frame, (sx,sy), 3, (255,255,255), -1)

                if prev_x is not None:
                    if frame_idx % 12 == 0:
                        undo_stack.append(canvas.copy())

                    m = MODES[mode_idx]
                    if m in ("FREEHAND","GLOW","RAINBOW","SPEED_ART"):
                        cv2.line(canvas,(prev_x,prev_y),(sx,sy),draw_color,brush_size)
                    elif m == "SYMMETRY":
                        cv2.line(canvas,(prev_x,prev_y),(sx,sy),draw_color,brush_size)
                        cv2.line(canvas,(w-prev_x,prev_y),(w-sx,sy),draw_color,brush_size)
                    elif m == "RADIAL":
                        radial_draw(canvas,sx,sy,prev_x,prev_y,cx,cy,draw_color,brush_size)
                    elif m == "CONSTELLATION":
                        constellation_pts.append((sx,sy))
                        if len(constellation_pts) > MAX_CONST_PTS:
                            constellation_pts.pop(0)

                    if len(particles) < MAX_PARTICLES:
                        for _ in range(4):
                            particles.append(Particle(sx, sy, draw_color))

                prev_x, prev_y = sx, sy

            elif gesture == "ERASE":
                cv2.circle(frame, (sx,sy), ERASE_RADIUS, (128,128,128), 2)
                cv2.circle(frame, (sx,sy), 4, (255,255,255), -1)
                if frame_idx % 8 == 0:
                    undo_stack.append(canvas.copy())
                cv2.circle(canvas, (sx,sy), ERASE_RADIUS, (0,0,0), -1)
                prev_x, prev_y = None, None

            elif gesture == "FIST":
                if fist_t is None:
                    fist_t = time.time()
                held  = time.time() - fist_t
                angle = int(360 * min(held/FIST_HOLD_SEC, 1.0))
                cv2.ellipse(frame,(sx,sy),(32,32),-90,0,angle,(0,0,255),3)
                cv2.putText(frame,"HOLD!",(sx-22,sy+5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
                if held >= FIST_HOLD_SEC:
                    undo_stack.append(canvas.copy())
                    canvas = np.zeros_like(frame)
                    constellation_pts.clear()
                    fist_t = None
                prev_x, prev_y = None, None

            elif gesture == "OPEN_PALM":
                if prev_gesture != "OPEN_PALM":
                    paused = not paused
                prev_x, prev_y = None, None

            elif gesture == "THUMBS_UP":
                if prev_gesture != "THUMBS_UP":
                    p = save_canvas(canvas, "manual")
                    saved_flash = 70
                    print(f"Saved: {p}")
                prev_x, prev_y = None, None

            elif gesture == "ROCK":
                if mode_cd <= 0 and prev_gesture != "ROCK":
                    mode_idx = (mode_idx+1) % len(MODES)
                    constellation_pts.clear()
                    mode_cd = 35
                prev_x, prev_y = None, None

            elif gesture == "THREE":
                if color_cd <= 0 and prev_gesture != "THREE":
                    color_idx = (color_idx+1) % len(PALETTE)
                    color_cd  = 35
                prev_x, prev_y = None, None

            else:
                prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        prev_gesture = gesture

    else:
        prev_x, prev_y = None, None
        prev_gesture   = "NONE"
        gesture        = "NONE"
        fist_t         = None
        vel_smooth     = 0.0
        kx.reset(0); ky.reset(0)

    mode_cd  = max(0, mode_cd-1)
    color_cd = max(0, color_cd-1)

    if MODES[mode_idx] == "CONSTELLATION" and len(constellation_pts) > 1:
        draw_constellation(canvas, constellation_pts, color_values[color_idx])

    particles[:] = [p for p in particles if p.step()]
    particle_layer = np.zeros_like(frame)
    for p in particles:
        p.draw(particle_layer)

    display_canvas = apply_glow(canvas) \
        if MODES[mode_idx] in ("GLOW","RAINBOW","CONSTELLATION") \
        else canvas

    output = cv2.add(frame, display_canvas)
    output = cv2.add(output, particle_layer)

    draw_hud(output, gesture, fps, emotion_str, emotion_color,
             brush_size, MODES[mode_idx],
             color_names[color_idx], color_values[color_idx],
             len(undo_stack), paused, vel_smooth)

    if paused:
        ov = output.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,30), -1)
        cv2.addWeighted(ov, 0.35, output, 0.65, 0, output)
        cv2.putText(output, "PAUSED  -  Open Palm to Resume",
                    (w//2-230, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

    if saved_flash > 0:
        cv2.putText(output, "SAVED!", (w//2-55, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,80), 3, cv2.LINE_AA)
        saved_flash -= 1

    if time.time() - last_auto_save > AUTO_SAVE_SEC:
        save_canvas(canvas, "autosave")
        last_auto_save = time.time()

    cv2.imshow("Air Canvas Pro v3.0", output)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'):
        save_canvas(canvas, "final")
        break
    elif key == ord('z'):
        if undo_stack:
            canvas = undo_stack.pop()
    elif key == ord('c'):
        undo_stack.append(canvas.copy())
        canvas = np.zeros_like(frame)
        constellation_pts.clear()
    elif key == ord('s'):
        p = save_canvas(canvas, "manual")
        saved_flash = 70
        print(f"Saved: {p}")
    elif key == ord('m'):
        mode_idx = (mode_idx+1) % len(MODES)
        constellation_pts.clear()

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Drawings saved in: {os.path.abspath(SAVE_DIR)}")
