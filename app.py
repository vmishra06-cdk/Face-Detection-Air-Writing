"""
╔══════════════════════════════════════════════════════════════╗
║        SMART AIR WHITEBOARD  v4.0  —  TEACHER EDITION       ║
╠══════════════════════════════════════════════════════════════╣
║  GESTURES:                                                   ║
║   ☝  Index only      → DRAW / WRITE                         ║
║   ✌  Peace sign      → ERASE                                ║
║   ✊  Fist (hold 1s) → CLEAR board                          ║
║   🖐  Open palm       → PAUSE / RESUME                       ║
║   👍  Thumbs up       → SAVE snapshot                        ║
║   🤟  Rock sign       → Cycle TOOL                           ║
║   3 fingers           → Cycle COLOR                          ║
║                                                              ║
║  KEYBOARD:                                                   ║
║   Z → Undo       C → Clear      S → Save                    ║
║   W → Whiteboard / Dark toggle                               ║
║   R → Recognize text (OCR)      T → Timer                    ║
║   1-7 → Quick tool select        Q → Quit                    ║
║                                                              ║
║  TOOLS: MARKER, CHALK, HIGHLIGHTER, POINTER,                 ║
║         SYMMETRY, RADIAL, CONSTELLATION                      ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── MUST BE FIRST (fixes protobuf crash) ──
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import datetime
import threading
from collections import deque

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    print("[OK] OCR (pytesseract) available")
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] pytesseract not found — install with: pip install pytesseract Pillow")
    print("       Also install Tesseract: https://github.com/tesseract-ocr/tesseract")

# ══════════════════════════════════════════════
#  MEDIAPIPE INIT
# ══════════════════════════════════════════════
try:
    mp_hands     = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    )
    face_mesh_detector = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[OK] MediaPipe loaded")
except AttributeError as e:
    print(f"[ERROR] MediaPipe solutions not available: {e}")
    print("  Run: pip install 'mediapipe<=0.10.31'")
    raise SystemExit(1)

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
SAVE_DIR        = "whiteboard_saves"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_UNDO        = 25
BRUSH_MIN       = 3
BRUSH_MAX       = 35
GLOW_SIGMA      = 4
AUTO_SAVE_SEC   = 60
MAX_PARTICLES   = 80
CONSTELLATION_R = 150
MAX_CONST_PTS   = 50
ERASE_RADIUS    = 35
FIST_HOLD_SEC   = 1.0
VELOCITY_SMOOTH = 0.25
OCR_PAUSE_SEC   = 2.5       # seconds of no drawing → trigger OCR
POINTER_RADIUS  = 18

# ══════════════════════════════════════════════
#  WHITEBOARD THEMES
# ══════════════════════════════════════════════
THEME_WHITEBOARD = {
    "name":       "WHITEBOARD",
    "bg":         (240, 240, 235),   # off-white
    "grid":       (220, 220, 215),
    "hud_bg":     (255, 255, 255),
    "hud_text":   (30,  30,  30),
    "hud_accent": (0,   120, 200),
    "cursor":     (50,  50,  50),
}
THEME_DARK = {
    "name":       "DARK BOARD",
    "bg":         (18,  18,  25),
    "grid":       (30,  30,  40),
    "hud_bg":     (10,  10,  18),
    "hud_text":   (220, 220, 220),
    "hud_accent": (0,   220, 200),
    "cursor":     (255, 255, 255),
}
THEME_BLACKBOARD = {
    "name":       "BLACKBOARD",
    "bg":         (28,  52,  38),    # dark green
    "grid":       (38,  65,  48),
    "hud_bg":     (20,  40,  28),
    "hud_text":   (240, 240, 220),
    "hud_accent": (255, 220, 80),
    "cursor":     (255, 255, 200),
}
THEMES    = [THEME_WHITEBOARD, THEME_DARK, THEME_BLACKBOARD]
theme_idx = 0

# ══════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════
TOOLS = ["MARKER", "CHALK", "HIGHLIGHTER", "POINTER",
         "SYMMETRY", "RADIAL", "CONSTELLATION"]
tool_idx = 0

# ══════════════════════════════════════════════
#  COLOR PALETTES  (whiteboard vs dark)
# ══════════════════════════════════════════════
PALETTE_DARK = [
    ("WHITE",   (255, 255, 255)),
    ("YELLOW",  (  0, 255, 255)),
    ("CYAN",    (255, 255,   0)),
    ("GREEN",   (  0, 255,   0)),
    ("PINK",    (147,  20, 255)),
    ("ORANGE",  (  0, 165, 255)),
    ("RED",     (  0,   0, 255)),
    ("SKY",     (255, 200,   0)),
]
PALETTE_LIGHT = [
    ("BLACK",   (  0,   0,   0)),
    ("BLUE",    (180,   0,   0)),
    ("RED",     (  0,   0, 200)),
    ("GREEN",   ( 20, 140,  20)),
    ("PURPLE",  (150,   0, 150)),
    ("ORANGE",  (  0, 100, 220)),
    ("BROWN",   ( 30,  60, 100)),
    ("TEAL",    (120, 120,   0)),
]

color_idx = 0

def get_palette():
    if THEMES[theme_idx]["name"] == "WHITEBOARD":
        return PALETTE_LIGHT
    return PALETTE_DARK

# ══════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════
class Kalman1D:
    def __init__(self, q=0.006, r=0.4):
        self.x = 0.0; self.P = 1.0; self.Q = q; self.R = r
    def reset(self, v=0): self.x = float(v); self.P = 1.0
    def update(self, z):
        P_ = self.P + self.Q; K = P_ / (P_ + self.R)
        self.x += K * (z - self.x); self.P = (1 - K) * P_; return self.x

kx = Kalman1D(); ky = Kalman1D()

# ══════════════════════════════════════════════
#  PARTICLE
# ══════════════════════════════════════════════
class Particle:
    def __init__(self, x, y, color):
        self.x = float(x) + np.random.uniform(-5, 5)
        self.y = float(y) + np.random.uniform(-5, 5)
        a = np.random.uniform(0, 2*math.pi)
        s = np.random.uniform(0.4, 2.5)
        self.vx = s*math.cos(a); self.vy = s*math.sin(a)
        self.life = 1.0; self.decay = np.random.uniform(0.05, 0.14)
        self.color = color; self.size = int(np.random.randint(2, 4))

    def step(self):
        self.x += self.vx; self.y += self.vy
        self.vy += 0.08; self.vx *= 0.97
        self.life -= self.decay; return self.life > 0

    def draw(self, img):
        a = max(0.0, self.life)
        c = tuple(min(255, int(ch * a)) for ch in self.color)
        ix, iy = int(self.x), int(self.y)
        h, w = img.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(img, (ix, iy), self.size, c, -1)

particles = []

# ══════════════════════════════════════════════
#  OCR ENGINE
# ══════════════════════════════════════════════
ocr_text         = ""
ocr_corrected    = ""
ocr_running      = False
ocr_flash        = 0
last_draw_time   = time.time()
ocr_triggered    = False

def run_ocr(canvas, theme):
    global ocr_text, ocr_corrected, ocr_running, ocr_flash
    if not OCR_AVAILABLE:
        ocr_running = False; return
    try:
        # Preprocess for better OCR
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # For whiteboard (light bg), invert to get dark text on white
        if theme["name"] == "WHITEBOARD":
            gray = cv2.bitwise_not(gray)

        # Threshold
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # Check if anything is drawn
        if thresh.sum() < 5000:
            ocr_running = False; return

        # Find bounding box of drawn content
        coords = cv2.findNonZero(thresh)
        if coords is None:
            ocr_running = False; return
        x, y, w, h = cv2.boundingRect(coords)
        pad = 20
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(thresh.shape[1], x+w+pad)
        y2 = min(thresh.shape[0], y+h+pad)
        crop = thresh[y1:y2, x1:x2]

        # Scale up for better OCR
        scale = max(1.0, 600 / max(crop.shape))
        crop_big = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        # Dilate slightly to connect strokes
        kernel = np.ones((2,2), np.uint8)
        crop_big = cv2.dilate(crop_big, kernel, iterations=1)

        pil_img = Image.fromarray(crop_big)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(pil_img, config=custom_config).strip()

        if text:
            ocr_text = text
            ocr_corrected = text   # could plug in a spell-checker here
            ocr_flash = 120
            print(f"[OCR] Recognized: {repr(text)}")
        else:
            ocr_text = "(couldn't read — write bigger & clearer)"
            ocr_flash = 80

    except Exception as e:
        ocr_text = f"(OCR error: {e})"
        ocr_flash = 60
    finally:
        ocr_running = False


# ══════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════
def draw_chalk_line(canvas, p1, p2, color, thickness):
    """Simulate chalk: rough edges, variable opacity."""
    cv2.line(canvas, p1, p2, color, thickness)
    for _ in range(3):
        ox = np.random.randint(-2, 3)
        oy = np.random.randint(-2, 3)
        alpha = np.random.uniform(0.2, 0.5)
        faint = tuple(int(c * alpha) for c in color)
        cv2.line(canvas,
                 (p1[0]+ox, p1[1]+oy),
                 (p2[0]+ox, p2[1]+oy),
                 faint, max(1, thickness-2))

def draw_highlighter_line(canvas, p1, p2, color, thickness):
    """Transparent wide highlight stroke."""
    overlay = canvas.copy()
    cv2.line(overlay, p1, p2, color, thickness * 3)
    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

def draw_constellation(canvas, pts, color):
    for i, p1 in enumerate(pts):
        cv2.circle(canvas, p1, 3, color, -1)
        for p2 in pts[i+1:]:
            d = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if d < CONSTELLATION_R:
                a  = 1 - d / CONSTELLATION_R
                lc = tuple(min(255, int(c * a * 0.7)) for c in color)
                cv2.line(canvas, p1, p2, lc, 1)

def radial_draw(canvas, px, py, ppx, ppy, cx, cy, color, t, tool):
    for (a, b) in [((px,py),(ppx,ppy)), ((2*cx-px,py),(2*cx-ppx,ppy)),
                   ((px,2*cy-py),(ppx,2*cy-ppy)), ((2*cx-px,2*cy-py),(2*cx-ppx,2*cy-ppy))]:
        cv2.line(canvas, a, b, color, t)

def apply_glow(canvas, sigma=GLOW_SIGMA):
    blur = cv2.GaussianBlur(canvas, (0,0), sigma)
    return cv2.addWeighted(canvas, 1.0, blur, 0.5, 0)

def draw_grid(frame, theme):
    h, w = frame.shape[:2]
    gc = theme["grid"]
    for x in range(0, w, 60):
        cv2.line(frame, (x,0), (x,h), gc, 1)
    for y in range(0, h, 60):
        cv2.line(frame, (0,y), (w,y), gc, 1)

# ══════════════════════════════════════════════
#  GESTURE
# ══════════════════════════════════════════════
TIP_IDS = [4,8,12,16,20]; PIP_IDS = [3,6,10,14,18]
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

def finger_states(lms, label="Right"):
    ext = [1 if lms[4].x < lms[3].x else 0] if label=="Right" \
          else [1 if lms[4].x > lms[3].x else 0]
    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        ext.append(1 if lms[tip].y < lms[pip].y else 0)
    return ext

def classify(f):
    return GESTURE_MAP.get(tuple(f), f"OTHER_{sum(f)}")

# ══════════════════════════════════════════════
#  FACE WINDOW + EMOTION
# ══════════════════════════════════════════════
def detect_emotion(lms):
    try:
        mt=lms[13]; mb=lms[14]; ml=lms[61]; mr=lms[291]
        mw=abs(mr.x-ml.x)+1e-6; mh=abs(mb.y-mt.y)
        r=mh/mw; ay=(ml.y+mr.y)/2; cy2=(mt.y+mb.y)/2
        if r>0.20: return "SURPRISED",(0,200,255)
        elif ay<cy2-0.004: return "HAPPY",(0,200,80)
        elif ay>cy2+0.004: return "SAD",(100,100,255)
        else: return "NEUTRAL",(160,160,160)
    except: return "NEUTRAL",(160,160,160)

def draw_face_window(frame, face_result, fw=130, fh=130):
    h, w = frame.shape[:2]
    if not face_result or not face_result.multi_face_landmarks:
        return None
    lms = face_result.multi_face_landmarks[0].landmark
    xs  = [int(l.x*w) for l in lms]; ys = [int(l.y*h) for l in lms]
    x1=max(0,min(xs)-15); x2=min(w,max(xs)+15)
    y1=max(0,min(ys)-15); y2=min(h,max(ys)+15)
    if x2<=x1 or y2<=y1: return None
    crop  = frame[y1:y2, x1:x2].copy()
    small = cv2.resize(crop, (fw, fh))
    for lm in lms:
        px=int((lm.x*w-x1)*fw/max(x2-x1,1))
        py=int((lm.y*h-y1)*fh/max(y2-y1,1))
        if 0<=px<fw and 0<=py<fh:
            cv2.circle(small,(px,py),1,(0,200,120),-1)
    frame[10:10+fh, 10:10+fw] = small
    cv2.rectangle(frame,(8,8),(12+fw,12+fh),(0,200,200),2)
    return lms

def draw_skeleton(frame, lms, w, h):
    for conn in mp_hands.HAND_CONNECTIONS:
        a,b=conn
        cv2.line(frame,(int(lms[a].x*w),int(lms[a].y*h)),
                       (int(lms[b].x*w),int(lms[b].y*h)),(60,60,80),1)
    for lm in lms:
        cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),3,(0,200,200),-1)

# ══════════════════════════════════════════════
#  TIMER
# ══════════════════════════════════════════════
timer_start  = None
timer_active = False
TIMER_MINS   = 5   # default 5-minute class timer

def toggle_timer():
    global timer_start, timer_active
    if timer_active:
        timer_active = False; timer_start = None
    else:
        timer_active = True; timer_start = time.time()

def get_timer_str():
    if not timer_active or timer_start is None: return ""
    elapsed = int(time.time() - timer_start)
    total   = TIMER_MINS * 60
    left    = max(0, total - elapsed)
    m, s    = divmod(left, 60)
    return f"{m:02d}:{s:02d}"

# ══════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════
def save_board(canvas, prefix="save"):
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.png")
    cv2.imwrite(path, canvas)
    return path

# ══════════════════════════════════════════════
#  HUD  (Teacher Toolbar)
# ══════════════════════════════════════════════
def draw_teacher_hud(frame, gesture, fps, emotion, em_color,
                     brush, tool_name, color_name, draw_color,
                     undo_cnt, paused, vel, theme, timer_str, ocr_text):
    h, w  = frame.shape[:2]
    T     = theme
    px    = w - 210
    PANEL = 380

    # Right panel
    ovl = frame.copy()
    cv2.rectangle(ovl, (px-8,0), (w,PANEL), T["hud_bg"], -1)
    cv2.addWeighted(ovl, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (px-8,0), (w,PANEL), T["hud_accent"], 1)

    def put(text, y, color=None, scale=0.48, bold=1):
        color = color or T["hud_text"]
        cv2.putText(frame, text, (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "SMART WHITEBOARD", (px-2, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, T["hud_accent"], 2, cv2.LINE_AA)
    cv2.putText(frame, "TEACHER EDITION", (px+8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, T["hud_text"], 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (px-8,50), (w,50), T["hud_accent"], 1)

    fps_c = (0,200,0) if fps>25 else (0,165,255) if fps>15 else (0,0,220)
    put(f"FPS:  {fps:4.0f}",              68,  fps_c, 0.44, 2)
    put(f"GESTURE: {gesture}",            90,  T["hud_accent"], 0.40)
    put(f"MOOD:    {emotion}",            110, em_color, 0.40)

    cv2.line(frame,(px-8,120),(w,120),T["grid"] if "grid" in T else (60,60,60),1)

    # Tool display
    put("TOOL", 138, T["hud_text"], 0.38)
    put(f"  {tool_name}", 158, T["hud_accent"], 0.50, 2)

    # Color swatch
    put("COLOR", 182, T["hud_text"], 0.38)
    cv2.rectangle(frame,(px,188),(px+60,208),draw_color,-1)
    cv2.rectangle(frame,(px,188),(px+60,208),(200,200,200),1)
    put(f"  {color_name}", 222, T["hud_text"], 0.38)

    # Brush bar
    put(f"BRUSH: {brush}px", 242, T["hud_text"], 0.38)
    bw = int(198 * brush / BRUSH_MAX)
    cv2.rectangle(frame,(px,248),(px+198,255),(60,60,60),-1)
    cv2.rectangle(frame,(px,248),(px+bw,255),draw_color,-1)

    # Theme
    put(f"THEME: {T['name']}", 272, T["hud_text"], 0.38)

    # Undo
    put(f"UNDO:  {undo_cnt}/{MAX_UNDO}", 292, T["hud_text"], 0.38)

    # Velocity
    put(f"VEL:   {vel:4.0f} px/f", 312, T["hud_text"], 0.36)

    # Timer
    if timer_str:
        tc = (0,200,0) if int(timer_str.split(":")[0])>1 else (0,0,220)
        put(f"TIMER: {timer_str}", 338, tc, 0.50, 2)

    # Paused
    if paused:
        cv2.rectangle(frame,(px,355),(w,375),(0,0,150),-1)
        put(">> PAUSED — OPEN PALM TO RESUME", 370, (0,220,255), 0.35, 2)

    # ── OCR result panel at bottom of screen ──
    if ocr_text and ocr_flash_ref[0] > 0:
        panel_y = h - 120
        ovl3 = frame.copy()
        cv2.rectangle(ovl3,(0,panel_y),(w,h),(20,20,50),-1)
        cv2.addWeighted(ovl3,0.85,frame,0.15,0,frame)
        cv2.line(frame,(0,panel_y),(w,panel_y),(0,200,255),2)

        cv2.putText(frame,"RECOGNIZED TEXT:", (12, panel_y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)

        # Word-wrap long text
        words   = ocr_text.replace('\n',' ').split()
        line    = ""; lines = []; max_chars = 60
        for word in words:
            if len(line)+len(word)+1 <= max_chars:
                line += (" " if line else "") + word
            else:
                lines.append(line); line = word
        if line: lines.append(line)

        for i, ln in enumerate(lines[:3]):
            cv2.putText(frame, ln, (12, panel_y + 42 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,200), 1, cv2.LINE_AA)

        cv2.putText(frame, "Press R to scan again | writing bigger helps accuracy",
                    (12, h-8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (150,150,150), 1, cv2.LINE_AA)

    # ── Bottom shortcut bar ──
    bar_h = 52
    if not (ocr_text and ocr_flash_ref[0] > 0):
        ovl2 = frame.copy()
        cv2.rectangle(ovl2,(0,h-bar_h),(px-8,h), T["hud_bg"],-1)
        cv2.addWeighted(ovl2,0.8,frame,0.2,0,frame)
        cv2.line(frame,(0,h-bar_h),(px-8,h-bar_h),T["hud_accent"],1)

        shortcuts=[
            ("☝ INDEX","Draw"),("✌ PEACE","Erase"),("✊ FIST","Clear"),
            ("🖐 PALM","Pause"),("👍 THUMB","Save"),("🤟 ROCK","Tool"),
            ("Z","Undo"),("W","Theme"),("R","OCR"),("Q","Quit"),
        ]
        slot=(px-8)//len(shortcuts)
        for i,(k,v) in enumerate(shortcuts):
            x=i*slot+4
            cv2.putText(frame,k,(x,h-32),cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,T["hud_accent"],1,cv2.LINE_AA)
            cv2.putText(frame,v,(x,h-14),cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,T["hud_text"],1,cv2.LINE_AA)

    # ── Title bar ──
    cv2.putText(frame,"SMART AIR WHITEBOARD  v4.0  —  TEACHER EDITION",
                (175,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, T["hud_accent"], 2, cv2.LINE_AA)

# Shared mutable for ocr_flash visible inside HUD
ocr_flash_ref = [0]

# ══════════════════════════════════════════════
#  WEBCAM
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    raise SystemExit(1)

print(f"[OK] Webcam ready")
print(f"[OK] Saves → {os.path.abspath(SAVE_DIR)}")

# ══════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════
canvas            = None
bg_canvas         = None   # background (whiteboard color fill)
prev_x, prev_y   = None, None
undo_stack        = deque(maxlen=MAX_UNDO)
gesture           = "NONE"
prev_gesture      = "NONE"
paused            = False
fist_t            = None
tool_cd           = 0
color_cd          = 0
saved_flash       = 0
emotion_str       = "NEUTRAL"
emotion_color     = (160,160,160)
vel_smooth        = 0.0
brush_size        = 8
frame_idx         = 0
fps               = 30.0
fps_t0            = time.time()
last_auto_save    = time.time()
cx, cy            = 0, 0
constellation_pts = []
ocr_text          = ""
ocr_running       = False
last_draw_time    = time.time()

print("\n╔══════════════════════════════════════════╗")
print("║  SMART AIR WHITEBOARD v4.0 — TEACHER    ║")
print("╚══════════════════════════════════════════╝")

# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.03); continue

    frame  = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    cx, cy = fw//2, fh//2
    theme  = THEMES[theme_idx]
    palette = get_palette()

    if canvas is None:
        canvas    = np.zeros((fh, fw, 3), np.uint8)
        bg_canvas = np.full((fh, fw, 3), theme["bg"], np.uint8)

    # ── FPS ──────────────────────────────────
    frame_idx += 1
    if frame_idx % 10 == 0:
        fps    = 10.0 / max(time.time()-fps_t0, 1e-6)
        fps_t0 = time.time()

    # ── MediaPipe ────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    face_result = face_mesh_detector.process(rgb)
    hand_result = hands.process(rgb)
    rgb.flags.writeable = True

    # ── Face ─────────────────────────────────
    face_lms = draw_face_window(frame, face_result)
    if face_lms:
        emotion_str, emotion_color = detect_emotion(face_lms)

    # ── Hand ─────────────────────────────────
    if hand_result.multi_hand_landmarks:
        lms_list   = hand_result.multi_hand_landmarks[0].landmark
        hand_label = hand_result.multi_handedness[0].classification[0].label

        raw_x = int(lms_list[8].x * fw)
        raw_y = int(lms_list[8].y * fh)
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

        draw_skeleton(frame, lms_list, fw, fh)

        draw_color = palette[color_idx][1]
        tool_name  = TOOLS[tool_idx]

        if not paused or gesture == "OPEN_PALM":

            # ── DRAW / WRITE ──────────────────
            if gesture == "DRAW":
                last_draw_time = time.time()
                ocr_triggered  = False

                wx   = lms_list[0].x; wy = lms_list[0].y
                ix   = lms_list[8].x; iy = lms_list[8].y
                span = math.hypot(wx-ix, wy-iy)
                brush_size = int(np.clip(span*85, BRUSH_MIN, BRUSH_MAX))

                # Cursor ring
                cc = theme["cursor"]
                cv2.circle(frame, (sx,sy), brush_size, cc, 2)
                cv2.circle(frame, (sx,sy), 3, cc, -1)

                if prev_x is not None:
                    if frame_idx % 10 == 0:
                        undo_stack.append(canvas.copy())

                    if tool_name == "MARKER":
                        cv2.line(canvas,(prev_x,prev_y),(sx,sy),draw_color,brush_size)

                    elif tool_name == "CHALK":
                        draw_chalk_line(canvas,(prev_x,prev_y),(sx,sy),draw_color,brush_size)

                    elif tool_name == "HIGHLIGHTER":
                        draw_highlighter_line(canvas,(prev_x,prev_y),(sx,sy),draw_color,brush_size)

                    elif tool_name == "POINTER":
                        # Red laser dot only — doesn't draw on canvas
                        cv2.circle(frame,(sx,sy),POINTER_RADIUS,(0,0,255),-1)
                        cv2.circle(frame,(sx,sy),POINTER_RADIUS+4,(0,0,200),2)
                        for r in [POINTER_RADIUS+8, POINTER_RADIUS+16]:
                            alpha=0.3*(1-r/30)
                            ovl_p=frame.copy()
                            cv2.circle(ovl_p,(sx,sy),r,(0,0,180),-1)
                            cv2.addWeighted(ovl_p,alpha,frame,1-alpha,0,frame)

                    elif tool_name == "SYMMETRY":
                        cv2.line(canvas,(prev_x,prev_y),(sx,sy),draw_color,brush_size)
                        cv2.line(canvas,(fw-prev_x,prev_y),(fw-sx,sy),draw_color,brush_size)

                    elif tool_name == "RADIAL":
                        radial_draw(canvas,sx,sy,prev_x,prev_y,
                                    cx,cy,draw_color,brush_size,"R")

                    elif tool_name == "CONSTELLATION":
                        constellation_pts.append((sx,sy))
                        if len(constellation_pts)>MAX_CONST_PTS:
                            constellation_pts.pop(0)

                    if tool_name != "POINTER" and len(particles)<MAX_PARTICLES:
                        for _ in range(2):
                            particles.append(Particle(sx,sy,draw_color))

                prev_x, prev_y = sx, sy

            # ── ERASE ────────────────────────
            elif gesture == "ERASE":
                cv2.circle(frame,(sx,sy),ERASE_RADIUS,(128,128,128),2)
                cv2.circle(frame,(sx,sy),4,(255,255,255),-1)
                if frame_idx%8==0:
                    undo_stack.append(canvas.copy())
                cv2.circle(canvas,(sx,sy),ERASE_RADIUS,(0,0,0),-1)
                prev_x, prev_y = None, None

            # ── FIST → CLEAR ─────────────────
            elif gesture == "FIST":
                if fist_t is None: fist_t=time.time()
                held  = time.time()-fist_t
                angle = int(360*min(held/FIST_HOLD_SEC,1.0))
                cv2.ellipse(frame,(sx,sy),(30,30),-90,0,angle,(0,0,255),3)
                cv2.putText(frame,"HOLD!",(sx-24,sy+6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
                if held >= FIST_HOLD_SEC:
                    undo_stack.append(canvas.copy())
                    canvas = np.zeros((fh,fw,3),np.uint8)
                    constellation_pts.clear()
                    ocr_text=""; ocr_flash_ref[0]=0
                    fist_t=None
                prev_x, prev_y = None, None

            # ── PALM → PAUSE ─────────────────
            elif gesture == "OPEN_PALM":
                if prev_gesture != "OPEN_PALM": paused = not paused
                prev_x, prev_y = None, None

            # ── THUMBS UP → SAVE ─────────────
            elif gesture == "THUMBS_UP":
                if prev_gesture != "THUMBS_UP":
                    # Save composite (bg + drawing)
                    composite = bg_canvas.copy()
                    mask      = canvas.any(axis=2)
                    composite[mask] = canvas[mask]
                    p = save_board(composite, "board")
                    saved_flash = 70
                    print(f"[SAVE] {p}")
                prev_x, prev_y = None, None

            # ── ROCK → CYCLE TOOL ────────────
            elif gesture == "ROCK":
                if tool_cd<=0 and prev_gesture!="ROCK":
                    tool_idx = (tool_idx+1) % len(TOOLS)
                    constellation_pts.clear()
                    tool_cd = 35
                prev_x, prev_y = None, None

            # ── THREE → CYCLE COLOR ──────────
            elif gesture == "THREE":
                if color_cd<=0 and prev_gesture!="THREE":
                    color_idx = (color_idx+1) % len(palette)
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
        kx.reset(); ky.reset()

    # ── Auto OCR after drawing pause ─────────
    if (OCR_AVAILABLE and not ocr_running
            and not ocr_triggered
            and canvas.any()
            and (time.time()-last_draw_time) > OCR_PAUSE_SEC
            and gesture != "DRAW"):
        ocr_triggered = True
        ocr_running   = True
        t = threading.Thread(
                target=run_ocr, args=(canvas.copy(), theme), daemon=True)
        t.start()

    # ── Cooldowns ────────────────────────────
    tool_cd  = max(0, tool_cd-1)
    color_cd = max(0, color_cd-1)

    # ── Constellation ────────────────────────
    if TOOLS[tool_idx]=="CONSTELLATION" and len(constellation_pts)>1:
        draw_constellation(canvas, constellation_pts,
                           palette[color_idx][1])

    # ── Particles ────────────────────────────
    particles[:] = [p for p in particles if p.step()]
    p_layer = np.zeros((fh,fw,3),np.uint8)
    for p in particles: p.draw(p_layer)

    # ── Build background ─────────────────────
    bg_canvas = np.full((fh,fw,3), theme["bg"], np.uint8)
    draw_grid(bg_canvas, theme)

    # ── Compose ──────────────────────────────
    # 1. background
    # 2. webcam (semi-transparent so whiteboard is clear)
    cam_alpha = 0.18 if theme["name"]=="WHITEBOARD" else 0.25
    output    = cv2.addWeighted(bg_canvas, 1.0-cam_alpha, frame, cam_alpha, 0)

    # 3. drawing canvas
    draw_mask = canvas.any(axis=2)
    output[draw_mask] = canvas[draw_mask]

    # 4. Glow for constellation
    if TOOLS[tool_idx] == "CONSTELLATION":
        output = cv2.addWeighted(output, 1.0,
                     apply_glow(canvas), 0.4, 0)

    # 5. Particles
    output = cv2.add(output, p_layer)

    # 6. Pointer on frame (draw on output not canvas)
    if TOOLS[tool_idx]=="POINTER" and hand_result.multi_hand_landmarks:
        if gesture == "DRAW":
            cv2.circle(output,(sx,sy),POINTER_RADIUS,(0,0,255),-1)
            cv2.circle(output,(sx,sy),POINTER_RADIUS+5,(0,0,150),2)

    # ── ocr_flash_ref update ─────────────────
    if ocr_flash > 0:
        ocr_flash_ref[0] = ocr_flash
        ocr_flash -= 1
    else:
        ocr_flash_ref[0] = max(0, ocr_flash_ref[0]-1)

    # ── HUD ──────────────────────────────────
    draw_teacher_hud(
        output, gesture, fps, emotion_str, emotion_color,
        brush_size, TOOLS[tool_idx],
        palette[color_idx][0], palette[color_idx][1],
        len(undo_stack), paused, vel_smooth,
        theme, get_timer_str(), ocr_text,
    )

    # ── Paused overlay ───────────────────────
    if paused:
        ov=output.copy()
        cv2.rectangle(ov,(0,0),(fw,fh),theme["bg"],-1)
        cv2.addWeighted(ov,0.4,output,0.6,0,output)
        cv2.putText(output,"⏸  PAUSED",(fw//2-80,fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX,1.4,theme["hud_accent"],3,cv2.LINE_AA)

    # ── Saved flash ──────────────────────────
    if saved_flash>0:
        cv2.putText(output,"✓ SAVED!",(fw//2-70,80),
                    cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,220,80),3,cv2.LINE_AA)
        saved_flash-=1

    # ── OCR running indicator ─────────────────
    if ocr_running:
        cv2.putText(output,"🔍 Recognizing text...",(fw//2-130,fh-70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2,cv2.LINE_AA)

    # ── Auto save ────────────────────────────
    if time.time()-last_auto_save > AUTO_SAVE_SEC:
        composite2 = bg_canvas.copy()
        m2 = canvas.any(axis=2); composite2[m2]=canvas[m2]
        save_board(composite2,"autosave")
        last_auto_save=time.time()

    cv2.imshow("Smart Air Whiteboard  v4.0  — Teacher Edition", output)

    # ── Keyboard ─────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        composite3=bg_canvas.copy(); m3=canvas.any(axis=2); composite3[m3]=canvas[m3]
        save_board(composite3,"final")
        break
    elif key == ord('z'):
        if undo_stack: canvas=undo_stack.pop(); print("[UNDO]")
    elif key == ord('c'):
        undo_stack.append(canvas.copy())
        canvas=np.zeros((fh,fw,3),np.uint8)
        constellation_pts.clear()
        ocr_text=""; ocr_flash_ref[0]=0
    elif key == ord('s'):
        composite4=bg_canvas.copy(); m4=canvas.any(axis=2); composite4[m4]=canvas[m4]
        p=save_board(composite4,"manual"); saved_flash=70; print(f"[SAVE] {p}")
    elif key == ord('w'):
        theme_idx=(theme_idx+1)%len(THEMES)
        canvas=np.zeros((fh,fw,3),np.uint8)   # reset canvas on theme change
        color_idx=0
        print(f"[THEME] {THEMES[theme_idx]['name']}")
    elif key == ord('r'):
        if OCR_AVAILABLE and not ocr_running and canvas.any():
            ocr_running=True
            ocr_triggered=True
            t2=threading.Thread(target=run_ocr,args=(canvas.copy(),theme),daemon=True)
            t2.start()
            print("[OCR] Manual scan triggered")
        elif not OCR_AVAILABLE:
            print("[OCR] Not available — pip install pytesseract Pillow")
    elif key == ord('t'):
        toggle_timer(); print(f"[TIMER] {'started' if timer_active else 'stopped'}")
    elif key in [ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7')]:
        tool_idx=min(int(chr(key))-1, len(TOOLS)-1)
        constellation_pts.clear()
        print(f"[TOOL] {TOOLS[tool_idx]}")
    elif key == ord('m'):
        tool_idx=(tool_idx+1)%len(TOOLS)
        constellation_pts.clear()

# ── Cleanup ───────────────────────────────────
cap.release(); cv2.destroyAllWindows()
hands.close(); face_mesh_detector.close()
print(f"\n[DONE] All boards saved in: {os.path.abspath(SAVE_DIR)}")
