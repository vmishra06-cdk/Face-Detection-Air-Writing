import cv2
import mediapipe as mp
import numpy as np
import os

# =============================
# Face Detection
# =============================
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# =============================
# MediaPipe Hands (STABLE)
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =============================
# Webcam
# =============================
cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    # ---------- FACE WINDOW ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_small = cv2.resize(frame[y:y+h, x:x+w], (140, 140))
        frame[10:150, 10:150] = face_small
        cv2.rectangle(frame, (10, 10), (150, 150), (0, 255, 0), 2)

    # ---------- HAND TRACKING ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        h, w, _ = frame.shape
        index_finger = result.multi_hand_landmarks[0].landmark[8]

        x = int(index_finger.x * w)
        y = int(index_finger.y * h)

        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)

        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    output = cv2.add(frame, canvas)

    cv2.putText(output, "Air Writing (Finger)", (170, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(output, "C: Clear | Q: Quit", (170, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Air Writing with Face Window", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
