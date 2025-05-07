import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.face_utils import get_face_landmarks, get_lip_coordinates, crop_mouth_region
import mediapipe as mp

# Параметры
SEQ_LENGTH = 16
IMG_SIZE = (64,64)
WORDS = sorted(__import__("os").listdir("data/mouth_frames_train/AVDigits"))
model = load_model("models/lipnet.h5")

# FaceMesh
mpf = mp.solutions.face_mesh
fm = mpf.FaceMesh(static_image_mode=False, max_num_faces=1)

# Источник: камеру или файл
cap = cv2.VideoCapture(0)  # или путь "data/raw/AVDigits/0/S1_0_01.mp4"
buffer = []

while True:
    ret, frame = cap.read()
    if not ret: break
    lms = get_face_landmarks(frame, fm)
    if lms:
        lips = get_lip_coordinates(lms, frame.shape)
        mouth = crop_mouth_region(frame, lips)
        mouth = cv2.resize(mouth, IMG_SIZE) / 255.0
        buffer.append(mouth)
        if len(buffer) == SEQ_LENGTH:
            seq = np.expand_dims(np.array(buffer), 0)
            probs = model.predict(seq)[0]
            word = WORDS[np.argmax(probs)]
            buffer.pop(0)
            cv2.putText(frame, f"Predict: {word}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Lip Reading", frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
