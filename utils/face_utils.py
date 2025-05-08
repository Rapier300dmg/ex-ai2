import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 415, 310
]

def get_face_landmarks(image, face_mesh):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0].landmark

def get_lip_coordinates(landmarks, img_shape):
    h, w = img_shape[:2]
    coords = []
    for idx in LIPS_IDX:
        lm = landmarks[idx]
        coords.append((int(lm.x * w), int(lm.y * h)))
    return np.array(coords)

def crop_mouth_region(image, lip_coords, pad=10):
    x_min, y_min = lip_coords.min(axis=0) - pad
    x_max, y_max = lip_coords.max(axis=0) + pad
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)
    return image[y_min:y_max, x_min:x_max]
