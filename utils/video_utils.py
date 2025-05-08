import os
import cv2
import mediapipe as mp
from utils.face_utils import get_face_landmarks, get_lip_coordinates, crop_mouth_region

def capture_mouth_frames(output_dir, max_frames=100):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = get_face_landmarks(frame, face_mesh)
        if landmarks:
            lip_coords = get_lip_coordinates(landmarks, frame.shape)
            mouth = crop_mouth_region(frame, lip_coords)
            mouth = cv2.resize(mouth, (64, 64))
            cv2.imwrite(f"{output_dir}/frame_{count:04d}.jpg", mouth)
            count += 1

        cv2.imshow("Видео", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
