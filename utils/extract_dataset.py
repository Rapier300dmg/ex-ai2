import os
import cv2
import argparse
from tqdm import tqdm
from utils.face_utils import get_face_landmarks, get_lip_coordinates, crop_mouth_region
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def process_videos(input_dir, output_dir, seq_per_video=50):
    os.makedirs(output_dir, exist_ok=True)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    for word in os.listdir(input_dir):
        word_in = os.path.join(input_dir, word)
        word_out = os.path.join(output_dir, word)
        os.makedirs(word_out, exist_ok=True)

        for vid_name in tqdm(os.listdir(word_in), desc=word):
            vid_path = os.path.join(word_in, vid_name)
            cap = cv2.VideoCapture(vid_path)
            saved = 0

            while saved < seq_per_video and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                landmarks = get_face_landmarks(frame, face_mesh)
                if landmarks:
                    lips = get_lip_coordinates(landmarks, frame.shape)
                    mouth = crop_mouth_region(frame, lips)
                    mouth = cv2.resize(mouth, (64, 64))
                    fname = f"{vid_name[:-4]}_{saved:03d}.jpg"
                    cv2.imwrite(os.path.join(word_out, fname), mouth)
                    saved += 1

            cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="input_dir",  required=True)
    parser.add_argument("--out", dest="output_dir", required=True)
    args = parser.parse_args()
    process_videos(args.input_dir, args.output_dir)
