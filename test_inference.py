import numpy as np
import cv2
import glob
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

paths = sorted(glob.glob("data/mouth_frames_train/AVDigits/0/*.jpg"))[:16]

for p in paths[:4]:
    img = cv2.imread(p)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(os.path.basename(p))
    plt.axis('off')
plt.show()

seq = [cv2.resize(cv2.imread(p), (64, 64)) / 255.0 for p in paths]
seq = np.expand_dims(seq, 0)

model = load_model("models/lipnet.h5")
probs = model.predict(seq)[0]
pred = np.argmax(probs)

print("Ожидали класс 0, предсказано:", pred)
print("Вероятности по классам:", probs)
