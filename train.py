import os
from tensorflow.keras import models
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from utils.dataset import build_lip_dataset

ROOT = "data/mouth_frames_train/AVDigits"
WORDS = sorted(os.listdir(ROOT))


def build_model(seq_length=16, img_size=(64, 64), num_classes=len(WORDS)):
    inputs = Input((seq_length, img_size[0], img_size[1], 3))
    backbone = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    x = TimeDistributed(backbone)(inputs)
    x = LSTM(128)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


if __name__ == "__main__":
    dataset = build_lip_dataset(ROOT, WORDS)
    dataset = dataset.shuffle(1000)
    val_ds = dataset.take(100)
    train_ds = dataset.skip(100)

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[early_stop]
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/lipnet.h5")
