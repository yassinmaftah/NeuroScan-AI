import os
import json
import argparse
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_classification")

DATA_PATH = "/app/data/processed/classification_airflow"
OUT_PATH = "/app/models/classification_airflow"
IMG_SIZE = (299, 299)
BATCH_SIZE = 32


def load_data():
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_PATH + "/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=True, seed=42)
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_PATH + "/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=False)

    class_names = train_ds.class_names

    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes):
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False

    model = keras.Sequential([
        keras.Input(shape=(299, 299, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-ft", type=int, default=10)
    parser.add_argument("--freeze-first", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH + "/train"):
        raise FileNotFoundError("Run prepare_data.py first: " + DATA_PATH + " not found")

    os.makedirs(OUT_PATH, exist_ok=True)
    train_ds, val_ds, class_names = load_data()
    log.info("Classes: %s", class_names)

    model, base_model = build_model(len(class_names))

    log.info("Phase 1: feature extraction, %d epochs max", args.epochs_head)
    compile_model(model, 0.001)
    stop_1 = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history_1 = model.fit(train_ds, validation_data=val_ds,
                          epochs=args.epochs_head, callbacks=[stop_1])

    log.info("Phase 2: fine-tuning, %d epochs max", args.epochs_ft)
    base_model.trainable = True
    for layer in base_model.layers[:args.freeze_first]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, 0.00001)
    stop_2 = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history_2 = model.fit(train_ds, validation_data=val_ds,
                          epochs=args.epochs_ft, callbacks=[stop_2])

    model.save(OUT_PATH + "/model.keras")
    json.dump(class_names, open(OUT_PATH + "/class_names.json", "w"))
    json.dump(history_1.history, open(OUT_PATH + "/history_feature_extraction.json", "w"))
    json.dump(history_2.history, open(OUT_PATH + "/history_fine_tuning.json", "w"))
    log.info("Model and history saved in %s", OUT_PATH)


if __name__ == "__main__":
    main()