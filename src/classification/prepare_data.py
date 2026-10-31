import os
import random
import shutil
import logging

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_classification")

RAW_PATH = "/app/data/raw/Data"
OUT_PATH = "/app/data/processed/classification_airflow"
GOOD_EXTENSIONS = [".jpg", ".jpeg", ".png"]
SEED = 42


def check_images():
    valid = {}
    bad = []
    for class_name in sorted(os.listdir(RAW_PATH)):
        class_path = os.path.join(RAW_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
        valid[class_name] = []
        for file_name in sorted(os.listdir(class_path)):
            file_path = os.path.join(class_path, file_name)
            extension = os.path.splitext(file_name)[1].lower()
            if extension not in GOOD_EXTENSIONS:
                bad.append(file_path)
                continue
            try:
                img = Image.open(file_path)
                img.verify()
                valid[class_name].append(file_path)
            except Exception as error:
                log.warning("Bad image %s: %s", file_path, error)
                bad.append(file_path)
    return valid, bad


def split_and_copy(valid):
    files = []
    labels = []
    for class_name in valid:
        for file_path in valid[class_name]:
            files.append(file_path)
            labels.append(class_name)

    train_f, temp_f, train_l, temp_l = train_test_split(
        files, labels, test_size=0.30, stratify=labels, random_state=SEED)
    val_f, test_f, val_l, test_l = train_test_split(
        temp_f, temp_l, test_size=0.50, stratify=temp_l, random_state=SEED)

    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)

    sets = {"train": (train_f, train_l), "val": (val_f, val_l), "test": (test_f, test_l)}
    for set_name in sets:
        file_list, label_list = sets[set_name]
        for file_path, label in zip(file_list, label_list):
            folder = os.path.join(OUT_PATH, set_name, label)
            os.makedirs(folder, exist_ok=True)
            shutil.copy(file_path, folder)
        log.info("%s: %d images", set_name, len(file_list))


def blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def flip(img):
    return ImageOps.mirror(img)


def add_noise(img):
    array = np.array(img).astype("float32")
    noise = np.random.normal(0, 15, array.shape)
    array = np.clip(array + noise, 0, 255).astype("uint8")
    return Image.fromarray(array)


def augment_train():
    train_path = os.path.join(OUT_PATH, "train")
    originals = {}
    for class_name in sorted(os.listdir(train_path)):
        folder = os.path.join(train_path, class_name)
        originals[class_name] = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]

    target = max(len(originals[c]) for c in originals)

    for class_name in originals:
        missing = target - len(originals[class_name])
        folder = os.path.join(train_path, class_name)
        for i in range(missing):
            file_path = random.choice(originals[class_name])
            function = random.choice([blur, flip, add_noise])
            new_img = function(Image.open(file_path))
            new_img.save(os.path.join(folder, "aug_" + str(i) + ".jpg"))
        log.info("%s: added %d augmented images", class_name, missing)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    valid, bad = check_images()
    for class_name in valid:
        log.info("%s: %d valid images", class_name, len(valid[class_name]))
    log.info("Excluded files: %d", len(bad))

    if len(valid) == 0:
        raise ValueError("No class folders found in " + RAW_PATH)

    split_and_copy(valid)
    augment_train()
    log.info("Done")


if __name__ == "__main__":
    main()