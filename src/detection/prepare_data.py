import os
import shutil
import logging

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_detection")

RAW_PATH = "/app/data/raw/brain_tumor"
OUT_PATH = "/app/data/processed/detection_airflow"
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
SEED = 42


def label_is_valid(label_path):
    """A label is valid if every line has 5 values: class id 0-3, then 4 numbers in 0-1."""
    file = open(label_path)
    content = file.read().strip()
    file.close()
    if content == "":
        return False, "empty label"

    for line in content.split("\n"):
        values = line.split()
        if len(values) != 5:
            return False, "not 5 values"
        try:
            class_id = int(values[0])
            numbers = [float(v) for v in values[1:]]
        except ValueError:
            return False, "not numbers"
        if class_id not in range(len(CLASS_NAMES)):
            return False, "bad class id"
        for number in numbers:
            if number < 0 or number > 1:
                return False, "value outside 0-1"
        if numbers[2] <= 0 or numbers[3] <= 0:
            return False, "box with zero size"
    return True, ""


def collect_pairs():
    pairs = []      
    excluded = []

    for split in ["Train", "Val"]:
        for class_folder in CLASS_NAMES:
            images_path = os.path.join(RAW_PATH, split, class_folder, "images")
            labels_path = os.path.join(RAW_PATH, split, class_folder, "labels")

            images = {}
            for f in sorted(os.listdir(images_path)):
                images[os.path.splitext(f)[0]] = f
            labels = {}
            for f in sorted(os.listdir(labels_path)):
                labels[os.path.splitext(f)[0]] = f

            for name in images:
                if name not in labels:
                    excluded.append((split, class_folder, name, "image without label"))
                    continue
                label_path = os.path.join(labels_path, labels[name])
                ok, reason = label_is_valid(label_path)
                if not ok:
                    excluded.append((split, class_folder, name, reason))
                    continue
                pairs.append([os.path.join(images_path, images[name]), label_path, class_folder])

            for name in labels:
                if name not in images:
                    excluded.append((split, class_folder, name, "label without image"))

    return pairs, excluded


def split_and_copy(pairs):
    classes = [p[2] for p in pairs]
    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=0.30, stratify=classes, random_state=SEED)
    temp_classes = [p[2] for p in temp_pairs]
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.50, stratify=temp_classes, random_state=SEED)

    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)

    sets = {"train": train_pairs, "valid": val_pairs, "test": test_pairs}
    counter = 0
    for set_name in sets:
        os.makedirs(os.path.join(OUT_PATH, "images", set_name))
        os.makedirs(os.path.join(OUT_PATH, "labels", set_name))

        for image_path, label_path, class_folder in sets[set_name]:
            counter = counter + 1
            new_name = class_folder.replace(" ", "") + "_" + str(counter).zfill(4)
            extension = os.path.splitext(image_path)[1]
            shutil.copy(image_path, os.path.join(OUT_PATH, "images", set_name, new_name + extension))
            shutil.copy(label_path, os.path.join(OUT_PATH, "labels", set_name, new_name + ".txt"))

    for set_name in sets:
        n_images = len(os.listdir(os.path.join(OUT_PATH, "images", set_name)))
        n_labels = len(os.listdir(os.path.join(OUT_PATH, "labels", set_name)))
        log.info("%s: %d images, %d labels", set_name, n_images, n_labels)
        if n_images != n_labels:
            raise ValueError("Images and labels do not match in " + set_name)


def write_yaml():
    text = "path: " + OUT_PATH + "\n"
    text = text + "train: images/train\n"
    text = text + "val: images/valid\n"
    text = text + "test: images/test\n"
    text = text + "nc: " + str(len(CLASS_NAMES)) + "\n"
    text = text + "names: " + str(CLASS_NAMES) + "\n"
    file = open(OUT_PATH + "/data.yaml", "w")
    file.write(text)
    file.close()
    log.info("data.yaml written")


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(RAW_PATH + " not found")

    pairs, excluded = collect_pairs()
    log.info("Valid pairs: %d", len(pairs))
    log.info("Excluded: %d", len(excluded))
    for item in excluded:
        log.info("   excluded: %s", item)

    if len(pairs) == 0:
        raise ValueError("No valid image-label pairs found")

    split_and_copy(pairs)
    write_yaml()
    log.info("Done")


if __name__ == "__main__":
    main()