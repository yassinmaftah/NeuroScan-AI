import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validate_data")

CLS_PATH = "/app/data/processed/classification_airflow"
DET_PATH = "/app/data/processed/detection_airflow"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
problems = []


def check_classification():
    originals = {}
    for set_name in ["train", "val", "test"]:
        originals[set_name] = set()
        for class_name in CLASSES:
            folder = os.path.join(CLS_PATH, set_name, class_name)
            if not os.path.isdir(folder):
                problems.append("missing folder: " + folder)
                continue
            files = os.listdir(folder)
            log.info("%s/%s: %d images", set_name, class_name, len(files))
            if len(files) == 0:
                problems.append("empty folder: " + folder)
            for f in files:
                if not f.startswith("aug_"):
                    originals[set_name].add(class_name + "/" + f)

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        common = originals[a] & originals[b]
        if len(common) > 0:
            problems.append(str(len(common)) + " images in both " + a + " and " + b)

    sizes = []
    for class_name in CLASSES:
        folder = os.path.join(CLS_PATH, "train", class_name)
        if os.path.isdir(folder):
            sizes.append(len(os.listdir(folder)))
    if len(set(sizes)) > 1:
        problems.append("train classes are not balanced: " + str(sizes))


def check_detection():
    for set_name in ["train", "valid", "test"]:
        images = os.listdir(os.path.join(DET_PATH, "images", set_name))
        labels = os.listdir(os.path.join(DET_PATH, "labels", set_name))
        image_names = set(os.path.splitext(f)[0] for f in images)
        label_names = set(os.path.splitext(f)[0] for f in labels)
        log.info("detection %s: %d images, %d labels", set_name, len(images), len(labels))
        if len(images) == 0:
            problems.append("no images in " + set_name)
        if image_names != label_names:
            problems.append("images and labels do not match in " + set_name)

    yaml_path = DET_PATH + "/data.yaml"
    if not os.path.exists(yaml_path):
        problems.append("data.yaml missing")
    elif "nc: 4" not in open(yaml_path).read():
        problems.append("data.yaml: nc is not 4")


def main():
    check_classification()
    check_detection()
    if len(problems) > 0:
        for p in problems:
            log.error("PROBLEM: %s", p)
        raise ValueError(str(len(problems)) + " data problems found")
    log.info("All checks passed")


if __name__ == "__main__":
    main()