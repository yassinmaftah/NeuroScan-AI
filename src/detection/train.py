import os
import shutil
import argparse
import logging

from ultralytics import YOLO, settings

settings.update({"mlflow": False})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_detection")

DATA_YAML = "/app/data/processed/detection_airflow/data.yaml"
OUT_DIR = "/app/models/detection_airflow"
RUN_NAME = "yolov8s_pipeline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError("Run prepare_data.py first: " + DATA_YAML + " not found")

    os.makedirs(OUT_DIR, exist_ok=True)

    model = YOLO("yolov8s.pt")
    log.info("Training: %d epochs, batch %d, image size %d", args.epochs, args.batch, args.imgsz)

    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        patience=10,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        seed=42,
        project=OUT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        degrees=10, translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.0, mosaic=1.0)

    best_path = OUT_DIR + "/" + RUN_NAME + "/weights/best.pt"
    if not os.path.exists(best_path):
        raise FileNotFoundError("Training finished but best.pt was not created")

    shutil.copy(best_path, OUT_DIR + "/best.pt")
    shutil.copy(DATA_YAML, OUT_DIR + "/data.yaml")
    log.info("Model saved: %s/best.pt", OUT_DIR)


if __name__ == "__main__":
    main()