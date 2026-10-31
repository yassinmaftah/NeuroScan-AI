import os
import json
import logging

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import pandas as pd
import mlflow
from ultralytics import YOLO, settings

settings.update({"mlflow": False})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate_detection")

MODEL_DIR = "/app/models/detection_airflow"
RUN_DIR = MODEL_DIR + "/yolov8s_pipeline"
DATA_YAML = "/app/data/processed/detection_airflow/data.yaml"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def main():
    if not os.path.exists(MODEL_DIR + "/best.pt"):
        raise FileNotFoundError("Run train.py first: no model in " + MODEL_DIR)

    model = YOLO(MODEL_DIR + "/best.pt")

    results = model.val(
        data=DATA_YAML, split="test", imgsz=640, batch=16, device=0,
        project=MODEL_DIR, name="test_eval", exist_ok=True)

    metrics = {
        "test_precision": float(results.box.mp),
        "test_recall": float(results.box.mr),
        "test_mAP50": float(results.box.map50),
        "test_mAP50_95": float(results.box.map),
    }
    for i in range(len(results.box.ap_class_index)):
        class_id = int(results.box.ap_class_index[i])
        name = model.names[class_id].replace(" ", "")
        metrics["test_recall_" + name] = float(results.box.r[i])
        metrics["test_mAP50_" + name] = float(results.box.ap50[i])

    json.dump(metrics, open(MODEL_DIR + "/test_metrics.json", "w"), indent=2)

    df = pd.read_csv(RUN_DIR + "/results.csv")
    df.columns = df.columns.str.strip()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("brain-tumor-detection")

    with mlflow.start_run(run_name="airflow_pipeline"):
        mlflow.log_param("source", "airflow_pipeline")
        mlflow.log_param("model", "yolov8s")
        mlflow.log_param("epochs_run", len(df))
        mlflow.log_param("imgsz", 640)
        mlflow.log_param("data_yaml", DATA_YAML)

        for name in metrics:
            mlflow.log_metric(name, metrics[name])

        for i in range(len(df)):
            epoch = int(df["epoch"][i])
            for column in df.columns:
                if column == "epoch" or column.startswith("lr/"):
                    continue
                mlflow.log_metric(column.replace("(B)", ""), float(df[column][i]), step=epoch)

        mlflow.log_artifact(MODEL_DIR + "/best.pt", "model")
        mlflow.log_artifact(MODEL_DIR + "/data.yaml", "config")
        mlflow.log_artifact(MODEL_DIR + "/test_metrics.json", "results")
        mlflow.log_artifact(RUN_DIR + "/results.csv", "results")

        cm_path = MODEL_DIR + "/test_eval/confusion_matrix.png"
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, "plots")
        else:
            log.warning("confusion_matrix.png not found, skipped")

    log.info("Test mAP@0.5: %.4f", metrics["test_mAP50"])
    log.info("Logged in MLflow at %s", MLFLOW_URI)


if __name__ == "__main__":
    main()