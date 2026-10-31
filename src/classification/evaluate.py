import os
import json
import logging

os.environ["GIT_PYTHON_REFRESH"] = "quiet" 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate_classification")

DATA_PATH = "/app/data/processed/classification_airflow"
MODEL_DIR = "/app/models/classification_airflow"
REPORT_DIR = "/app/reports/classification_airflow"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def main():
    if not os.path.exists(MODEL_DIR + "/model.keras"):
        raise FileNotFoundError("Run train.py first: no model in " + MODEL_DIR)

    os.makedirs(REPORT_DIR, exist_ok=True)

    model = keras.models.load_model(MODEL_DIR + "/model.keras")
    class_names = json.load(open(MODEL_DIR + "/class_names.json"))

    test_ds = keras.utils.image_dataset_from_directory(
        DATA_PATH + "/test", image_size=(299, 299), batch_size=32,
        label_mode="categorical", shuffle=False)
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

    y_true = []
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix (Airflow pipeline)")
    cm_path = REPORT_DIR + "/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    metrics = {
        "test_accuracy": report["accuracy"],
        "test_precision_macro": report["macro avg"]["precision"],
        "test_recall_macro": report["macro avg"]["recall"],
        "test_f1_macro": report["macro avg"]["f1-score"],
    }
    for name in class_names:
        metrics["test_recall_" + name] = report[name]["recall"]

    json.dump(metrics, open(REPORT_DIR + "/test_metrics.json", "w"), indent=2)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("brain-tumor-classification")

    with mlflow.start_run(run_name="airflow_pipeline"):
        mlflow.log_param("source", "airflow_pipeline")
        mlflow.log_param("base_model", "InceptionV3")
        mlflow.log_param("data_path", DATA_PATH)

        for name in metrics:
            mlflow.log_metric(name, metrics[name])

        for history_file in ["history_feature_extraction.json", "history_fine_tuning.json"]:
            history = json.load(open(MODEL_DIR + "/" + history_file))
            prefix = history_file.replace("history_", "").replace(".json", "")
            for key in history:
                for epoch in range(len(history[key])):
                    mlflow.log_metric(prefix + "_" + key, history[key][epoch], step=epoch + 1)

        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(REPORT_DIR + "/test_metrics.json", "results")
        mlflow.log_artifact(MODEL_DIR + "/model.keras", "model")
        mlflow.log_artifact(MODEL_DIR + "/class_names.json", "config")

    log.info("Test accuracy: %.4f", metrics["test_accuracy"])
    log.info("Logged in MLflow at %s", MLFLOW_URI)


if __name__ == "__main__":
    main()