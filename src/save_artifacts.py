import os
import json
import shutil
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("save_artifacts")

OUT = "/app/models/pipeline_latest"
FILES = {
    "classification": [
        "/app/models/classification_airflow/model.keras",
        "/app/models/classification_airflow/class_names.json",
        "/app/reports/classification_airflow/test_metrics.json",
        "/app/reports/classification_airflow/confusion_matrix.png",
    ],
    "detection": [
        "/app/models/detection_airflow/best.pt",
        "/app/models/detection_airflow/data.yaml",
        "/app/models/detection_airflow/test_metrics.json",
    ],
}


def main():
    missing = []
    for group in FILES:
        for path in FILES[group]:
            if not os.path.exists(path):
                missing.append(path)
    if len(missing) > 0:
        raise FileNotFoundError("Missing files: " + str(missing))

    for group in FILES:
        folder = os.path.join(OUT, group)
        os.makedirs(folder, exist_ok=True)
        for path in FILES[group]:
            shutil.copy(path, folder)
            log.info("saved %s", os.path.join(folder, os.path.basename(path)))

    cls = json.load(open("/app/reports/classification_airflow/test_metrics.json"))
    det = json.load(open("/app/models/detection_airflow/test_metrics.json"))
    manifest = {
        "created": datetime.now().isoformat(),
        "classification_test_accuracy": cls["test_accuracy"],
        "detection_test_mAP50": det["test_mAP50"],
    }
    json.dump(manifest, open(OUT + "/manifest.json", "w"), indent=2)
    log.info("Manifest: %s", manifest)


if __name__ == "__main__":
    main()