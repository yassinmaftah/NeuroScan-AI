# NeuroScan-AI

Brain tumor analysis from MRI images, in two parts:

- **Classification** with InceptionV3 (pretrained on ImageNet): glioma, meningioma, pituitary, no tumor.
- **Detection and localization** with YOLOv8s.

Around the models: MLflow (experiment tracking), Apache Airflow (pipeline), FastAPI (two REST APIs), Streamlit (interface), all in Docker.

> Educational project. Not a medical device.

## Results (test set)

| Model | Setup | Result |
|---|---|---|
| InceptionV3, feature extraction | frozen base, new head | 90.5 % accuracy |
| InceptionV3, fine-tuning | most layers trainable, lr 1e-5 | **96.3 % accuracy** (selected) |
| InceptionV3, fine-tuning | last 50 layers | 95.4 % accuracy |
| YOLOv8s | 50 epochs | **mAP@0.5 0.950**, mAP@0.5:0.95 0.753 |

The selected models come from the notebooks. The Airflow pipeline runs the same logic as scripts, on its own data split, so its numbers are separate and, in quick mode (1 epoch), lower.

## Structure

```
data/            raw and processed images (not in git)
models/          trained models (not in git)
weights/         pretrained weights, downloaded once (not in git)
notebooks/       experiments, one notebook per step
src/             classification/ and detection/ scripts used by Airflow
airflow/dags/    pipeline DAG
api/             classification_api.py, detection_api.py
ui/              Streamlit app
docs/            work plan and schedule
```

## Requirements

- Docker Desktop with the engine running
- An NVIDIA GPU with up-to-date drivers (training uses `--gpus all`)

## Installation

1. **Datasets.** Put them here (not included in the repo):
   - `data/raw/Data/` with 4 folders: `glioma`, `meningioma`, `notumor`, `pituitary` (classification)
   - `data/raw/brain_tumor/` from the Kaggle dataset "MRI for brain tumor with bounding boxes", with `Train/` and `Val/` (detection)

2. **Environment file.**
   ```
   cp .env.example .env
   ```
   Edit `HOST_PROJECT_DIR` to the absolute path of this folder, with forward slashes.

3. **Build the images** (the first build is slow, several GB):
   ```
   docker compose build notebook-tf notebook-yolo mlflow airflow streamlit
   ```

4. **Download the pretrained weights once:**
   ```
   mkdir weights
   docker compose run --rm notebook-tf python -c "from tensorflow.keras.applications import InceptionV3; m = InceptionV3(weights='imagenet', include_top=False); m.save_weights('/app/weights/inception_v3_notop.weights.h5')"
   docker compose run --rm notebook-yolo python -c "from ultralytics import YOLO; import shutil; YOLO('yolov8s.pt'); shutil.copy('yolov8s.pt', '/app/weights/yolov8s.pt')"
   ```

## Run

**Notebooks** (data preparation and training, one at a time):
```
docker compose up notebook-tf     # http://localhost:8888/?token=neuroscan
docker compose up notebook-yolo   # http://localhost:8889/?token=neuroscan
```
Running the notebooks produces the production models used by the APIs:
`models/classification/inceptionv3_final.keras` + `class_names.json` and `models/detection/best.pt`.

**Application:**
```
docker compose up -d mlflow api-classification api-detection streamlit airflow
```

| Service | URL |
|---|---|
| Streamlit interface | http://localhost:8502 |
| Classification API (docs) | http://localhost:8001/docs |
| Detection API (docs) | http://localhost:8002/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

Airflow login: user `admin`, password in
`docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt`

Stop everything with `docker compose down`.

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /health` (both) | API and model status |
| `POST /predict` (classification) | image in, predicted class, confidence and probabilities out |
| `POST /predict` (detection) | image in, detected boxes (class, confidence, pixel coordinates) out |
| `POST /predict/image` (detection) | image in, the same image with boxes drawn out (PNG) |

Full interactive documentation: `/docs` on each API.

## Airflow pipeline

The DAG `neuroscan_pipeline` runs 8 tasks:
prepare classification data and prepare detection data, then validate the data, then train classification, train detection, evaluate classification and detection (results logged in MLflow), and finally save the artifacts in `models/pipeline_latest/`.

Each task starts the right Docker image. Trigger the DAG from the Airflow page; the parameters `epochs_head`, `epochs_ft` and `epochs_det` default to 1 (quick test). A full run uses 10, 10 and 50. Failed tasks are retried once, and every task has its own log.

## MLflow

Experiments `brain-tumor-classification` and `brain-tumor-detection` hold the logged runs (parameters, metrics, curves, models). The best models are registered as `neuroscan-classifier` and `neuroscan-detector` (alias `production`).

## Notes

- Data, models and weights are not in git because of their size.
- The two models use separate images because TensorFlow and PyTorch cannot share one environment here (conflicting CUDA libraries).