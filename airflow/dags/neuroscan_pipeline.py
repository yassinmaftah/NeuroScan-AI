import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

log = logging.getLogger("neuroscan")

HOST = os.environ.get("HOST_PROJECT_DIR", "")
NETWORK = "neuroscan-ai_default"


def docker_run(image, command, gpu=False, shm=None):
    """Build the 'docker run' command for one task."""
    parts = ["docker run --rm",
             "--network " + NETWORK,
             "-e MLFLOW_TRACKING_URI=http://mlflow:5000"]
    if gpu:
        parts.append("--gpus all")
    if shm:
        parts.append("--shm-size " + shm)
    for folder in ["data", "models", "src", "reports", "weights"]:
        parts.append("-v " + HOST + "/" + folder + ":/app/" + folder)
    parts.append(image)
    parts.append(command)
    return " ".join(parts)


def on_failure(context):
    task = context["task_instance"]
    log.error("TASK FAILED: %s (try %s) in run %s", task.task_id, task.try_number, context["run_id"])


default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(hours=3),
    "on_failure_callback": on_failure,
}

with DAG(
    dag_id="neuroscan_pipeline",
    description="Brain tumor pipeline: data, training, evaluation, MLflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["neuroscan"],
    params={
        "epochs_head": Param(1, type="integer", minimum=1),
        "epochs_ft": Param(1, type="integer", minimum=1),
        "epochs_det": Param(1, type="integer", minimum=1),
    },
) as dag:

    prepare_cls = BashOperator(
        task_id="prepare_classification_data",
        bash_command=docker_run("neuroscan-tf", "python src/classification/prepare_data.py"))

    prepare_det = BashOperator(
        task_id="prepare_detection_data",
        bash_command=docker_run("neuroscan-yolo", "python src/detection/prepare_data.py"))

    validate = BashOperator(
        task_id="validate_data",
        bash_command=docker_run("neuroscan-tf", "python src/validate_data.py"))

    train_cls = BashOperator(
        task_id="train_classification",
        bash_command=docker_run(
            "neuroscan-tf",
            "python src/classification/train.py --epochs-head {{ params.epochs_head }} --epochs-ft {{ params.epochs_ft }}",
            gpu=True))

    train_det = BashOperator(
        task_id="train_detection",
        bash_command=docker_run(
            "neuroscan-yolo",
            "python src/detection/train.py --epochs {{ params.epochs_det }}",
            gpu=True, shm="4g"))

    eval_cls = BashOperator(
        task_id="evaluate_classification",
        bash_command=docker_run("neuroscan-tf", "python src/classification/evaluate.py", gpu=True))

    eval_det = BashOperator(
        task_id="evaluate_detection",
        bash_command=docker_run("neuroscan-yolo", "python src/detection/evaluate.py", gpu=True, shm="4g"))

    save = BashOperator(
        task_id="save_artifacts",
        bash_command=docker_run("neuroscan-tf", "python src/save_artifacts.py"))

    [prepare_cls, prepare_det] >> validate >> train_cls >> train_det >> eval_cls >> eval_det >> save