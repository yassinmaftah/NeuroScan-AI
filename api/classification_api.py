import io
import json
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input

MODEL_PATH = "/app/models/classification/inceptionv3_final.keras"
CLASSES_PATH = "/app/models/classification/class_names.json"

model = None
class_names = []


@asynccontextmanager
async def lifespan(app):
    global model, class_names
    model = keras.models.load_model(MODEL_PATH)
    class_names = json.load(open(CLASSES_PATH))
    yield


app = FastAPI(
    title="NeuroScan-AI Classification API",
    description="Classifies a brain MRI image into glioma, meningioma, notumor or pituitary (InceptionV3).",
    version="1.0",
    lifespan=lifespan)


@app.get("/health", tags=["system"])
def health():
    """Tells if the API is running and the model is loaded."""
    return {"status": "ok", "model_loaded": model is not None, "classes": class_names}


@app.post("/predict", tags=["classification"])
async def predict(file: UploadFile = File(...)):
    """Upload a brain MRI image (jpg or png) and get the predicted class with its confidence."""
    content = await file.read()

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The file is not a valid image")

    array = np.array(image).astype("float32")
    array = tf.image.resize(array, (299, 299)).numpy()
    array = preprocess_input(array)
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    best = int(np.argmax(probabilities))

    return {
        "predicted_class": class_names[best],
        "confidence": float(probabilities[best]),
        "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))},
    }