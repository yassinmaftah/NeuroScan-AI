import io
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "/app/models/detection/best.pt"

model = None


@asynccontextmanager
async def lifespan(app):
    global model
    model = YOLO(MODEL_PATH)
    yield


app = FastAPI(
    title="NeuroScan-AI Detection API",
    description="Detects and localizes brain tumors in an MRI image (YOLOv8).",
    version="1.0",
    lifespan=lifespan)


def read_image(content):
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The file is not a valid image")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "classes": list(model.names.values()) if model is not None else []}


@app.post("/predict", tags=["detection"])
async def predict(file: UploadFile = File(...),
                  confidence: float = Query(0.25, ge=0.0, le=1.0)):
    """Upload a brain MRI image and get the detected tumors: class, confidence and box coordinates (pixels)."""
    image = read_image(await file.read())
    result = model.predict(image, conf=confidence, device="cpu", verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append({
            "class": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    return {"image_width": image.width, "image_height": image.height,
            "count": len(detections), "detections": detections}


@app.post("/predict/image", tags=["detection"])
async def predict_image(file: UploadFile = File(...),
                        confidence: float = Query(0.25, ge=0.0, le=1.0)):
    image = read_image(await file.read())
    result = model.predict(image, conf=confidence, device="cpu", verbose=False)[0]

    drawn = result.plot()                         
    drawn = Image.fromarray(drawn[:, :, ::-1])
    buffer = io.BytesIO()
    drawn.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")