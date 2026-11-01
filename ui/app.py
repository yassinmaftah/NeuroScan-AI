import os

import pandas as pd
import requests
import streamlit as st

CLASSIFICATION_URL = os.environ.get("CLASSIFICATION_API_URL", "http://localhost:8001")
DETECTION_URL = os.environ.get("DETECTION_API_URL", "http://localhost:8002")

st.set_page_config(page_title="NeuroScan-AI", page_icon="🧠", layout="wide")
st.title("🧠 NeuroScan-AI")
st.caption("Brain tumor classification (InceptionV3) and detection (YOLOv8). "
           "Educational project, not a medical device.")


def send_image(url, uploaded, params=None):
    """Send the uploaded image to an API. Returns the response, or None if it failed."""
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    try:
        response = requests.post(url, files=files, params=params, timeout=60)
    except requests.exceptions.RequestException:
        st.error("Cannot reach the API at " + url + ". Is the API container running?")
        return None
    if response.status_code != 200:
        st.error("API error " + str(response.status_code) + ": " + response.text)
        return None
    return response


tab_cls, tab_det = st.tabs(["Classification", "Detection"])

with tab_cls:
    st.subheader("Which type of brain image is it?")
    uploaded_cls = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"], key="cls")

    if uploaded_cls is not None:
        left, right = st.columns(2)
        left.image(uploaded_cls, caption="Uploaded image", use_container_width=True)

        if right.button("Classify", key="btn_cls"):
            with st.spinner("Classifying..."):
                response = send_image(CLASSIFICATION_URL + "/predict", uploaded_cls)
            if response is not None:
                result = response.json()
                right.metric("Predicted class", result["predicted_class"])
                right.metric("Confidence", str(round(result["confidence"] * 100, 1)) + " %")

                table = pd.DataFrame({
                    "class": list(result["probabilities"].keys()),
                    "probability": list(result["probabilities"].values()),
                }).set_index("class")
                right.bar_chart(table)

with tab_det:
    st.subheader("Where is the tumor?")
    uploaded_det = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"], key="det")
    confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)

    if uploaded_det is not None:
        left, right = st.columns(2)
        left.image(uploaded_det, caption="Uploaded image", use_container_width=True)

        if right.button("Detect", key="btn_det"):
            with st.spinner("Detecting..."):
                drawn = send_image(DETECTION_URL + "/predict/image", uploaded_det, {"confidence": confidence})
                info = send_image(DETECTION_URL + "/predict", uploaded_det, {"confidence": confidence})

            if drawn is not None:
                right.image(drawn.content, caption="Detections", use_container_width=True)

            if info is not None:
                data = info.json()
                if data["count"] == 0:
                    st.warning("No detection above this confidence. Try a lower value.")
                else:
                    st.write(str(data["count"]) + " detection(s):")
                    rows = []
                    for d in data["detections"]:
                        rows.append({
                            "class": d["class"],
                            "confidence": round(d["confidence"], 3),
                            "x1": round(d["box"]["x1"]), "y1": round(d["box"]["y1"]),
                            "x2": round(d["box"]["x2"]), "y2": round(d["box"]["y2"]),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)