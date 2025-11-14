import streamlit as st
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from ultralytics import YOLO
import pandas as pd
import numpy as np
from PIL import Image
import io

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Bytex Falcon – Safety Object Detection",
    page_icon="🦅",
    layout="wide"
)

# =========================================
# LOAD YOLO MODEL (ONNX VERSION FOR STREAMLIT)
# =========================================
MODEL_PATH = "model/best.onnx" 

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH) 

model = load_model()

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/743/743131.png", width=80)
    st.title("Bytex Falcon")
    st.markdown("---")

    page = st.radio(
        "📍 Navigate",
        ["🏠 Home", "🔍 Object Detection", "📊 Training Results", "🖼 Saved Predictions"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Made by: BYTEX")

# =========================================
# HOME PAGE
# =========================================
if page == "🏠 Home":
    st.title("🦅 Bytex Falcon – Safety Object Detection System")
    st.markdown("""
    ## Welcome to the Falcon Safety AI Dashboard

    This project uses a **YOLOv8 object detection model** trained to detect:
    - Oxygen Tank  
    - Nitrogen Tank  
    - First Aid Box  
    - Fire Alarm  
    - Safety Switch Panel  
    - Emergency Phone  
    - Fire Extinguisher  

    ### 🎯 Project Goal  
    To build a **real-time safety monitoring system** capable of detecting critical safety equipment in industrial and public environments.

    ### 👥 Team Members  
    - **Vinayak Dixit**  
    - **Sayed Anas**
    - **Prince Kumar Yadav**
    - **Krishna Soti**

    ---
    Use the sidebar to explore:
    - 🔍 **Run Detection**  
    - 📊 **View Training Graphs**  
    - 🖼 **Saved Predictions**  
    """)

# =========================================
# OBJECT DETECTION PAGE
# =========================================
elif page == "🔍 Object Detection":
    st.title("🔍 Object Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run ONNX YOLO
        results = model.predict(image)

        # Render YOLO output
        result_img = results[0].plot()

        st.markdown("### 🦅 YOLOv8 Output")
        st.image(result_img, use_column_width=True)

        # Count detections
        class_counts = {}
        for box in results[0].boxes:
            cls_id = int(box.cls)
            class_name = model.names[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        st.subheader("📦 Objects Detected")
        st.json(class_counts)

        # Download the prediction
        buf = io.BytesIO()
        Image.fromarray(result_img).save(buf, format="PNG")
        st.download_button(
            "⬇ Download Prediction",
            data=buf.getvalue(),
            file_name="prediction.png",
            mime="image/png"
        )

# =========================================
# TRAINING RESULTS PAGE
# =========================================
elif page == "📊 Training Results":
    st.title("📊 Training Results & Graphs")
    results_dir = "training_results"

    # Show CSV
    csv_path = os.path.join(results_dir, "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.subheader("📄 Training Metrics (results.csv)")
        st.dataframe(df)
    else:
        st.warning("results.csv not found!")

    # Show training graphs
    st.subheader("📈 Training Curves")
    graph_files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".png") and f != "results.png"
    ]
    if graph_files:
        for g in graph_files:
            st.image(os.path.join(results_dir, g), caption=g)
    else:
        st.warning("No graphs found in training_results/")

# =========================================
# SAVED PREDICTIONS PAGE
# =========================================
elif page == "🖼 Saved Predictions":
    st.title("🖼 Saved Prediction Samples")

    pred_dir = "predictions"
    if os.path.exists(pred_dir):
        files = [f for f in os.listdir(pred_dir) if f.lower().endswith(("png", "jpg"))]
        if files:
            for f in files:
                st.image(os.path.join(pred_dir, f), caption=f)
        else:
            st.warning("No prediction images found!")
    else:
        st.error("predictions/ folder not found!")