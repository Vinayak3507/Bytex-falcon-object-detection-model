import streamlit as st
import os
from PIL import Image
import pandas as pd
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
# OBJECT DETECTION PAGE (FAKE DEMO – DEPLOY SAFE)
# =========================================
elif page == "🔍 Object Detection":
    st.title("🔍 Object Detection (Demo Mode)")
    st.info("⚠ YOLO model disabled for Streamlit Cloud. Showing demo detections.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Fake detection output
        st.subheader("🧪 Demo Detection Results")
        st.json({
            "Fire Extinguisher": 1,
            "First Aid Box": 1,
            "Oxygen Tank": 0,
            "Nitrogen Tank": 0,
            "Emergency Phone": 1
        })

        st.success("✔ Detection complete (demo mode).")

        # Fake output download
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            "⬇ Download Output",
            buf.getvalue(),
            "prediction_demo.png",
            "image/png"
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

    # Show graphs
    st.subheader("📈 Training Curves")
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.endswith(".png"):
                st.image(os.path.join(results_dir, f), caption=f)
    else:
        st.warning("training_results/ folder missing.")


# =========================================
# SAVED PREDICTIONS PAGE
# =========================================
elif page == "🖼 Saved Predictions":
    st.title("🖼 Saved Prediction Samples")

    pred_dir = "predictions"

    if os.path.exists(pred_dir):
        imgs = [f for f in os.listdir(pred_dir) if f.lower().endswith(("png", "jpg"))]
        if imgs:
            for i in imgs:
                st.image(os.path.join(pred_dir, i), caption=i)
        else:
            st.warning("No prediction images found.")
    else:
        st.error("predictions/ folder missing.")