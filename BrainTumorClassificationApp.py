import streamlit as st
import numpy as np
import joblib
import cv2
from utils.preprocessing import preprocess_image
from utils.features import feature_extraction
import os

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Brain MRI Classifier", layout="wide", page_icon="🧠")
st.markdown("""
<style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 6px solid #4a90e2;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Brain MRI Tumor Classifier")
st.markdown("Upload an MRI image and let the model analyze it using texture-based features (GLCM).")

# -----------------------------
# ✅ Model Path (loads from models folder)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_bundle.pkl")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]

# -----------------------------
# UI Image Uploader
# -----------------------------
uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])
process = st.button("Process Image")

# -----------------------------
# Process Image
# -----------------------------
if process:
    if uploaded is None:
        st.warning("Please upload an image before pressing Process.")
        st.stop()
    else:
        # Read image
        try:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            st.error("Unable to read the uploaded file as an image.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Input Image")
            st.image(image, width=500, channels="BGR")

        with col2:
            with st.spinner("Preprocessing and extracting features..."):
                img_arr = preprocess_image(image, target_size=(224, 224))
                features = feature_extraction(img_arr)

                # Feature mismatch protection
                if features.shape[1] != scaler.n_features_in_:
                    st.error(
                        f"Feature mismatch ❌ | Model expects {scaler.n_features_in_} features "
                        f"but got {features.shape[1]}"
                    )
                    st.stop()

                features_scaled = scaler.transform(features)

            with st.spinner("Running model inference..."):
                pred = int(model.predict(features_scaled)[0])

                label_map = {
                    0: "Pituitary Tumor",
                    1: "No Tumor Detected",
                    2: "Meningioma Tumor",
                    3: "Glioma Tumor"
                }

                class_name = label_map.get(pred, "Unknown")

            st.markdown("### ✅ Prediction Result")
            st.markdown(f"""
            <div class="prediction-card" style="background-color: #3D3D3D;">
                <h3>Predicted Class: <b>{class_name}</b></h3>
            </div>
            """, unsafe_allow_html=True)

            # Insights panel
            st.markdown("### 📘 What This Result Means")
            if pred == 0:
                st.write("A **Pituitary** tumor develops in the pituitary gland and may affect hormones.")
            elif pred == 1:
                st.write("No tumor detected. The model found no abnormal texture patterns.")
            elif pred == 2:
                st.write("A **Meningioma** tumor grows from the meninges covering the brain.")
            else:
                st.write("A **Glioma** tumor originates from glial cells in the brain.")
