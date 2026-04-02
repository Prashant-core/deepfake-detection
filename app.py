import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import time
import os

# --- THE "COMPATIBILITY SHIELD" ---
# This fixes all versioning errors (DTypePolicy, batch_shape, InputLayer) at once
from tensorflow.keras.layers import InputLayer, Conv2D

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

# Registry for the loader to recognize the modern environment
custom_objects = {
    'InputLayer': PatchedInputLayer,
    'DTypePolicy': lambda **x: None # Ignores the policy error entirely
}

# --- 1. SCIENTIFIC DARK THEME SETUP ---
st.set_page_config(page_title="Deepfake Shield | Forensic Portal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #00ffcc; font-family: 'Courier New', Courier, monospace; }
    .stHeader { color: #00ffcc; text-shadow: 0 0 15px #00ffcc; border-bottom: 2px solid #00ffcc; padding-bottom: 10px; }
    .report-card { 
        background-color: #111111; 
        border: 1px solid #00ffcc; 
        padding: 25px; 
        border-radius: 5px; 
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.1);
    }
    .stStatus { background-color: #0a0a0a; border-left: 5px solid #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FORENSIC ENGINE ---
@st.cache_resource
def load_engine():
    model_path = 'model/deepfake_final.h5'
    if os.path.exists(model_path):
        try:
            # We load using the custom objects shield
            return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        except Exception as e:
            st.error(f"Engine Load Error: {e}")
            return None
    return None

model = load_engine()

# --- 3. UI LAYOUT ---
st.markdown("<h1 class='stHeader'>🛡️ DEEPFAKE SHIELD: NEURAL FORENSIC UNIT</h1>", unsafe_allow_html=True)
st.write(" ")

with st.sidebar:
    st.markdown("### 🛠️ System Controls")
    st.info("ENGINE STATUS: ONLINE")
    st.markdown("---")
    st.write("Model: MobileNetV2-Forensic")
    st.write("Architecture: Cloud-Native 2026")
    st.write("Patch: Compatibility-Shield-v3")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📡 Input Stream")
    uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="TARGET_IDENTIFIED.JPG", use_container_width=True)

with col2:
    st.subheader("📊 Forensic Analysis")
    if uploaded_file and model:
        with st.status("Analyzing...", expanded=True) as status:
            img = image.resize((160, 160))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            prediction = model.predict(img_array)[0][0]
            
            # Logic: 1.0 = Real, 0.0 = Fake
            is_fake = prediction < 0.5 
            confidence = (1 - prediction) if is_fake else prediction
            
            status.update(label="ANALYSIS COMPLETE", state="complete", expanded=False)

        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        if is_fake:
            st.markdown(f"<h2 style='color:#ff4b4b;'>🚨 VERDICT: DEEPFAKE</h2>", unsafe_allow_html=True)
            st.error(f"Integrity Compromised: {confidence*100:.2f}% Probable Manipulation")
        else:
            st.markdown(f"<h2 style='color:#00ffcc;'>✅ VERDICT: AUTHENTIC</h2>", unsafe_allow_html=True)
            st.success(f"Integrity Verified: {confidence*100:.2f}% Probability Real")
        st.progress(float(confidence))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("WAITING FOR TARGET INPUT...")