import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="Deepfake Shield AI", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stHeader { color: #00ffcc; text-align: center; font-family: 'Courier New', Courier, monospace; }
    .status-box { border: 2px solid #00ffcc; padding: 20px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True) # FIXED: Changed from unsafe_allow_whitespace

st.markdown("<h1 class='stHeader'>🛡️ DEEPFAKE FORENSIC PORTAL</h1>", unsafe_allow_html=True)
st.write("---")

# 2. Model Loading Logic
model_path = 'model/deepfake_final.h5'

@st.cache_resource
def load_forensic_model():
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_forensic_model()

# 3. User Interface
uploaded_file = st.file_uploader("Upload target image for forensic analysis...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Analysis Target', use_column_width=True)
    
    if model is not None:
        with st.spinner('Neural Network scanning for pixel inconsistencies...'):
            # Preprocessing to match training (160x160)
            img = image.resize((160, 160))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            prediction = model.predict(img_array)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
        # 4. Results Display
        st.write("---")
        if prediction > 0.5:
            st.error(f"🚨 VERDICT: DEEPFAKE DETECTED ({confidence*100:.2f}% Confidence)")
        else:
            st.success(f"✅ VERDICT: REAL IMAGE ({confidence*100:.2f}% Confidence)")
    else:
        st.error("Model file not found. Please ensure 'model/deepfake_final.h5' exists.")