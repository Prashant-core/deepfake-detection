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
    </style>
    """, unsafe_allow_html=True)

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
    # Ensure image is in RGB mode
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Analysis Target', use_container_width=True)
    
    if model is not None:
        with st.spinner('Neural Network scanning for pixel inconsistencies...'):
            # Preprocessing to match training (160x160)
            img = image.resize((160, 160))
            img_array = np.array(img) / 255.0  # Ensure this matches your train.py scaling
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            prediction = model.predict(img_array)[0][0]
            
            # --- THE LABEL CHECK ---
            # If your model detects REAL as 1 and FAKE as 0:
            # Change "is_fake = prediction < 0.5" to "is_fake = prediction > 0.5" if results are inverted.
            is_fake = prediction < 0.5 
            confidence = (1 - prediction) if is_fake else prediction
            
        # 4. Results Display
        st.write("---")
        if is_fake:
            st.error(f"🚨 VERDICT: DEEPFAKE DETECTED ({confidence*100:.2f}% Confidence)")
        else:
            st.success(f"✅ VERDICT: REAL IMAGE ({confidence*100:.2f}% Confidence)")
            
        st.info(f"Raw Model Score: {prediction:.4f} (Closer to 0 usually means Fake, closer to 1 means Real)")
    else:
        st.error("Model file not found. Please ensure 'model/deepfake_final.h5' exists.")