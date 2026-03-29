import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. Professional UI Setup
st.set_page_config(page_title="Deepfake Shield AI", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stHeader { color: #00ffcc; text-align: center; font-family: 'Courier New', Courier, monospace; }
    .verdict-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='stHeader'>🛡️ DEEPFAKE FORENSIC PORTAL</h1>", unsafe_allow_html=True)
st.write("---")

# 2. Model Loading (Cached for Speed)
model_path = 'model/deepfake_final.h5'

@st.cache_resource
def load_forensic_model():
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_forensic_model()

# 3. Analysis Interface
uploaded_file = st.file_uploader("Upload target image for forensic analysis...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Ensure RGB format for the Neural Network
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Analysis Target', use_container_width=True)
    
    if model is not None:
        with st.spinner('Running Neural Network Forensic Scan...'):
            # --- STEP 1: PREPROCESSING (Critical Fix) ---
            img = image.resize((160, 160))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Use official MobileNetV2 scaling (-1 to 1)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # --- STEP 2: PREDICTION ---
            prediction = model.predict(img_array)[0][0]
            
            # --- STEP 3: LABEL LOGIC (The "Flip" Fix) ---
            # If your Real images show a high score (close to 1), set this to False.
            # If your Real images show a low score (close to 0), set this to True.
            invert_labels = True 
            
            if invert_labels:
                is_fake = prediction < 0.5
                confidence = (1 - prediction) if is_fake else prediction
            else:
                is_fake = prediction > 0.5
                confidence = prediction if is_fake else (1 - prediction)

        # 4. Final Verdict Display
        st.write("---")
        if is_fake:
            st.error(f"🚨 VERDICT: DEEPFAKE DETECTED ({confidence*100:.2f}% Confidence)")
        else:
            st.success(f"✅ VERDICT: REAL IMAGE ({confidence*100:.2f}% Confidence)")
            
        # Forensic Debug Info
        with st.expander("View Forensic Raw Data"):
            st.write(f"Raw Model Probability Score: `{prediction:.4f}`")
            st.write("Note: Scores closer to 1.0 indicate high probability of GAN-generated artifacts.")
    else:
        st.error("Forensic Engine (model) not found. Please ensure 'model/deepfake_final.h5' is synced.")