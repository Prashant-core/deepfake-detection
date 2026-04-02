import streamlit as st
import os
import time

# --- CRITICAL CLOUD COMPATIBILITY IMPORTS ---
import tensorflow as tf
import tf_keras as keras  # Use legacy keras for .h5 compatibility
import numpy as np
from PIL import Image

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
    [data-testid="stMetricValue"] { color: #00ffcc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FORENSIC ENGINE ---
@st.cache_resource
def load_engine():
    model_path = 'model/deepfake_final.h5'
    if os.path.exists(model_path):
        try:
            # Loading using the tf_keras legacy library to fix the TypeError
            return keras.models.load_model(model_path, compile=False)
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
    st.write("Architecture: Keras-Legacy")
    st.write("Platform: Cloud-Production v1.0.8")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📡 Input Stream")
    uploaded_file = st.file_uploader("Upload Image for Scanning...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="TARGET_IDENTIFIED.JPG", use_container_width=True)

with col2:
    st.subheader("📊 Forensic Analysis")
    
    if uploaded_file and model:
        with st.status("Initiating Neural Decomposition...", expanded=True) as status:
            st.write("⚡ Extracting pixel-level artifacts...")
            time.sleep(0.4)
            
            # --- AI LOGIC ---
            img = image.resize((160, 160))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            prediction = model.predict(img_array)[0][0]
            
            # 1.0 is REAL, 0.0 is FAKE
            is_fake = prediction < 0.5 
            confidence = (1 - prediction) if is_fake else prediction
            
            st.write("🔍 Running Frequency Domain Analysis...")
            time.sleep(0.4)
            status.update(label="ANALYSIS COMPLETE", state="complete", expanded=False)

        # Report Card
        st.markdown("<div class='report-card'>", unsafe_allow_html=True)
        if is_fake:
            st.markdown(f"<h2 style='color:#ff4b4b;'>🚨 VERDICT: DEEPFAKE DETECTED</h2>", unsafe_allow_html=True)
            st.error(f"Integrity Compromised: {confidence*100:.2f}% Probable Manipulation")
        else:
            st.markdown(f"<h2 style='color:#00ffcc;'>✅ VERDICT: ASSET AUTHENTIC</h2>", unsafe_allow_html=True)
            st.success(f"Integrity Verified: {confidence*100:.2f}% Probability Real")
        
        st.progress(float(confidence))
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔬 Technical Metadata"):
            m1, m2 = st.columns(2)
            m1.metric("Raw AI Score", f"{prediction:.4f}")
            m2.metric("Threshold", "0.5000")

    else:
        st.info("WAITING FOR TARGET INPUT...")