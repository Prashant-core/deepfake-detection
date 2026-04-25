import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from PIL import Image
import numpy as np
import os

# --- 1. THE COMPATIBILITY SHIELD ---
# This class fixes the 'batch_shape' and 'shape' errors automatically
class UniversalInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # We ensure the layer always has the correct 160x160 shape 
        # regardless of what the old file says
        kwargs['input_shape'] = (160, 160, 3)
        kwargs.pop('batch_shape', None)
        kwargs.pop('batch_input_shape', None)
        kwargs.pop('optional', None)
        kwargs.pop('sparse', None)
        kwargs.pop('ragged', None)
        super().__init__(**kwargs)

custom_objects = {
    'InputLayer': UniversalInputLayer,
    'DTypePolicy': lambda **x: None
}

# --- 2. SCIENTIFIC DARK THEME SETUP ---
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

# --- 3. FORENSIC ENGINE ---
@st.cache_resource
def load_engine():
    model_path = 'model/deepfake_final.h5'
    if os.path.exists(model_path):
        try:
            # We use compile=False to save RAM and custom_objects for compatibility
            return tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects, 
                compile=False
            )
        except Exception as e:
            st.error(f"⚠️ Engine Offline: {e}")
            return None
    return None

model = load_engine()

# --- 4. UI LAYOUT ---
st.markdown("<h1 class='stHeader'>🛡️ DEEPFAKE SHIELD: NEURAL FORENSIC UNIT</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛠️ System Controls")
    st.info("ENGINE STATUS: ONLINE")
    st.markdown("---")
    st.write("Model: MobileNetV2-Forensic")
    st.write("Patch: Universal-Shape-Fix")
    st.write("Platform: Production v1.1.0")

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
        # Fast Analysis (Removed time.sleep for speed)
        with st.status("Analyzing Neural Patterns...", expanded=True) as status:
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
        
        with st.expander("🔬 View Metadata"):
            st.metric("Raw AI Score", f"{prediction:.4f}")
    else:
        st.info("WAITING FOR TARGET INPUT...")