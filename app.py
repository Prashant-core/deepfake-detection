import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from PIL import Image
import numpy as np
import os
import time

class CompatInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs and 'input_shape' not in kwargs:
            bs = kwargs.pop('batch_shape')
            if bs is not None:
                kwargs['input_shape'] = tuple(bs[1:])
        elif 'batch_input_shape' in kwargs and 'input_shape' not in kwargs:
            bs = kwargs.pop('batch_input_shape')
            if bs is not None:
                kwargs['input_shape'] = tuple(bs[1:])
        else:
            kwargs.pop('batch_shape', None)
            kwargs.pop('batch_input_shape', None)
        kwargs.pop('optional', None)
        kwargs.pop('sparse', None)
        kwargs.pop('ragged', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config and 'input_shape' not in config:
            bs = config.pop('batch_shape')
            if bs is not None:
                config['input_shape'] = tuple(bs[1:])
        config.pop('batch_input_shape', None)
        config.pop('optional', None)
        config.pop('sparse', None)
        config.pop('ragged', None)
        return cls(**config)

custom_objects = {
    'InputLayer': CompatInputLayer,
    'DTypePolicy': lambda **kwargs: None,
}

st.set_page_config(
    page_title="DeepFake Detection | AI Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] > div {
    background: #020408 !important; color: #c8d6e5;
}
[data-testid="stHeader"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"]::before {
    content: ''; position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(0,255,136,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,136,0.025) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
}
.dfd-header {
    position: relative; z-index: 10;
    background: linear-gradient(180deg,#040c14 0%,#020408 100%);
    border-bottom: 1px solid rgba(0,255,136,0.15);
    padding: 22px 48px; display: flex; align-items: center; gap: 18px;
}
.dfd-header-icon {
    width: 48px; height: 48px; background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3); border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.dfd-header-text h1 {
    font-family: 'Orbitron', monospace; font-size: 1.4rem; font-weight: 700;
    color: #ffffff; letter-spacing: 4px; line-height: 1; margin: 0;
}
.dfd-header-text p {
    font-family: 'Inter', sans-serif; font-size: 0.68rem;
    color: rgba(0,255,136,0.6); letter-spacing: 5px; margin-top: 5px;
}
.dfd-header-badges { margin-left: auto; display: flex; align-items: center; gap: 10px; }
.badge {
    font-family: 'Inter', sans-serif; font-size: 0.65rem;
    letter-spacing: 2px; text-transform: uppercase; padding: 5px 14px; border-radius: 20px;
}
.badge-online {
    color: #00ff88; border: 1px solid rgba(0,255,136,0.4);
    background: rgba(0,255,136,0.07); display: flex; align-items: center; gap: 7px;
}
.badge-model { color: #64748b; border: 1px solid rgba(100,116,139,0.3); background: rgba(100,116,139,0.05); }
.pulse-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #00ff88;
    animation: pulse 2s ease-in-out infinite; box-shadow: 0 0 6px #00ff88;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.8)} }
.dfd-body { position: relative; z-index: 1; padding: 36px 48px; }
.section-label {
    font-family: 'Inter', sans-serif; font-size: 0.65rem; letter-spacing: 4px;
    color: rgba(0,255,136,0.5); text-transform: uppercase; margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: rgba(0,255,136,0.1); }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,255,136,0.025) !important;
    border: 1px dashed rgba(0,255,136,0.2) !important; border-radius: 12px !important;
}
[data-testid="stImage"] img {
    border-radius: 12px !important; border: 1px solid rgba(0,255,136,0.15) !important;
}
.meta-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.meta-cell {
    background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px; padding: 9px 12px;
}
.meta-k {
    font-family: 'Inter', sans-serif; font-size: 0.58rem;
    letter-spacing: 3px; color: #334155; text-transform: uppercase; margin-bottom: 3px;
}
.meta-v { font-family: 'Orbitron', monospace; font-size: 0.82rem; color: #64748b; }
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important; font-size: 1rem !important; color: #94a3b8 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important; font-size: 0.62rem !important;
    letter-spacing: 2px !important; color: #334155 !important; text-transform: uppercase !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 10px !important; padding: 12px 14px !important;
}
.dfd-footer {
    position: relative; z-index: 1; border-top: 1px solid rgba(255,255,255,0.04);
    padding: 16px 48px; display: flex; justify-content: space-between;
    font-family: 'Inter', sans-serif; font-size: 0.62rem;
    color: #1e293b; letter-spacing: 2px; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model_cached():
    model_path = 'model/deepfake_final.h5'
    if not os.path.exists(model_path):
        return None, f"Model file not found at `{model_path}`."
    try:
        m = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return m, None
    except Exception as e1:
        try:
            m = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            return m, None
        except Exception:
            pass
        return None, str(e1)

def run_inference(model, image: Image.Image):
    """
    Your model uses Convention: raw_score near 1.0 = REAL, near 0.0 = FAKE
    Evidence: fake_10001.jpg returned 0.0000 and was wrongly called Authentic.
    Fix: is_fake = raw_score < 0.5
    """
    try:
        _, h, w, _ = model.input_shape
        target_size = (int(h or 160), int(w or 160))
    except Exception:
        target_size = (160, 160)

    img = image.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    raw_score = float(model.predict(arr, verbose=0)[0][0])

    # raw_score = P(Real). Low score = Fake.
    is_fake    = raw_score < 0.5
    real_prob  = raw_score
    fake_prob  = 1.0 - raw_score
    confidence = fake_prob if is_fake else real_prob
    return is_fake, confidence, fake_prob, real_prob, raw_score, target_size

# HEADER
st.markdown("""
<div class="dfd-header">
    <div class="dfd-header-icon">🔍</div>
    <div class="dfd-header-text">
        <h1>DEEPFAKE DETECTION</h1>
        <p>AI Image Forensics · Neural Analysis Engine</p>
    </div>
    <div class="dfd-header-badges">
        <span class="badge badge-model">MobileNetV2</span>
        <span class="badge badge-online"><div class="pulse-dot"></div>ENGINE ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="dfd-body">', unsafe_allow_html=True)

model, load_error = load_model_cached()
if load_error:
    st.error(f"Engine Error: {load_error}")

col_left, _, col_right = st.columns([5, 0.3, 6])

with col_left:
    st.markdown('<div class="section-label">Input Stream</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        st.markdown(f"""
        <div class="meta-row">
            <div class="meta-cell"><div class="meta-k">Filename</div><div class="meta-v">{uploaded_file.name[:22]}</div></div>
            <div class="meta-cell"><div class="meta-k">Dimensions</div><div class="meta-v">{image.size[0]}x{image.size[1]}</div></div>
            <div class="meta-cell"><div class="meta-k">Mode</div><div class="meta-v">{image.mode}</div></div>
            <div class="meta-cell"><div class="meta-k">File Size</div><div class="meta-v">{uploaded_file.size // 1024} KB</div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1.5px dashed rgba(0,255,136,0.12);border-radius:12px;
                    padding:48px 24px;text-align:center;margin-top:8px;">
            <div style="font-size:2.5rem;margin-bottom:14px;opacity:0.35;">🖼️</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.8rem;
                        letter-spacing:2px;color:#334155;">UPLOAD A FACE IMAGE TO BEGIN</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.7rem;
                        color:#1e293b;margin-top:8px;">JPG · PNG · JPEG</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-label">Forensic Analysis</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.markdown("""
        <div style="height:300px;border:1px dashed rgba(0,255,136,0.1);border-radius:14px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;">
            <div style="font-size:2.8rem;filter:grayscale(1);opacity:0.4;">🔬</div>
            <div style="font-family:'Orbitron',monospace;font-size:0.65rem;
                        letter-spacing:5px;color:#1e293b;">AWAITING TARGET INPUT</div>
        </div>
        """, unsafe_allow_html=True)

    elif not model:
        st.error("Model not loaded.")

    else:
        with st.spinner("Scanning neural patterns..."):
            t0 = time.time()
            is_fake, confidence, fake_prob, real_prob, raw_score, target_size = run_inference(model, image)
            elapsed = time.time() - t0

        conf_pct = confidence * 100

        if is_fake:
            bg    = "rgba(255,59,48,0.07)"
            brd   = "rgba(255,59,48,0.35)"
            glow  = "#ff3b30"
            title = "DEEPFAKE DETECTED"
            sub   = "Neural analysis found synthetic artifacts consistent with AI generation or GAN manipulation."
            top_line = "background:linear-gradient(90deg,transparent,#ff3b30,transparent)"
        else:
            bg    = "rgba(0,255,136,0.04)"
            brd   = "rgba(0,255,136,0.28)"
            glow  = "#00ff88"
            title = "IMAGE IS AUTHENTIC"
            sub   = "No significant synthetic artifacts detected. Image appears to be genuine photographic content."
            top_line = "background:linear-gradient(90deg,transparent,#00ff88,transparent)"

        st.markdown(
            f"<div style='background:{bg};border:1px solid {brd};border-radius:14px;"
            f"padding:28px 30px;position:relative;overflow:hidden;'>"
            f"<div style='position:absolute;top:0;left:0;right:0;height:1px;{top_line};'></div>"
            f"<div style='font-family:Inter,sans-serif;font-size:0.62rem;letter-spacing:4px;"
            f"color:#475569;text-transform:uppercase;margin-bottom:10px;'>Forensic Verdict</div>"
            f"<div style='font-family:Orbitron,monospace;font-size:1.9rem;font-weight:700;"
            f"color:{glow};letter-spacing:2px;text-shadow:0 0 28px {glow}44;"
            f"margin-bottom:8px;'>{'⚠ ' if is_fake else '✓ '}{title}</div>"
            f"<div style='font-family:Inter,sans-serif;font-size:0.85rem;color:#64748b;"
            f"line-height:1.5;'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        bar_col = "#ff3b30" if is_fake else "#00ff88"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-family:Inter,sans-serif;"
            f"font-size:0.7rem;color:#475569;letter-spacing:2px;text-transform:uppercase;"
            f"margin-bottom:6px;'><span>Confidence in verdict</span>"
            f"<span style='font-size:0.95rem;font-weight:600;color:{bar_col};'>{conf_pct:.1f}%</span></div>",
            unsafe_allow_html=True
        )
        st.progress(float(confidence))
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Score", f"{raw_score:.4f}")
        c2.metric("Fake Prob", f"{fake_prob*100:.1f}%")
        c3.metric("Real Prob", f"{real_prob*100:.1f}%")
        c4.metric("Scan Time", f"{elapsed:.2f}s")

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        if is_fake:
            st.warning("⚠️ GAN-style artifacts or lighting inconsistencies detected. High-end manually retouched fakes may still evade detection.")
        else:
            st.success("✅ Passed forensic checks. Heavy compression or filters may lower confidence. Best with uncompressed face-forward images.")

        with st.expander("🔬 Technical Details"):
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Model | MobileNetV2 |
| Input size | {target_size[0]}x{target_size[1]} px |
| Preprocessing | MobileNetV2 (range: -1 to 1) |
| Score convention | Near 1.0 = Real · Near 0.0 = Fake |
| Raw sigmoid output | `{raw_score:.6f}` |
| Analysis time | `{elapsed:.3f}s` |
            """)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="dfd-footer">
    <span>Deepfake Detection · AI Forensics</span>
    <span>MobileNetV2 · TensorFlow · Streamlit</span>
    <span>Built by Prashant</span>
</div>
""", unsafe_allow_html=True)