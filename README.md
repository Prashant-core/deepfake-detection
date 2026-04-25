# 🛡️ Deepfake Detection : Neural Forensic Unit

**Live Demo:** [Open Deepfake Shield Portal](https://deepfake-detection-m6wlt6zljowdxneyvbip5h.streamlit.app/)

### 🛠️ Tech Stack & Libraries
The engine is built on a high-efficiency **MobileNetV2** architecture optimized for real-time cloud inference.

* **TensorFlow / Keras:** Primary framework for the CNN architecture and model weights.
* **Streamlit:** Web-native interface for the forensic dashboard.
* **Pillow (PIL):** Pre-processing, pixel normalization, and image scaling.
* **NumPy:** Linear algebra for handling image tensors and probability arrays.
* **OpenCV:** Utility for computer vision and asset decoding.

---

### ✅ Pros & ❌ Cons

| Pros | Cons |
| :--- | :--- |
| **Low Latency:** High-speed scanning using MobileNetV2. | **Filter Sensitivity:** Can flag heavy filters as "Synthetic." |
| **User-Centric UI:** Designed for clarity with zero-tuning required. | **Compression Loss:** Social media compression can hide artifacts. |
| **Privacy First:** Assets are processed in-session and never stored. | **Single Subject:** Accuracy peaks with a single subject in frame. |

---

### 📡 System Capabilities

**What it can do:**
* **Detect GAN Signatures:** Identifies mathematical inconsistencies in skin texture and lighting.
* **Probabilistic Verdicts:** Provides a confidence percentage rather than just a "Yes/No" guess.
* **Neural Decomposition:** Scales and normalizes assets to 160x160 for precise artifact scanning.

**What it can't do:**
* **Video Analysis:** Currently limited to static image files (`.jpg`, `.png`, `.jpeg`).
* **Human-Cleaned Fakes:** High-end fakes manually touched up by an artist may bypass detection.
* **Contextual Analysis:** The AI detects "AI Pixels" but doesn't understand the intent of the image.

---

### 🔧 Local Installation
1. **Clone:** `git clone https://github.com/Prashant-core/deepfake-detection.git`
2. **Setup:** `pip install -r requirements.txt`
3. **Launch:** `streamlit run app.py`

---
**Author:** Prashant