# 🛡️ Deepfake Shield AI
### Senior Capstone: Image Forensic System for Synthetic Media Detection

This project identifies AI-generated facial manipulations by analyzing pixel-level inconsistencies that are invisible to the human eye. It was developed to provide a lightweight, real-time solution for verifying the authenticity of social media and ID profile captures.

## 🚀 The Tech Stack
* **Deep Learning:** MobileNetV2 (TensorFlow/Keras)
* **Methodology:** Transfer Learning + Fine-tuning (Top 30 layers)
* **Deployment:** Streamlit (Local Web Dashboard)
* **Hardware:** Optimized for NVIDIA RTX 5050 (Mixed Precision Training)

## 📊 Results & Benchmarks
After 10 epochs of training on a balanced dataset of 140k images:
* **Final Accuracy:** 92.4% (Verified on Test Set)
* **Inference Speed:** ~140ms per scan
* **Key Insight:** The model specializes in detecting "boundary blurring" and "frequency artifacts" common in GAN-generated images.

## 📁 Repository Guide
* `train.py`: The training pipeline featuring data augmentation and early stopping.
* `app.py`: The forensic portal UI (Upload -> Scan -> Verdict).
* `model/`: Contains the optimized weights (`deepfake_final.h5`).
* `.gitignore`: Configured to keep the 3GB+ raw dataset local while sharing the logic.

## ⚙️ How to Run
1. Activate environment: `.\venv\Scripts\Activate.ps1`
2. Run Dashboard: `python -m streamlit run app.py`

---
*Created by Prashant Kumar