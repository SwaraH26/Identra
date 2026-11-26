# Identra
## Advanced Deep Learning Facial Recognition & Verification System

**Experiential Learning Project | Artificial Neural Networks and Deep Learning**  
**Student:** Swara  
**Version:** 1.0  

---

## 📌 Project Overview

This project implements an end-to-end **Biometric Face Recognition and Verification System** powered by deep Convolutional Neural Networks (CNNs), transfer learning, face alignment, and metric distance learning.

The system supports:

1. **User Enrollment**: Captures face crops, assesses anti-spoofing liveness, and extracts 512-dimensional feature embedding vectors.
2. **1:1 Biometric Verification**: Confirms whether a test face belongs to a specific claimed identity.
3. **1:N Biometric Identification**: Matches an unknown face against the entire enrolled database.
4. **Anti-Spoofing / Liveness Detection**: Analyzes texture micro-sharpness (Laplacian variance) and high-frequency Fourier energy to prevent photo and screen presentation attacks.
5. **Academic Benchmarking**: Computes **Accuracy**, **FAR (False Acceptance Rate)**, **FRR (False Rejection Rate)**, and **EER (Equal Error Rate)**, generating publication-ready evaluation charts.

---

## 🏗️ Architecture & Modules

```text
deep-learning el/
├── config.py                 # Central configurations (thresholds, image dimensions, paths)
├── app.py                    # Interactive Streamlit Web Application (Webcam + Photo Upload)
├── evaluate.py               # Automated benchmarking and ablation evaluation script
├── generate_sample_data.py   # Multi-identity benchmark and test generator
├── requirements.txt          # Python dependencies
├── database/
│   ├── enrolled_users.json   # Biometric database storing identities and embeddings
│   └── faces/                # Standardized enrolled face crops
├── evaluation_results/
│   ├── architecture_comparison.csv          # Quantitative comparison across backbones
│   └── biometric_evaluation_benchmark.png   # 4-panel evaluation figure
└── src/
    ├── face_detector.py      # OpenCV Haar Cascade detection, margin padding, normalization
    ├── feature_extractor.py  # Deep CNN encoders (MobileNetV2, ResNet18, Custom 5-layer CNN)
    ├── liveness_detector.py  # Frequency & texture anti-spoofing analyzer
    ├── face_recognition_engine.py  # Enrollment, matching, decision logic
    └── evaluation.py         # FAR, FRR, EER, ROC-AUC metric calculations
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Application

Run the Streamlit application to access the graphical interface:

```bash
streamlit run app.py
```

This opens a local browser interface (`http://localhost:8501`) featuring:

* **Tab 1 - Verification & Identification**: Live webcam or photo matching with visual bounding boxes, real-time similarity gauge, and Accepted/Rejected status banners.
* **Tab 2 - User Enrollment**: Register new identities with live webcam capture or photo upload.
* **Tab 3 - Database Gallery**: Inspect enrolled identities and manage stored templates.
* **Tab 4 - Performance Analytics**: View benchmark metrics, ROC curves, and score distributions.

### 3. Run Automated Benchmarking

To re-run the benchmark across all CNN backbones and regenerate the ROC/EER plots:

```bash
python evaluate.py
```

---

## 📊 Experimental Results & Architecture Comparison

Evaluated on 144 multi-condition genuine and impostor face pairs across lighting, orientation, and sensor variations:

| CNN Architecture | Transfer Learning Backbone | Top Accuracy | Equal Error Rate (EER) | ROC-AUC | Optimal Threshold ($\tau^*$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MobileNetV2** | Pretrained ImageNet | **84.03%** | **16.25%** | **0.909** | **0.76** |
| **ResNet18** | Pretrained ImageNet | 77.78% | 21.67% | 0.876 | 0.89 |
| **Custom Face CNN** | Built from scratch (5 Conv layers) | 70.83% | 32.50% | 0.851 | 0.99 |

### Metric Definitions

* **FAR (False Acceptance Rate)**: Probability that an impostor face is erroneously accepted.
* **FRR (False Rejection Rate)**: Probability that an enrolled genuine face is erroneously rejected.
* **EER (Equal Error Rate)**: Operating threshold where $\text{FAR} = \text{FRR}$. Lower indicates higher discriminative capability.
* **ROC-AUC**: Area under the Receiver Operating Characteristic curve ($1.0$ indicates perfect separation).

---

## 🛡️ PRD Requirement Traceability

| PRD Requirement ID | Description | Implementation Details |
| :--- | :--- | :--- |
| **FR1 & FR2** | Capture face & detect/crop | `FaceDetector` in `src/face_detector.py` using Haar Cascades + margin alignment. |
| **FR5** | Convert face to feature vector | `FaceFeatureExtractor` in `src/feature_extractor.py` producing L2-normalized 512-d embeddings. |
| **FR6** | User registration / database | `enroll_user()` in `src/face_recognition_engine.py` saving to `database/enrolled_users.json`. |
| **FR7** | Distance / similarity measure | Calibrated Cosine Similarity on unit vectors: $\cos(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$. |
| **FR8 & FR9** | Score thresholding & Decision | Acceptance logic: $S \ge \tau \implies \text{Accepted}$, otherwise $\text{Rejected}$. |
| **FR10 & Sec 11** | Accuracy, FAR, FRR, EER evaluation | `BiometricEvaluator` in `src/evaluation.py` producing 4-panel visual report. |
| **Sec 13 Risks** | Anti-spoofing protection | `LivenessDetector` in `src/liveness_detector.py` analyzing Laplacian variance and FFT high-frequency content. |

---

## 🔒 Privacy & Data Protection

Personal biometric images, enrolled face databases, and generated evaluation data are excluded from version control using `.gitignore`.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Scikit-learn
* Scikit-image
* Matplotlib
* Pandas
* Pillow
* Streamlit

---

## 👩‍💻 Author

**Swara**

Experiential Learning Project  
Artificial Neural Networks and Deep Learning

---

**Repository:** https://github.com/SwaraH26/Identra
