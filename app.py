"""
Advanced Facial Recognition & Verification System
Interactive Web Application (Streamlit)
Experiential Learning Project | Artificial Neural Networks and Deep Learning
Student: Swara
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path
import time

import config
from src.face_detector import FaceDetector
from src.feature_extractor import FaceFeatureExtractor
from src.face_recognition_engine import FaceRecognitionEngine
from src.evaluation import BiometricEvaluator

# Page Configuration
st.set_page_config(
    page_title="Deep Learning Facial Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    .status-accepted {
        background-color: #DCFCE7;
        color: #166534;
        padding: 16px;
        border-radius: 8px;
        font-size: 1.25rem;
        font-weight: bold;
        text-align: center;
        border: 1px solid #86EFAC;
    }
    .status-rejected {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 16px;
        border-radius: 8px;
        font-size: 1.25rem;
        font-weight: bold;
        text-align: center;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)

# Cache Face Engine in session state
@st.cache_resource
def get_engine(backbone: str, threshold: float, enable_liveness: bool):
    return FaceRecognitionEngine(
        backbone=backbone,
        embedding_dim=config.EMBEDDING_DIM,
        verification_threshold=threshold,
        enable_liveness=enable_liveness
    )

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/facial-recognition.png", width=70)
    st.title("System Controls")
    
    st.subheader("Model Configuration")
    backbone_choice = st.selectbox(
        "CNN Backbone Architecture",
        options=["mobilenet_v2", "resnet18", "custom_cnn"],
        index=0,
        help="MobileNetV2: Fast & lightweight | ResNet18: Deep residual representations | Custom CNN: Built from scratch"
    )
    
    threshold_slider = st.slider(
        "Decision Threshold (τ)",
        min_value=0.40,
        max_value=0.95,
        value=config.DEFAULT_VERIFICATION_THRESHOLD,
        step=0.01,
        help="Matches with similarity >= threshold are accepted."
    )
    
    liveness_toggle = st.toggle(
        "Anti-Spoofing / Liveness Check",
        value=True,
        help="Rejects flat screen replays and printed photo spoofing attacks."
    )

    st.divider()
    
    # Initialize engine
    engine = get_engine(backbone_choice, threshold_slider, liveness_toggle)
    engine.threshold = threshold_slider
    
    enrolled_users = engine.list_users()
    st.metric("Enrolled Identities", len(enrolled_users))
    
    if st.button("🔄 Reload Database"):
        engine.load_database()
        st.rerun()

    st.markdown("---")
    st.caption("**Student:** Swara")
    st.caption("**Course:** ANN & Deep Learning")
    st.caption("**Project:** Experiential Learning")

# --- HEADER ---
st.markdown('<div class="main-header">Advanced Facial Recognition & Verification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Experiential Learning Project | Artificial Neural Networks and Deep Learning</div>', unsafe_allow_html=True)

# --- MAIN TABS ---
tab_verify, tab_enroll, tab_gallery, tab_eval = st.tabs([
    "🔍 1. Verification & Identification",
    "📝 2. User Enrollment",
    "👥 3. Database Gallery",
    "📊 4. Benchmark & Performance Evaluation"
])

# ==========================================
# TAB 1: VERIFICATION & IDENTIFICATION
# ==========================================
with tab_verify:
    st.subheader("Biometric Verification & Identification")
    st.write("Test an individual's face image against enrolled database identities.")

    col_input, col_result = st.columns([1, 1], gap="medium")

    with col_input:
        input_mode = st.radio("Input Source", ["Upload Photo", "Webcam Snapshot"], horizontal=True)
        test_img_bgr = None

        if input_mode == "Upload Photo":
            uploaded_file = st.file_uploader("Upload Face Image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="verify_upload")
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                test_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB), caption="Input Image", use_container_width=True)

        else:
            camera_snap = st.camera_input("Capture Webcam Photo", key="verify_cam")
            if camera_snap:
                file_bytes = np.asarray(bytearray(camera_snap.read()), dtype=np.uint8)
                test_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        match_mode = st.radio(
            "Matching Strategy",
            ["1:N Identification (Find Who It Is)", "1:1 Verification (Verify Claimed Identity)"],
            horizontal=True
        )

        claimed_id = None
        if match_mode == "1:1 Verification (Verify Claimed Identity)":
            if not enrolled_users:
                st.warning("No enrolled users found. Please enroll at least one user in Tab 2.")
            else:
                user_options = {u["user_id"]: f"{u['name']} ({u['user_id']})" for u in enrolled_users}
                claimed_id = st.selectbox("Claimed Identity", options=list(user_options.keys()), format_func=lambda x: user_options[x])

        verify_btn = st.button("🚀 Run Biometric Matching", type="primary", use_container_width=True, disabled=(test_img_bgr is None))

    with col_result:
        st.subheader("Verification Outcome")
        if verify_btn and test_img_bgr is not None:
            with st.spinner("Processing face detection, feature extraction, and neural scoring..."):
                start_t = time.time()
                
                if match_mode == "1:N Identification (Find Who It Is)":
                    is_accepted, msg, res = engine.identify_user(test_img_bgr, threshold=threshold_slider)
                else:
                    is_accepted, msg, res = engine.verify_user(claimed_id, test_img_bgr, threshold=threshold_slider)
                
                elapsed = time.time() - start_t

            if "face_rgb" in res:
                # Bounding box visual
                annotated_img = engine.detector.draw_face_box(
                    test_img_bgr, res["bbox"],
                    label=res.get("matched_name", res.get("user_name", "Face")),
                    score=res.get("similarity"),
                    is_accepted=is_accepted
                )
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Face Detection & Alignment", use_container_width=True)

            # Decision Banner
            if is_accepted:
                st.markdown(f'<div class="status-accepted">✅ ACCEPTED<br><span style="font-size:1.1rem; font-weight:normal;">Identity: {res.get("matched_name", res.get("user_name"))}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-rejected">❌ REJECTED<br><span style="font-size:1rem; font-weight:normal;">{msg}</span></div>', unsafe_allow_html=True)

            st.write("")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Similarity Score", f"{res.get('similarity', 0.0):.1%}")
            with m_col2:
                st.metric("Decision Threshold", f"{threshold_slider:.1%}")
            with m_col3:
                st.metric("Latency", f"{elapsed*1000:.1f} ms")

            # Score bar
            sim_val = max(0.0, min(1.0, res.get("similarity", 0.0)))
            st.progress(sim_val, text=f"Cosine Similarity Match Confidence: {sim_val:.1%}")

            # Liveness diagnostics
            if res.get("liveness"):
                liv = res["liveness"]
                st.markdown("##### 🛡️ Liveness / Anti-Spoofing Report")
                liv_col1, liv_col2 = st.columns(2)
                with liv_col1:
                    st.write(f"**Status:** {'🟢 Live Face' if liv.get('is_live') else '🔴 Spoof / Inactive'}")
                    st.write(f"**Sharpness Score:** `{liv.get('sharpness_score', 0):.2f}`")
                with liv_col2:
                    st.write(f"**Laplacian Variance:** `{liv.get('laplacian_variance', 0)}`")
                    st.write(f"**High Freq Energy:** `{liv.get('high_freq_energy', 0)}`")

            # Score Breakdown table if 1:N
            if "all_scores" in res and res["all_scores"]:
                st.markdown("##### 🏆 Candidate Ranking")
                ranked = sorted(res["all_scores"].items(), key=lambda x: x[1]["similarity"], reverse=True)
                score_data = [
                    {"User ID": k, "Name": v["name"], "Similarity Score": f"{v['similarity']:.2%}", "Above Threshold": "Yes" if v['similarity'] >= threshold_slider else "No"}
                    for k, v in ranked[:5]
                ]
                st.dataframe(pd.DataFrame(score_data), use_container_width=True)
        else:
            st.info("Upload an image or take a snapshot and click **Run Biometric Matching** to see results.")

# ==========================================
# TAB 2: USER ENROLLMENT
# ==========================================
with tab_enroll:
    st.subheader("Enroll New Identity")
    st.write("Register a new user into the biometric feature database by capturing their face vector.")

    enroll_col1, enroll_col2 = st.columns([1, 1], gap="medium")

    with enroll_col1:
        new_uid = st.text_input("User ID (e.g. 'swara_01')", placeholder="e.g. user_101")
        new_name = st.text_input("Full Name", placeholder="e.g. Swara Kashyap")
        
        enroll_source = st.radio("Enrollment Image Source", ["Upload Photo", "Webcam Capture"], horizontal=True, key="enroll_src")
        enroll_bgr = None

        if enroll_source == "Upload Photo":
            up_enroll = st.file_uploader("Upload High-Quality Face Photo", type=["jpg", "jpeg", "png"], key="enroll_up")
            if up_enroll:
                file_bytes = np.asarray(bytearray(up_enroll.read()), dtype=np.uint8)
                enroll_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(enroll_bgr, cv2.COLOR_BGR2RGB), caption="Candidate Photo", use_container_width=True)
        else:
            cam_enroll = st.camera_input("Capture Enrollment Photo", key="enroll_cam")
            if cam_enroll:
                file_bytes = np.asarray(bytearray(cam_enroll.read()), dtype=np.uint8)
                enroll_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        submit_enroll = st.button("💾 Enroll & Save Face Embedding", type="primary", use_container_width=True, disabled=(enroll_bgr is None or not new_uid or not new_name))

    with enroll_col2:
        st.subheader("Enrollment Preview & Extraction")
        if submit_enroll and enroll_bgr is not None:
            with st.spinner("Detecting face and computing deep feature representation..."):
                success, msg, details = engine.enroll_user(
                    user_id=new_uid,
                    name=new_name,
                    image_bgr=enroll_bgr,
                    check_liveness=liveness_toggle
                )

            if success:
                st.success(msg)
                if details and "face_rgb" in details:
                    st.image(details["face_rgb"], caption=f"Extracted Face Crop ({config.FACE_INPUT_SIZE[0]}x{config.FACE_INPUT_SIZE[1]})", width=224)
                    st.json({
                        "User ID": details["user_id"],
                        "Name": details["name"],
                        "Embedding Dimension": details["embedding_dim"],
                        "Backbone": backbone_choice
                    })
                st.balloons()
            else:
                st.error(msg)
        else:
            st.info("Enter user credentials and provide a face image to extract biometric feature vectors.")

# ==========================================
# TAB 3: DATABASE GALLERY
# ==========================================
with tab_gallery:
    st.subheader("Biometric Database Gallery")
    st.write("Browse, inspect, and manage registered identities in the system.")

    users = engine.list_users()

    # Pre-populate sample dataset button if empty
    if len(users) == 0:
        st.info("No identities currently registered.")
        if st.button("📥 Load Benchmark Sample Identities"):
            with st.spinner("Enrolling sample benchmark identities..."):
                from generate_sample_data import generate_benchmark_dataset
                dataset = generate_benchmark_dataset(config.SAMPLE_DATA_DIR, num_identities=5)
                for uid, data in dataset.items():
                    img = cv2.imread(data["enrollment_image"])
                    engine.enroll_user(uid, data["name"], img, check_liveness=False)
                st.success("Successfully loaded benchmark sample identities!")
                st.rerun()

    else:
        # Display users in grid cards
        cols_per_row = 3
        for i in range(0, len(users), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(users):
                    u = users[idx]
                    with row_cols[j]:
                        with st.container():
                            st.markdown(f"**{u['name']}**")
                            st.caption(f"ID: `{u['user_id']}` | Registered: {u['created_at']}")
                            img_path = u.get("image_path")
                            if img_path and os.path.exists(img_path):
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            else:
                                st.write("*(No preview available)*")
                            
                            if st.button(f"🗑️ Delete User", key=f"del_{u['user_id']}"):
                                engine.delete_user(u["user_id"])
                                st.warning(f"Deleted user {u['name']}.")
                                st.rerun()

# ==========================================
# TAB 4: BENCHMARK & EVALUATION
# ==========================================
with tab_eval:
    st.subheader("Biometric Performance Benchmark & Analytics")
    st.markdown("Detailed quantitative evaluation across **Accuracy**, **FAR (False Acceptance Rate)**, **FRR (False Rejection Rate)**, and **EER (Equal Error Rate)** as required by PRD Section 6 (FR10) & Section 11.")

    eval_btn = st.button("📊 Run Full Benchmark & Generate Report", type="primary")
    
    benchmark_img_path = config.EVALUATION_DIR / "biometric_evaluation_benchmark.png"
    comparison_csv_path = config.EVALUATION_DIR / "architecture_comparison.csv"

    if eval_btn:
        with st.spinner("Generating pairs, extracting multi-backbone features, and computing ROC/EER curves..."):
            from evaluate import main as run_eval_script
            run_eval_script()
            st.success("Evaluation complete! Benchmark plots and comparison metrics updated.")

    if comparison_csv_path.exists():
        df_comp = pd.read_csv(comparison_csv_path)
        st.markdown("#### 1. CNN Architecture Comparison Table")
        st.dataframe(df_comp.style.format({
            "accuracy_at_eer": "{:.2%}",
            "far_at_eer": "{:.2%}",
            "frr_at_eer": "{:.2%}",
            "eer": "{:.2%}",
            "roc_auc": "{:.3f}",
            "optimal_threshold": "{:.2f}"
        }), use_container_width=True)

    if benchmark_img_path.exists():
        st.markdown("#### 2. Biometric Verification Performance Graphs")
        st.image(str(benchmark_img_path), caption="Score Distribution, FAR vs FRR Curve, ROC Curve, and Confusion Matrix", use_container_width=True)
        
        st.markdown("""
        > **Academic Metric Explanations**:
        > * **FAR (False Acceptance Rate)**: Probability that an impostor face is incorrectly accepted by the system.
        > * **FRR (False Rejection Rate)**: Probability that a genuine enrolled user is incorrectly rejected.
        > * **EER (Equal Error Rate)**: The operating threshold where $\\text{FAR} = \\text{FRR}$. Lower EER indicates superior discriminative power.
        > * **ROC-AUC (Area Under Curve)**: Overall classification metric where $1.0$ represents perfect separation.
        """)

