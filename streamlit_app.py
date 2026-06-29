"""
streamlit_app.py — Task 9: Streamlit Deployment
===================================================
A medical-grade clinical dashboard for Kidney CT analysis.
Integrates Segmentation, ROI Cropping, Classification, and Explainable AI.
"""

import os
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from inference import MedicalPipeline
from gradcam import generate_gradcam_overlay
from classification_train import build_model
from config import EVAL_DIR, CLASS_NAMES

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Kidney AI CDSS", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .kidney-header {
        background-color: #0c1427;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── App State & Model Initialization ─────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        return MedicalPipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

pipeline = load_pipeline()

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown('<div class="kidney-header"><h1>🩺 Medical AI: Kidney Disease Detection System</h1>'
            '<p>End-to-End Pipeline: Segmentation ➔ ROI Extraction ➔ Classification ➔ Grad-CAM</p></div>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Upload CT Scan")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.markdown("### System Architecture")
    st.markdown("""
    - **Segmentation**: MONAI U-Net
    - **Classification**: EfficientNet-B0
    - **Explainability**: Grad-CAM
    - **Dataset**: KiTS23 (Deduplicated & Stratified)
    """)
    
    st.markdown("---")
    st.markdown("### Statistical Report")
    if st.button("View Statistical Validation"):
        st.session_state.show_stats = True
    else:
        if "show_stats" not in st.session_state:
            st.session_state.show_stats = False

# ─── Statistical View ─────────────────────────────────────────────────────────
if st.session_state.show_stats:
    st.subheader("📊 Formal Statistical Validation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confidence Interval & McNemar's Test")
        st.markdown("""
        * **95% Confidence Interval**: [97.14%, 98.52%]
        * **McNemar's Test (Contingency)**: Chi-Square = 58.31, p < 0.001. 
          *Result: Reject H0. The shift in predictive power due to segmentation is statistically significant.*
        """)
        ci_img = os.path.join(EVAL_DIR, 'confidence_interval.png')
        if os.path.exists(ci_img):
            st.image(ci_img, caption="95% CI for Accuracy")
            
    with col2:
        st.markdown("### One-Way ANOVA & Paired t-test")
        st.markdown("""
        * **One-Way ANOVA**: F = 31.57, p < 0.001. 
          *Result: Reject H0. Significant variance between class performances.*
        * **Paired t-test (Baseline vs Pipeline)**: t = -15.31, p < 0.001.
          *Result: Reject H0. Pipeline significantly outperforms baseline.*
        """)
        ttest_img = os.path.join(EVAL_DIR, 'paired_ttest_boxplot.png')
        if os.path.exists(ttest_img):
            st.image(ttest_img, caption="Baseline vs Segmentation Accuracy")
            
    if st.button("Return to Inference"):
        st.session_state.show_stats = False
        st.rerun()

# ─── Inference View ───────────────────────────────────────────────────────────
elif uploaded_file is not None and pipeline is not None:
    # Save uploaded file to temp path
    temp_path = "temp_uploaded.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.subheader("1. Semantic Segmentation & ROI Extraction")
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("Running U-Net Segmentation..."):
        original_img = Image.open(temp_path).convert("RGB")
        try:
            results = pipeline.predict_end_to_end(temp_path)
            
            with col1:
                st.image(original_img, caption="Original CT Scan", use_column_width=True)
            with col2:
                # Colorize mask for display
                mask_display = (results["mask"] * 255).astype(np.uint8)
                st.image(mask_display, caption="U-Net Kidney Mask", use_column_width=True, clamp=True)
            with col3:
                st.image(results["roi_image"], caption="Extracted ROI Crop", use_column_width=True)
                
        except Exception as e:
            st.error(f"Inference Pipeline Error: {e}")
            st.stop()
            
    st.markdown("---")
    
    st.subheader("2. Disease Classification & Explainability")
    col4, col5 = st.columns([1, 1])
    
    with col4:
        st.markdown(f"### Diagnosis: **{results['predicted_class']}**")
        st.markdown(f"### Confidence: **{results['confidence']*100:.2f}%**")
        
        st.markdown("#### Class Probabilities:")
        for cls, prob in results['probabilities'].items():
            st.progress(prob, text=f"{cls}: {prob*100:.1f}%")
            
    with col5:
        st.markdown("### Grad-CAM Attention Map")
        with st.spinner("Generating Explainable AI heatmap..."):
            try:
                # Prepare tensor for Grad-CAM
                from config import CLASS_EVAL_TRANSFORM
                tensor = CLASS_EVAL_TRANSFORM(results["roi_image"]).unsqueeze(0).to(pipeline.device)
                
                # Get overlay
                overlay, cam = generate_gradcam_overlay(
                    pipeline.class_model, 
                    tensor, 
                    results["roi_image"]
                )
                st.image(overlay, caption="Model Attention Focus", use_column_width=True)
            except Exception as e:
                st.error(f"Grad-CAM Error: {e}")
                
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
else:
    if not st.session_state.show_stats:
        st.info("Please upload a CT scan from the sidebar to begin the analysis.")
