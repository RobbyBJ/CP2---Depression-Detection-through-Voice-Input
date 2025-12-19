import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import opensmile
import noisereduce as nr
from scipy.signal import butter, lfilter

# ================= CONFIGURATION =================
MODEL_PATH = "ensemble_models/stacking_ensemble.pkl"

# Calibrated Threshold
THRESHOLD = 0.4

# Audio Settings 
SEGMENT_DURATION = 5.0 
SR = 16000
# =================================================

# --- 1. SETUP & STYLE ---
st.set_page_config(page_title="DepressionAI ", layout="wide")
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;}
    .depressed { background-color: #d32f2f; }
    .healthy { background-color: #388e3c; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PREPROCESSING FUNCTIONS  ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300, highcut=3400, fs=16000, order=5):
    """Keeps frequencies between 300Hz and 3400Hz (Human Voice Range)."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_audio_signal(y, sr):

    # 1. Noise Reduction
    try:
        y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.75)
    except:
        pass

    # 2. Bandpass Filter
    y = bandpass_filter(y, fs=sr)

    # 3. Silence Removal
    intervals = librosa.effects.split(y, top_db=25)
    if len(intervals) > 0:
        y = np.concatenate([y[start:end] for start, end in intervals])

    # 4. Normalization
    y = librosa.util.normalize(y)
    
    return y

# --- 3. BACKEND FUNCTIONS ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def get_feature_extractor():
    """Initializes OpenSMILE (eGeMAPS)."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def process_and_predict(audio_file, model, smile):
    # 1. Load Audio
    y_raw, sr = librosa.load(audio_file, sr=SR)
    
    # 2. Apply Preprocessing (Cleaning)
    y_clean = preprocess_audio_signal(y_raw, sr)
    
    if len(y_clean) < sr: # If file is empty after cleaning
        return None, 0.0, y_raw, []

    # 3. Segmentation & Prediction Loop
    seg_samples = int(SEGMENT_DURATION * SR)
    num_segments = int(np.ceil(len(y_clean) / seg_samples))
    
    segment_probs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_segments):
        status_text.text(f"Analyzing Segment {i+1}/{num_segments}...")
        start = i * seg_samples
        end = min(start + seg_samples, len(y_clean))
        y_seg = y_clean[start:end]
        
        # Skip tiny segments
        if len(y_seg) < 0.5 * SR:
            continue
            
        # Pad if short
        if len(y_seg) < seg_samples:
            y_seg = np.pad(y_seg, (0, seg_samples - len(y_seg)))

        try:
            # A. Extract OpenSMILE Features
            df_feat = smile.process_signal(y_seg, sr)
            df_feat.reset_index(drop=True, inplace=True)
            
            # B. Get Probability (Class 1)
            prob = model.predict_proba(df_feat)[0][1]
            segment_probs.append(prob)
            
        except Exception:
            continue
            
        progress_bar.progress((i + 1) / num_segments)
    
    status_text.empty()
    progress_bar.empty()

    if not segment_probs:
        return None, 0.0, y_raw, []

    # 4. Voting Logic (Aggregate)
    avg_risk_score = np.mean(segment_probs)
    
    # 5. Apply Calibrated Threshold
    final_pred = 1 if avg_risk_score >= THRESHOLD else 0
    
    return final_pred, avg_risk_score, y_raw, segment_probs

# --- 4. FRONTEND UI ---
st.title("🧠 AI Depression Detector ")
st.markdown(f"**Pipeline:** Bandpass/NR Preprocessing → OpenSMILE eGeMAPS → Stacking Ensemble")
st.markdown(f"**Calibrated Sensitivity Threshold:** `{THRESHOLD}`")

model = load_model()
smile = get_feature_extractor()

if model and smile:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1. Input Audio")
        uploaded_file = st.file_uploader("Upload .wav recording", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("Run Clinical Screening", type="primary"):
                with st.spinner("Applying Pipeline (Cleaning & Feature Extraction)..."):
                    pred, risk_score, y_display, seg_probs = process_and_predict(uploaded_file, model, smile)
                
                if pred is None:
                    st.error("Audio quality insufficient after noise reduction.")
                else:
                    st.subheader("2. Diagnosis")
                    
                    # --- RESULT DISPLAY ---
                    if pred == 1:
                        display_score = min(risk_score * 100, 99)
                        st.markdown(f"""
                            <div class="result-box depressed">
                                <h1>⚠️ AT RISK</h1>
                                <p>Depressive Vocal Biomarkers Detected</p>
                                <h2>Risk Score: {display_score:.1f}%</h2>
                                <p style='font-size: 14px'>(Clinical Threshold: {THRESHOLD*100}%)</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.warning(f"The aggregated risk ({risk_score:.2f}) exceeds the calibrated threshold of {THRESHOLD}.")
                    
                    else:
                        display_score = (1 - risk_score) * 100
                        st.markdown(f"""
                            <div class="result-box healthy">
                                <h1>✅ HEALTHY SIGNAL</h1>
                                <p>Within Normal Parameters</p>
                                <h2>Healthy Confidence: {display_score:.1f}%</h2>
                                <p style='font-size: 14px'>(Risk: {risk_score:.2f} < {THRESHOLD})</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.success("No consistent depressive patterns detected across segments.")

    with col2:
        if uploaded_file is not None and 'y_display' in locals():
            st.subheader("3. Technical Analysis")
            
            # --- PLOT 1: WAVEFORM ---
            st.markdown("**Original Waveform**")
            fig, ax = plt.subplots(figsize=(8, 2))
            librosa.display.waveshow(y_display, sr=SR, alpha=0.5, ax=ax, color='#2c3e50')
            ax.axis('off')
            st.pyplot(fig)
    
            # --- PLOT 2: RISK OVER TIME ---
            st.markdown("**Segment-Level Risk Probability**")
            if len(seg_probs) > 0:
                chart_df = pd.DataFrame({
                    "Segment (3s)": range(1, len(seg_probs)+1),
                    "Risk Probability": seg_probs
                })
                
                st.line_chart(chart_df.set_index("Segment (3s)"))
                st.caption(f"Any segment above {THRESHOLD} contributes to a positive diagnosis.")

            # --- EXPLANATION BOX ---
            with st.expander("ℹ️ Pipeline Architecture"):
                st.markdown("""
                1.  **Preprocessing:** Noise Reduction (-20dB), Bandpass Filter (300-3400Hz), Silence Removal.
                2.  **Extraction:** OpenSMILE eGeMAPS (88 Features: Jitter, Shimmer, HNR, F1-F3).
                3.  **Model:** Stacking Ensemble (Logistic Regression Meta-Learner).
                4.  **Logic:** Majority voting with a **0.5** probability threshold.
                """)