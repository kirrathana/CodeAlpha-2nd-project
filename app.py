"""
Premium AI Application for Speech Emotion Recognition.
Modern SaaS-style interface with glassmorphism UI and dynamic visualizations.
"""

import io
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from feature_extraction import (
    extract_features_from_audio,
    extract_features_from_file,
    get_n_features,
    load_audio,
    load_dataset,
    MAX_FRAMES,
    MODELS_DIR,
    N_MFCC,
    SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EmotionAI - Speech Emotion Recognition",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

EMOTION_ICONS = {
    "Neutral": "😐", "Happy": "😊", "Sad": "😢",
    "Angry": "😡", "Fearful": "😨", "Disgust": "🤢", "Surprised": "😲",
}

EMOTION_COLORS = {
    "Neutral": "#94a3b8", "Happy": "#fbbf24",
    "Sad": "#6366f1", "Angry": "#ef4444", "Fearful": "#a855f7",
    "Disgust": "#22c55e", "Surprised": "#f97316",
}

# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Glassmorphism Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    #MainMenu, footer, header {visibility: hidden;}
    footer:after {
        content: 'EmotionAI © 2025';
        visibility: visible; display: block; position: relative;
        padding: 5px; color: #64748b; font-size: 12px; text-align: center;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
        width: 220px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1rem;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 500;
        padding: 12px 16px;
        border-radius: 12px;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(99, 102, 241, 0.1);
        transform: translateX(4px);
    }

    section[data-testid="stSidebar"] .stRadio label[data-testid="stRadio-option"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 28px;
        margin: 12px 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }

    .kpi-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.1) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 28px 24px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(99, 102, 241, 0.2);
    }

    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 600;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e2e8f0 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 3rem;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.1) 0%, rgba(99,102,241,0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }

    .emotion-icon {
        font-size: 5rem;
        margin-bottom: 16px;
        animation: bounce 1s ease infinite;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    .emotion-name {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    .confidence-text {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-top: 12px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 16px 48px;
        font-weight: 700;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 32px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }

    .sidebar-logo {
        text-align: center;
        padding: 24px 0;
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .expandable-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ---------------------------------------------------------------------------
# Resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_resources():
    """Load trained model, scaler, and label encoder."""
    model_path = MODELS_DIR / "emotion_model.keras"
    scaler_path = MODELS_DIR / "scaler.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"

    resources = {"model": None, "scaler": None, "encoder": None, "metrics": {}}

    if model_path.exists():
        resources["model"] = tf.keras.models.load_model(str(model_path))
    if scaler_path.exists():
        resources["scaler"] = joblib.load(scaler_path)
    if encoder_path.exists():
        resources["encoder"] = joblib.load(encoder_path)

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            resources["metrics"] = json.load(f)

    return resources


@st.cache_data
def load_dataset_metadata():
    """Load dataset metadata and statistics."""
    csv_path = MODELS_DIR / "dataset_metadata.csv"
    stats_path = MODELS_DIR / "dataset_stats.json"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        try:
            df = load_dataset()
        except Exception:
            df = pd.DataFrame()

    stats = {}
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)

    return df, stats


@st.cache_data
def load_training_history():
    """Load model training history."""
    path = MODELS_DIR / "training_history.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def scale_features(features: np.ndarray, scaler) -> np.ndarray:
    """Apply StandardScaler to feature matrix."""
    n_frames, n_feat = features.shape
    flat = features.reshape(1, -1)
    scaled = scaler.transform(flat)
    return scaled.reshape(1, n_frames, n_feat)


def predict_emotion(audio_path: str, resources: dict) -> dict:
    """Run emotion prediction on an audio file."""
    start = time.time()

    features = extract_features_from_file(audio_path)
    if resources["scaler"] is not None:
        features = scale_features(features, resources["scaler"])

    probs = resources["model"].predict(features, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    emotion = resources["encoder"].classes_[pred_idx]
    confidence = float(probs[pred_idx])
    elapsed = time.time() - start

    prob_dict = {
        resources["encoder"].classes_[i]: float(probs[i])
        for i in range(len(probs))
    }

    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": prob_dict,
        "prediction_time": elapsed,
    }


def create_gauge_chart(confidence: float) -> go.Figure:
    """Create Plotly gauge chart for confidence meter."""
    pct = confidence * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 28, "color": "#f1f5f9"}},
        title={"text": "Emotion Confidence Meter", "font": {"size": 18, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar": {"color": "#6366f1"},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 2,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 25], "color": "rgba(239,68,68,0.3)"},
                {"range": [25, 50], "color": "rgba(249,115,22,0.3)"},
                {"range": [50, 75], "color": "rgba(251,191,36,0.3)"},
                {"range": [75, 100], "color": "rgba(34,197,94,0.3)"},
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 3},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        height=300,
        margin=dict(t=60, b=20, l=30, r=30),
    )
    return fig


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Create sorted probability bar chart."""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions = [e for e, _ in sorted_probs]
    values = [p * 100 for _, p in sorted_probs]
    colors = [EMOTION_COLORS.get(e, "#6366f1") for e in emotions]

    fig = go.Figure(go.Bar(
        x=values, y=emotions, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        textfont=dict(color="#f1f5f9"),
    ))
    fig.update_layout(
        title=dict(text="Emotion Probability Distribution", font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=400,
        margin=dict(t=50, b=40, l=100, r=40),
    )
    return fig


def plot_waveform(audio_path: str) -> go.Figure:
    """Generate interactive waveform plot."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    times = np.arange(len(y)) / sr
    fig = go.Figure(go.Scatter(
        x=times, y=y, mode="lines",
        line=dict(color="#818cf8", width=1),
        fill="tozeroy",
        fillcolor="rgba(129, 140, 248, 0.15)",
    ))
    fig.update_layout(
        title="Waveform",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Amplitude", gridcolor="rgba(255,255,255,0.05)"),
        height=280,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def plot_spectrogram(audio_path: str) -> go.Figure:
    """Generate interactive mel spectrogram."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig = go.Figure(go.Heatmap(
        z=mel_db, colorscale="Viridis", showscale=True,
        colorbar=dict(title="dB"),
    ))
    fig.update_layout(
        title="Mel Spectrogram",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(title="Frames"),
        yaxis=dict(title="Mel Bands"),
        height=280,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def plot_mel_spectrogram(audio_path: str) -> go.Figure:
    """Generate mel spectrogram with different visualization."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig = go.Figure(go.Heatmap(
        z=mel_db, colorscale="Plasma", showscale=True,
        colorbar=dict(title="dB"),
    ))
    fig.update_layout(
        title="Mel Spectrogram (Plasma)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(title="Frames"),
        yaxis=dict(title="Mel Bands"),
        height=280,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def plot_chroma_features(audio_path: str) -> go.Figure:
    """Generate chroma features visualization."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    fig = go.Figure(go.Heatmap(
        z=chroma, colorscale="Cividis", showscale=True,
        colorbar=dict(title="Intensity"),
    ))
    fig.update_layout(
        title="Chroma Features",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(title="Frames"),
        yaxis=dict(title="Chroma Bins"),
        height=280,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def plot_mfcc_heatmap(audio_path: str) -> go.Figure:
    """Generate MFCC heatmap."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    fig = go.Figure(go.Heatmap(
        z=mfcc, colorscale="RdBu_r", showscale=True,
        colorbar=dict(title="Coeff"),
    ))
    fig.update_layout(
        title="MFCC Heatmap",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(title="Frames"),
        yaxis=dict(title="MFCC Coefficients"),
        height=280,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def create_radar_chart(probabilities: dict) -> go.Figure:
    """Create radar chart comparing all emotion probabilities."""
    emotions = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [EMOTION_COLORS.get(e, "#6366f1") for e in emotions]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        line=dict(color="#818cf8", width=2),
        marker=dict(color=colors, size=8),
    ))
    fig.update_layout(
        title=dict(text="Emotion Radar Chart", font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
    )
    return fig


def create_pie_chart(probabilities: dict, predicted_emotion: str) -> go.Figure:
    """Create pie chart showing predicted emotion vs remaining."""
    pred_value = probabilities[predicted_emotion]
    remaining = 1 - pred_value

    fig = go.Figure(data=[go.Pie(
        labels=[predicted_emotion, "Other Emotions"],
        values=[pred_value, remaining],
        hole=0.4,
        marker=dict(colors=[EMOTION_COLORS.get(predicted_emotion, "#6366f1"), "#334155"]),
    )])
    fig.update_layout(
        title=dict(text="Emotion Distribution", font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        height=400,
        margin=dict(t=50, b=40, l=40, r=40),
        showlegend=True,
    )
    return fig


def create_confidence_progress(probabilities: dict, predicted_emotion: str) -> go.Figure:
    """Create animated confidence progress bar chart."""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions = [e for e, _ in sorted_probs]
    values = [p * 100 for _, p in sorted_probs]
    colors = [EMOTION_COLORS.get(e, "#6366f1") for e in emotions]

    fig = go.Figure(go.Bar(
        x=emotions,
        y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#f1f5f9", size=12),
    ))
    fig.update_layout(
        title=dict(text="Confidence Progress", font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(title="Emotion", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Confidence (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
        height=350,
        margin=dict(t=50, b=40, l=60, r=40),
    )
    return fig


def render_kpi_cards(metrics: dict, dataset_stats: dict, prediction_count: int):
    """Render hero KPI metric cards - 4 cards only."""
    cols = st.columns(4)
    kpis = [
        ("Model Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%"),
        ("Supported Emotions", str(len(EMOTION_ICONS))),
        ("Dataset Size", str(dataset_stats.get("total_files", "—"))),
        ("Prediction Count", str(prediction_count)),
    ]
    for col, (label, value) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <p class="kpi-value">{value}</p>
                <p class="kpi-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------
def page_home(resources, df, stats):
    """Render home page with hero section and KPI cards."""
    metrics = resources.get("metrics", {})
    prediction_count = len(st.session_state.prediction_history)

    st.markdown('<p class="hero-title">🎙 Speech Emotion Recognition System</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-powered human emotion analysis from speech.</p>', unsafe_allow_html=True)

    render_kpi_cards(metrics, stats, prediction_count)

    st.markdown('<p class="section-header">System Overview</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#818cf8;margin-top:0;">🧠 Deep Learning Pipeline</h4>
            <p style="color:#94a3b8;line-height:1.7;">
                Hybrid <strong>CNN + LSTM</strong> architecture processes temporal audio features
                including MFCC, Chroma, Mel Spectrogram, Spectral Contrast, Tonnetz, ZCR, and RMS Energy
                to classify speech into 7 distinct emotional states.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#c084fc;margin-top:0;">🎯 Real-Time Analysis</h4>
            <p style="color:#94a3b8;line-height:1.7;">
                Upload any WAV audio file to instantly detect emotions with high accuracy.
                The system provides detailed confidence scores and visualizations
                for comprehensive emotion analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)


def page_prediction(resources):
    """Render emotion prediction page."""
    st.markdown('<p class="hero-title">� Predict Emotion</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Upload a WAV file to analyze emotional content</p>', unsafe_allow_html=True)

    if resources["model"] is None:
        st.warning("⚠️ Model not found. Please run `python train_model.py` first.")
        return

    uploaded = st.file_uploader("Upload WAV Audio File", type=["wav"], help="Supported format: .wav")

    if uploaded:
        audio_path = BASE_DIR / "audio_files" / uploaded.name
        audio_path.parent.mkdir(exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.audio(uploaded)

        st.plotly_chart(plot_waveform(str(audio_path)), use_container_width=True)

        if st.button("� Analyze Emotion", use_container_width=True):
            with st.spinner("Analyzing emotional patterns..."):
                result = predict_emotion(str(audio_path), resources)

            icon = EMOTION_ICONS.get(result["emotion"], "🎭")
            conf_pct = result["confidence"] * 100
            emotion_color = EMOTION_COLORS.get(result["emotion"], "#6366f1")

            st.markdown(f"""
            <div class="result-card" style="border-color: {emotion_color}; background: linear-gradient(135deg, {emotion_color}20 0%, {emotion_color}10 100%);">
                <div class="emotion-icon">{icon}</div>
                <div class="emotion-name" style="color: {emotion_color};">{result['emotion']}</div>
                <div class="confidence-text">
                    Confidence: {conf_pct:.1f}% &nbsp;|&nbsp;
                    Prediction Time: {result['prediction_time']*1000:.0f}ms
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-header">Dynamic Visualization</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_gauge_chart(result["confidence"]), use_container_width=True)
            with col2:
                st.plotly_chart(create_probability_chart(result["probabilities"]), use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(create_radar_chart(result["probabilities"]), use_container_width=True)
            with col4:
                st.plotly_chart(create_pie_chart(result["probabilities"], result["emotion"]), use_container_width=True)

            st.plotly_chart(create_confidence_progress(result["probabilities"], result["emotion"]), use_container_width=True)

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded.name,
                "emotion": result["emotion"],
                "confidence": f"{conf_pct:.1f}%",
                "time_ms": f"{result['prediction_time']*1000:.0f}",
            }
            st.session_state.prediction_history.append(record)


def page_analytics(resources):
    """Render analytics page with uploaded audio visualizations only."""
    st.markdown('<p class="hero-title">📊 Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Audio feature visualizations for uploaded files</p>', unsafe_allow_html=True)

    if resources["model"] is None:
        st.warning("⚠️ Model not found. Please run `python train_model.py` first.")
        return

    uploaded = st.file_uploader("Upload WAV Audio File for Analysis", type=["wav"], key="analytics_upload")

    if uploaded:
        audio_path = BASE_DIR / "audio_files" / f"analytics_{uploaded.name}"
        audio_path.parent.mkdir(exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.audio(uploaded)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_waveform(str(audio_path)), use_container_width=True)
        with col2:
            st.plotly_chart(plot_spectrogram(str(audio_path)), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_mfcc_heatmap(str(audio_path)), use_container_width=True)
        with col4:
            st.plotly_chart(plot_mel_spectrogram(str(audio_path)), use_container_width=True)

        st.plotly_chart(plot_chroma_features(str(audio_path)), use_container_width=True)
    else:
        st.info("👆 Upload a WAV file to see audio feature visualizations")


def page_insights(resources):
    """Render insights page with expandable cards for model metrics."""
    st.markdown('<p class="hero-title">📈 Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Model performance metrics and training analysis</p>', unsafe_allow_html=True)

    metrics = resources.get("metrics", {})
    history = load_training_history()

    with st.expander("📊 Model Metrics", expanded=True):
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [
            ("Accuracy", "accuracy"), ("Precision", "precision"),
            ("Recall", "recall"), ("F1 Score", "f1_score"),
        ]):
            with col:
                val = metrics.get(key, 0) * 100
                st.metric(label, f"{val:.1f}%")

    with st.expander("🧠 Model Architecture"):
        arch_path = MODELS_DIR / "model_architecture.txt"
        if arch_path.exists():
            with open(arch_path, encoding="utf-8") as f:
                st.code(f.read(), language="text")
        elif resources["model"]:
            buf = io.StringIO()
            resources["model"].summary(print_fn=lambda x: buf.write(x + "\n"))
            st.code(buf.getvalue(), language="text")

    if history:
        with st.expander("📈 Training Accuracy Curve"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.get("accuracy", []), name="Train", line=dict(color="#818cf8")))
            fig.add_trace(go.Scatter(y=history.get("val_accuracy", []), name="Validation", line=dict(color="#c084fc")))
            fig.update_layout(
                title="Accuracy vs Epoch", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font={"color": "#94a3b8"},
                xaxis=dict(title="Epoch"), yaxis=dict(title="Accuracy"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📉 Loss Curve"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.get("loss", []), name="Train", line=dict(color="#ef4444")))
            fig.add_trace(go.Scatter(y=history.get("val_loss", []), name="Validation", line=dict(color="#f97316")))
            fig.update_layout(
                title="Loss vs Epoch", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font={"color": "#94a3b8"},
                xaxis=dict(title="Epoch"), yaxis=dict(title="Loss"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🎯 Confusion Matrix"):
        path = MODELS_DIR / "confusion_matrix.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("Confusion matrix image not found. Run training to generate.")

    with st.expander("📊 ROC Curve"):
        path = MODELS_DIR / "roc_curve.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("ROC curve image not found. Run training to generate.")

    with st.expander("🔍 Feature Importance"):
        path = MODELS_DIR / "feature_importance.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("Feature importance image not found. Run training to generate.")




def page_about():
    """Render about page."""
    st.markdown('<p class="hero-title">ℹ About</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Problem Statement</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="color:#94a3b8;line-height:1.8;">
            Understanding human emotions from speech is a critical challenge in affective computing.
            Emotion recognition enables applications in healthcare, customer service, education,
            and human-computer interaction. This system addresses the challenge of accurately
            classifying emotional states from raw audio signals using deep learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Technologies Used</p>', unsafe_allow_html=True)
    tech_cols = st.columns(4)
    technologies = [
        ("TensorFlow", "Deep Learning"), ("Librosa", "Audio Processing"),
        ("CNN + LSTM", "Neural Networks"), ("Streamlit", "Web Dashboard"),
    ]
    for i, (name, desc) in enumerate(technologies):
        with tech_cols[i % 4]:
            st.markdown(f"""
            <div class="kpi-card" style="padding:16px;">
                <p class="kpi-value" style="font-size:1.2rem;">{name}</p>
                <p class="kpi-label">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Future Scope</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <ul style="color:#94a3b8;line-height:2;">
            <li>🎤 <strong>Real-time Emotion Detection</strong> from live microphone input</li>
            <li>🌍 <strong>Multilingual Support</strong> for non-English speech analysis</li>
            <li>🤖 <strong>Voice Assistant Integration</strong> with Alexa, Google Assistant</li>
            <li>🧬 <strong>Transformer Models</strong> using Wav2Vec 2.0 and HuBERT</li>
            <li>📱 <strong>Mobile Deployment</strong> via TensorFlow Lite</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Author Information</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#818cf8;margin-top:0;">Sivaraj</h4>
        <p style="color:#94a3b8;">AI/ML Engineer</p>
        <p style="color:#64748b;margin-top:12px;">
            Specialized in deep learning, computer vision, and natural language processing.
            Passionate about building AI solutions that solve real-world problems.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    """Main application entry point."""
    st.sidebar.markdown('<div class="sidebar-logo">🎙 EmotionAI</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "� Predict Emotion",
            "📊 Analytics",
            "📈 Insights",
            "ℹ About",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="color:#64748b;font-size:0.75rem;text-align:center;">'
        "EmotionAI v1.0</p>",
        unsafe_allow_html=True,
    )

    resources = load_model_resources()
    df, stats = load_dataset_metadata()

    if page == "🏠 Home":
        page_home(resources, df, stats)
    elif page == "� Predict Emotion":
        page_prediction(resources)
    elif page == "📊 Analytics":
        page_analytics(resources)
    elif page == "📈 Insights":
        page_insights(resources)
    elif page == "ℹ About":
        page_about()


if __name__ == "__main__":
    main()
