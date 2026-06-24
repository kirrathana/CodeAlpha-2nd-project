# 🎙 Speech Emotion Recognition System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI-Powered Emotion Detection from Human Speech** — A production-ready deep learning system that classifies emotions from audio using a Hybrid CNN+LSTM architecture and a premium analytics dashboard.

---

## 📋 Project Overview

This project implements an end-to-end **Speech Emotion Recognition (SER)** pipeline that analyzes human speech audio and predicts one of **8 emotional states**. Built with modern deep learning techniques and deployed through an enterprise-grade Streamlit dashboard, it is designed for portfolio, resume, and production demonstrations.

### 🎯 Objective

Develop a robust machine learning system capable of:

- Automatically processing the RAVDESS speech emotion dataset
- Extracting rich audio features (MFCC, Chroma, Mel Spectrogram, etc.)
- Training a Hybrid **CNN + LSTM** neural network
- Delivering real-time emotion predictions via an interactive web dashboard

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **8 Emotion Classes** | Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised |
| **Hybrid CNN+LSTM** | Deep learning architecture for temporal audio pattern recognition |
| **Data Augmentation** | Noise injection, pitch shift, time stretch, random shift |
| **Premium Dashboard** | Dark-themed glassmorphism UI with Plotly visualizations |
| **Real-time Prediction** | Upload `.wav` files and get instant emotion analysis |
| **Model Analytics** | Confusion matrix, ROC curves, training history |
| **Prediction History** | Session-based tracking with CSV export |

---

## 📊 Dataset Information

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

| Property | Value |
|----------|-------|
| Actors | 24 (12 Male, 12 Female) |
| Total Files | 1,440 audio samples |
| Format | `.wav` (16-bit, mono) |
| Emotions | 8 classes |
| Filename Format | `modality-vocal-emotion-intensity-statement-repetition-actor.wav` |

Place the dataset in:

```
dataset/Audio_Speech_Actors_01-24/
├── Actor_01/
├── Actor_02/
...
└── Actor_24/
```

---

## 🧠 Model Architecture

```
Input (128 × 107)
    │
    ▼
Conv1D (64 filters) → BatchNorm → MaxPool → Dropout
    │
    ▼
Conv1D (128 filters) → BatchNorm → MaxPool → Dropout
    │
    ▼
LSTM (128 units)
    │
    ▼
Dense (64) → Dropout → Dense (8) → Softmax
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 50 |
| Batch Size | 32 |
| Train/Test Split | 80/20 |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## 🛠 Technologies Used

- **TensorFlow / Keras** — Deep learning framework
- **Librosa** — Audio feature extraction
- **NumPy / Pandas** — Data processing
- **Scikit-learn** — Preprocessing & metrics
- **Plotly** — Interactive visualizations
- **Streamlit** — Web dashboard
- **Matplotlib / Seaborn** — Static plots & EDA

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd Emotion_Recognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training

### Step 1: Extract Features

```bash
python feature_extraction.py
```

This will:
- Parse all RAVDESS audio files
- Apply data augmentation
- Extract and normalize features
- Save artifacts to `models/`

### Step 2: Train Model

```bash
python train_model.py
```

This will:
- Split data (80/20)
- Train the CNN+LSTM model
- Generate evaluation metrics and plots
- Save `models/emotion_model.keras`

---

## ▶️ Run Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Folder Structure

```
Emotion_Recognition/
│
├── dataset/
│   └── Audio_Speech_Actors_01-24/    # RAVDESS dataset
│
├── audio_files/                       # Uploaded audio storage
│
├── notebooks/
│   └── EDA.ipynb                      # Exploratory Data Analysis
│
├── models/                            # Saved models & artifacts
│   ├── emotion_model.keras
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   ├── training_history.json
│   └── *.png                          # Evaluation plots
│
├── feature_extraction.py              # Feature extraction pipeline
├── train_model.py                     # Model training & evaluation
├── app.py                             # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 📸 Screenshots

> Add screenshots of your dashboard here after running the app:
>
> - Dashboard home with KPI cards
> - Emotion prediction page with gauge chart
> - Analytics visualizations
> - Model insights page

---

## 🔮 Future Improvements

- [ ] Real-time emotion detection from microphone input
- [ ] Multilingual support (non-English speech)
- [ ] Voice assistant integration (Alexa, Google Assistant)
- [ ] Transformer-based architecture (Wav2Vec 2.0)
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Multi-modal fusion (audio + text)

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

The RAVDESS dataset is subject to its own [Creative Commons license](https://zenodo.org/record/1188976).

---

## 👤 Author

**Sivaraj**

- GitHub: [github.com/yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

<p align="center">
  Built with ❤️ using TensorFlow, Librosa & Streamlit
</p>
