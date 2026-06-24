"""
Feature extraction module for Speech Emotion Recognition.

Extracts audio features from RAVDESS dataset, applies augmentation,
and saves processed arrays for model training.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "Audio_Speech_Actors_01-24"
MODELS_DIR = BASE_DIR / "models"
AUDIO_FILES_DIR = BASE_DIR / "audio_files"

SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 40
N_MELS = 40
N_CHROMA = 12
N_CONTRAST = 7
N_TONNETZ = 6
HOP_LENGTH = 512
N_FFT = 2048
MAX_FRAMES = 128

EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse RAVDESS filename to extract metadata.

    Format: modality-vocal-emotion-intensity-statement-repetition-actor.wav
    """
    try:
        parts = Path(filename).stem.split("-")
        if len(parts) < 7:
            raise ValueError(f"Invalid filename format: {filename}")

        actor_num = int(parts[6])
        gender = "Male" if actor_num % 2 == 1 else "Female"

        return {
            "filename": filename,
            "emotion_code": parts[2],
            "emotion": EMOTION_MAP.get(parts[2], "Unknown"),
            "actor": f"Actor_{parts[6]}",
            "gender": gender,
            "filepath": "",
        }
    except (IndexError, ValueError) as exc:
        logger.error("Failed to parse filename %s: %s", filename, exc)
        raise


def load_dataset(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Traverse Actor_01 to Actor_24 and build a dataset dataframe.
    """
    data_dir = data_dir or DATASET_DIR
    records: List[Dict[str, str]] = []

    actor_dirs = sorted(glob.glob(str(data_dir / "Actor_*")))
    if not actor_dirs:
        raise FileNotFoundError(f"No Actor folders found in {data_dir}")

    for actor_dir in actor_dirs:
        wav_files = glob.glob(os.path.join(actor_dir, "*.wav"))
        for wav_path in wav_files:
            try:
                meta = parse_filename(os.path.basename(wav_path))
                meta["filepath"] = wav_path
                records.append(meta)
            except ValueError:
                continue

    df = pd.DataFrame(records)
    logger.info("Loaded %d audio files from %d actors", len(df), len(actor_dirs))
    return df


def display_dataset_stats(df: pd.DataFrame) -> None:
    """Print dataset statistics to console."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total files       : {len(df)}")
    print(f"Missing values    : {df.isnull().sum().sum()}")
    print(f"Duplicate rows    : {df.duplicated(subset=['filename']).sum()}")
    print("\nClass Distribution:")
    print(df["emotion"].value_counts().to_string())
    print("\nGender Distribution:")
    print(df["gender"].value_counts().to_string())
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Audio loading & augmentation
# ---------------------------------------------------------------------------
def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load and normalize audio to fixed duration."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=DURATION)
        target_len = int(sr * DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        return audio
    except Exception as exc:
        logger.error("Error loading audio %s: %s", file_path, exc)
        raise


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Inject random Gaussian noise into audio signal."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def pitch_shift(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply random pitch shifting."""
    n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray) -> np.ndarray:
    """Apply random time stretching."""
    rate = np.random.uniform(0.8, 1.2)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    target_len = len(audio)
    if len(stretched) < target_len:
        stretched = np.pad(stretched, (0, target_len - len(stretched)))
    return stretched[:target_len]


def random_shift(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply random time shift."""
    shift = np.random.randint(-sr // 10, sr // 10)
    return np.roll(audio, shift)


def augment_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    augment: bool = True,
) -> List[np.ndarray]:
    """
    Return original audio plus augmented variants.
    """
    if not augment:
        return [audio]

    augmented = [audio]
    try:
        augmented.append(add_noise(audio))
        augmented.append(pitch_shift(audio, sr))
        augmented.append(time_stretch(audio))
        augmented.append(random_shift(audio, sr))
    except Exception as exc:
        logger.warning("Augmentation failed, using original only: %s", exc)
    return augmented


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_frame_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract per-frame features and concatenate into a 2D matrix.

    Returns:
        np.ndarray: Shape (MAX_FRAMES, n_features)
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_chroma=N_CHROMA
    )
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=N_CONTRAST - 1
    )
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)

    # Align tonnetz to same number of frames
    min_frames = min(
        mfcc.shape[1], chroma.shape[1], mel_db.shape[1],
        contrast.shape[1], tonnetz.shape[1], zcr.shape[1], rms.shape[1],
    )
    mfcc = mfcc[:, :min_frames]
    chroma = chroma[:, :min_frames]
    mel_db = mel_db[:, :min_frames]
    contrast = contrast[:, :min_frames]
    tonnetz = tonnetz[:, :min_frames]
    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]

    features = np.vstack([mfcc, chroma, mel_db, contrast, tonnetz, zcr, rms])
    features = features.T  # (time, n_features)

    # Pad or truncate to MAX_FRAMES
    if features.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
    else:
        features = features[:MAX_FRAMES, :]

    return features.astype(np.float32)


def extract_features_from_file(
    file_path: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Extract feature matrix from a single audio file."""
    audio = load_audio(file_path, sr)
    return extract_frame_features(audio, sr)


def extract_features_from_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Extract feature matrix from raw audio array."""
    target_len = int(sr * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return extract_frame_features(audio, sr)


def get_n_features() -> int:
    """Return total number of features per frame."""
    return N_MFCC + N_CHROMA + N_MELS + N_CONTRAST + N_TONNETZ + 1 + 1


def build_feature_dataset(
    df: pd.DataFrame,
    augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Extract features from all files in dataframe with optional augmentation.
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["emotion"].values)

    feature_list: List[np.ndarray] = []
    label_list: List[int] = []

    total = len(df)
    for count, (idx, row) in enumerate(df.iterrows(), start=1):
        try:
            audio = load_audio(row["filepath"])
            audios = augment_audio(audio, augment=augment)
            for aug_audio in audios:
                features = extract_frame_features(aug_audio)
                feature_list.append(features)
                label_list.append(labels[idx])
            if count % 50 == 0 or count == total:
                logger.info("Processed %d / %d files (%.0f%%)", count, total, 100 * count / total)
        except Exception as exc:
            logger.warning("Skipping %s: %s", row["filepath"], exc)

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int32)

    logger.info("Feature matrix shape: %s, Labels shape: %s", X.shape, y.shape)
    return X, y, label_encoder


def normalize_features(
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler on flattened feature dimensions.
    """
    n_samples, n_frames, n_features = X.shape
    X_flat = X.reshape(n_samples, -1)

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        X_scaled = scaler.transform(X_flat)

    X_scaled = X_scaled.reshape(n_samples, n_frames, n_features)
    return X_scaled.astype(np.float32), scaler


def save_artifacts(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    df: pd.DataFrame,
) -> None:
    """Save extracted features, labels, encoder, and dataset metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(MODELS_DIR / "features_X.npy", X)
    np.save(MODELS_DIR / "features_y.npy", y)
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    df.to_csv(MODELS_DIR / "dataset_metadata.csv", index=False)

    stats = {
        "total_files": len(df),
        "total_samples_after_augmentation": len(y),
        "n_emotions": len(label_encoder.classes_),
        "emotions": list(label_encoder.classes_),
        "feature_shape": list(X.shape),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated(subset=["filename"]).sum()),
        "emotion_distribution": df["emotion"].value_counts().to_dict(),
        "gender_distribution": df["gender"].value_counts().to_dict(),
        "actor_count": df["actor"].nunique(),
    }
    pd.Series(stats).to_json(MODELS_DIR / "dataset_stats.json")

    logger.info("Saved features and artifacts to %s", MODELS_DIR)


def main() -> None:
    """Run full feature extraction pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_FILES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", DATASET_DIR)
    df = load_dataset()
    display_dataset_stats(df)

    logger.info("Extracting features with augmentation...")
    X, y, label_encoder = build_feature_dataset(df, augment=True)

    logger.info("Normalizing features...")
    X_normalized, _ = normalize_features(X, fit=True)

    save_artifacts(X_normalized, y, label_encoder, df)
    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
