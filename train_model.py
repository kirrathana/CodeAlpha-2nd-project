"""
Training module for Speech Emotion Recognition CNN+LSTM model.

Handles data splitting, model training, evaluation, and artifact saving.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from feature_extraction import (
    MODELS_DIR,
    build_feature_dataset,
    load_dataset,
    normalize_features,
    get_n_features,
    MAX_FRAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EPOCHS = 50
BATCH_SIZE = 32
RANDOM_STATE = 42


def build_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Build Hybrid CNN + LSTM architecture for emotion classification.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.LSTM(128, return_sequences=False),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history, save_dir: Path) -> None:
    """Plot and save accuracy and loss curves."""
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax.set_title("Model Accuracy vs Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "training_accuracy.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_title("Model Loss vs Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "training_loss.png", dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    save_dir: Path,
) -> None:
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list,
    save_dir: Path,
) -> None:
    """Generate and save ROC curve."""
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{cls} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png", dpi=150)
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list,
    save_dir: Path,
) -> None:
    """Generate and save Precision-Recall curve."""
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, linewidth=2, label=f"{cls} (AP={ap:.2f})")

    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "pr_curve.png", dpi=150)
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    save_dir: Path,
) -> dict:
    """Run full model evaluation and save metrics."""
    classes = list(label_encoder.classes_)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_prob = model.predict(X_test, verbose=0)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"\nBest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("=" * 60 + "\n")

    plot_confusion_matrix(y_test, y_pred, classes, save_dir)
    plot_roc_curve(y_test, y_prob, classes, save_dir)
    plot_pr_curve(y_test, y_prob, classes, save_dir)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "best_accuracy": float(accuracy),
        "classification_report": report,
        "num_classes": len(classes),
        "classes": classes,
    }
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    """Run full training pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    features_path = MODELS_DIR / "features_X.npy"
    labels_path = MODELS_DIR / "features_y.npy"

    if features_path.exists() and labels_path.exists():
        logger.info("Loading pre-extracted features...")
        X = np.load(features_path)
        y = np.load(labels_path)
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    else:
        logger.info("Extracting features from dataset...")
        df = load_dataset()
        X, y, label_encoder = build_feature_dataset(df, augment=True)
        X, scaler = normalize_features(X, fit=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        np.save(features_path, X)
        np.save(labels_path, y)
        joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    logger.info("Scaling features with StandardScaler...")
    X_train_scaled, scaler = normalize_features(X_train, fit=True)
    X_test_scaled, _ = normalize_features(X_test, scaler=scaler, fit=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    n_features = get_n_features()
    input_shape = (MAX_FRAMES, n_features)
    num_classes = len(label_encoder.classes_)

    logger.info("Building model with input shape %s, %d classes", input_shape, num_classes)
    model = build_model(input_shape, num_classes)
    model.summary()

    cb_early = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1,
    )
    cb_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        str(MODELS_DIR / "emotion_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    logger.info("Training for %d epochs...", EPOCHS)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cb_early, cb_lr, cb_checkpoint],
        verbose=1,
    )

    model.save(MODELS_DIR / "emotion_model.keras")

    with open(MODELS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    plot_training_history(history, MODELS_DIR)

    metrics = evaluate_model(model, X_test_scaled, y_test, label_encoder, MODELS_DIR)

    architecture = []
    model.summary(print_fn=lambda x: architecture.append(x))
    with open(MODELS_DIR / "model_architecture.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(architecture))

    logger.info("Training complete. Best accuracy: %.4f", metrics["best_accuracy"])


if __name__ == "__main__":
    main()
