from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

FEATURE_COLUMNS = ["x-coordinate", "y-coordinate", "z-coordinate", "fan_speed"]
TARGET_COLUMNS = ["x-velocity", "y-velocity", "z-velocity", "pressure"]
VELOCITY_COLUMNS = ["x-velocity", "y-velocity", "z-velocity"]

DEFAULT_SENSORS: Dict[str, Tuple[float, float, float]] = {
    "Sensor_S1": (0.157458, 0.449922, 36.3381),
    "Sensor_S2": (0.128734, 0.443072, 21.9805),
    "Sensor_S3": (0.335913, 0.407175, 8.19691),
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def read_cfd_csv(path: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_columns(df)
    return df


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def process_single_file(filepath: str | os.PathLike, speed_value: float) -> pd.DataFrame:
    df = read_cfd_csv(filepath)
    ensure_required_columns(df, [
        "x-coordinate", "y-coordinate", "z-coordinate",
        "x-velocity", "y-velocity", "z-velocity", "pressure",
    ])
    df = df.dropna().copy()
    df = df[[
        "x-coordinate", "y-coordinate", "z-coordinate",
        "x-velocity", "y-velocity", "z-velocity", "pressure",
    ]]
    df["fan_speed"] = float(speed_value)
    return df


def combine_cases(case_map: Dict[str, float], output_csv: str | os.PathLike) -> pd.DataFrame:
    frames = []
    for filepath, speed in case_map.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        frames.append(process_single_file(filepath, speed))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    return df


def load_training_arrays(csv_path: str | os.PathLike):
    df = read_cfd_csv(csv_path)
    ensure_required_columns(df, FEATURE_COLUMNS + TARGET_COLUMNS)

    X_raw = df[FEATURE_COLUMNS].values.astype(np.float32)
    U_raw = df[["x-velocity"]].values.astype(np.float32)
    V_raw = df[["y-velocity"]].values.astype(np.float32)
    W_raw = df[["z-velocity"]].values.astype(np.float32)
    P_raw = df[["pressure"]].values.astype(np.float32)

    X_min = X_raw.min(axis=0).astype(np.float32)
    X_max = X_raw.max(axis=0).astype(np.float32)
    X_range = X_max - X_min
    X_range = np.where(X_range == 0, 1.0, X_range).astype(np.float32)
    X_norm = ((X_raw - X_min) / X_range).astype(np.float32)

    return {
        "df": df,
        "X_raw": X_raw,
        "X_norm": X_norm,
        "U_raw": U_raw,
        "V_raw": V_raw,
        "W_raw": W_raw,
        "P_raw": P_raw,
        "X_min": X_min,
        "X_range": X_range,
    }


def save_normalization_stats(path: str | os.PathLike, X_min: np.ndarray, X_range: np.ndarray) -> None:
    np.savez(path, X_min=X_min, X_range=X_range)


def load_normalization_stats(path: str | os.PathLike):
    data = np.load(path)
    return data["X_min"].astype(np.float32), data["X_range"].astype(np.float32)


def normalize_inputs(X: np.ndarray, X_min: np.ndarray, X_range: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float32) - X_min) / X_range).astype(np.float32)


def build_fourier_model(hidden_sizes=(128, 128, 128, 128, 64), freqs=(1.0, 2.0, 4.0, 8.0)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(4,), name="inputs")
    xyz = inputs[:, 0:3]
    fan = inputs[:, 3:4]

    features = [xyz]
    for freq in freqs:
        features.append(tf.math.sin(np.pi * freq * xyz))
        features.append(tf.math.cos(np.pi * freq * xyz))

    x = tf.concat(features + [fan], axis=-1)
    for units in hidden_sizes:
        x = tf.keras.layers.Dense(units, activation="swish")(x)
    outputs = tf.keras.layers.Dense(4, name="outputs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="fourier_pinn")


def load_model_with_weights(weights_path: str | os.PathLike) -> tf.keras.Model:
    model = build_fourier_model()
    _ = model(tf.zeros((1, 4), dtype=tf.float32))
    model.load_weights(weights_path)
    return model


def velocity_magnitude(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(u) + np.square(v) + np.square(w))


def predict_velocity_components(model: tf.keras.Model, X_norm: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    preds = model.predict(tf.cast(X_norm, tf.float32), batch_size=batch_size, verbose=0)
    return preds


def nearest_node_value(points_xyz: np.ndarray, target_xyz: Tuple[float, float, float], values: np.ndarray):
    sx, sy, sz = target_xyz
    distances = np.sqrt(
        np.square(points_xyz[:, 0] - sx) +
        np.square(points_xyz[:, 1] - sy) +
        np.square(points_xyz[:, 2] - sz)
    )
    idx = int(np.argmin(distances))
    return values[idx], float(distances[idx]), idx
