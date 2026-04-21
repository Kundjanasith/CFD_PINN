from __future__ import annotations

import argparse
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from pinn_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DEFAULT_SENSORS,
    load_model_with_weights,
    load_normalization_stats,
    load_training_arrays,
    nearest_node_value,
    normalize_inputs,
    predict_velocity_components,
    velocity_magnitude,
)
from train_nn import build_nn_model # Import the NN model builder


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PINN and Traditional NN models.")
    parser.add_argument("--data", default="combined_pinn_data.csv", help="Training-domain combined CSV for evaluation")
    parser.add_argument("--pinn-weights", default="parametrized_pinn_model.weights.h5", help="PINN model weights")
    parser.add_argument("--nn-weights", default="traditional_nn_model.weights.h5", help="Traditional NN model weights")
    parser.add_argument("--pinn-norm", default="normalization_pinn_stats.npz", help="PINN Normalization stats file")
    parser.add_argument("--nn-norm", default="normalization_nn_stats.npz", help="Traditional NN Normalization stats file")
    parser.add_argument("--fan-speed", type=float, default=1.0, help="Fan speed for sensor validation")
    return parser.parse_args()


def evaluate_model(model, X_norm, Y_true):
    preds = model.predict(X_norm, verbose=0)
    mse = tf.reduce_mean(tf.square(Y_true - preds)).numpy()
    return preds, mse


def main():
    args = parse_args()

    # Load data and raw arrays
    arrays = load_training_arrays(args.data)
    X_raw = arrays["X_raw"]
    U_raw = arrays["U_raw"]
    V_raw = arrays["V_raw"]
    W_raw = arrays["W_raw"]
    P_raw = arrays["P_raw"]

    # Load normalization stats for PINN and Traditional NN
    pinn_X_min, pinn_X_range = load_normalization_stats(args.pinn_norm)
    nn_X_min, nn_X_range = load_normalization_stats(args.nn_norm)

    # For overall model evaluation (MSE), we need a consistent normalization.
    # Let's use the NN's normalization for this general evaluation.
    X_norm_for_nn_eval = normalize_inputs(X_raw, nn_X_min, nn_X_range)
    X_norm_for_pinn_eval = normalize_inputs(X_raw, pinn_X_min, pinn_X_range)


    Y_true_all = tf.concat([U_raw, V_raw, W_raw, P_raw], axis=1)

    # Load PINN model
    pinn_model = load_model_with_weights(args.pinn_weights)
    _ = pinn_model(tf.zeros((1, len(FEATURE_COLUMNS)), dtype=tf.float32)) # Build model with dummy input

    # Load Traditional NN model
    nn_model = build_nn_model()
    _ = nn_model(tf.zeros((1, len(FEATURE_COLUMNS)), dtype=tf.float32)) # Build model with dummy input
    nn_model.load_weights(args.nn_weights)

    print("--- Evaluating PINN ---")
    pinn_preds, pinn_mse = evaluate_model(pinn_model, X_norm_for_pinn_eval, Y_true_all)
    print(f"PINN Total MSE: {pinn_mse:.6f}")

    print("\n--- Evaluating Traditional NN ---")
    nn_preds, nn_mse = evaluate_model(nn_model, X_norm_for_nn_eval, Y_true_all)
    print(f"Traditional NN Total MSE: {nn_mse:.6f}")

    # --- Sensor Validation Comparison ---
    print("\n--- Sensor Validation Comparison ---")

    # Filter CFD data for the specified fan speed
    idx_fan = np.where(np.isclose(X_raw[:, 3], args.fan_speed))[0]
    if len(idx_fan) == 0:
        print(f"Warning: No CFD data found for fan speed {args.fan_speed} in the evaluation dataset.")
        X_cfd_fan = np.empty((0, 3))
        vel_cfd_fan = np.empty((0,))
    else:
        X_cfd_fan = X_raw[idx_fan][:, :3]
        vel_cfd_fan = velocity_magnitude(U_raw[idx_fan].flatten(), V_raw[idx_fan].flatten(), W_raw[idx_fan].flatten())

    results = []
    for name, coords in DEFAULT_SENSORS.items():
        sensor_point_raw = np.array([[coords[0], coords[1], coords[2], args.fan_speed]], dtype=np.float32)

        # PINN prediction
        sensor_point_norm_pinn = normalize_inputs(sensor_point_raw, pinn_X_min, pinn_X_range)
        pinn_pred_components = predict_velocity_components(pinn_model, sensor_point_norm_pinn, batch_size=1)
        pinn_pred_vel = velocity_magnitude(pinn_pred_components[:, 0], pinn_pred_components[:, 1], pinn_pred_components[:, 2])[0]

        # NN prediction
        sensor_point_norm_nn = normalize_inputs(sensor_point_raw, nn_X_min, nn_X_range)
        nn_pred_components = nn_model.predict(sensor_point_norm_nn, verbose=0)
        nn_pred_vel = velocity_magnitude(nn_pred_components[:, 0], nn_pred_components[:, 1], nn_pred_components[:, 2])[0]

        # CFD actual velocity
        if len(X_cfd_fan) > 0:
            actual_vel_cfd, distance_offset, _ = nearest_node_value(X_cfd_fan, coords, vel_cfd_fan)
        else:
            actual_vel_cfd = 0.0
            distance_offset = np.inf # Indicate no CFD data found

        # Calculate errors
        pinn_abs_error = abs(pinn_pred_vel - actual_vel_cfd)
        if actual_vel_cfd > 0.01:
            pinn_error_percent = (pinn_abs_error / actual_vel_cfd) * 100
        else:
            pinn_error_percent = "N/A"

        nn_abs_error = abs(nn_pred_vel - actual_vel_cfd)
        if actual_vel_cfd > 0.01:
            nn_error_percent = (nn_abs_error / actual_vel_cfd) * 100
        else:
            nn_error_percent = "N/A"

        results.append({
            "Sensor Name": name,
            "Target (X,Y,Z)": coords,
            "Dist to CFD Node (m)": distance_offset,
            "CFD Vel (m/s)": actual_vel_cfd,
            "PINN Vel (m/s)": pinn_pred_vel,
            "PINN Abs Error (m/s)": pinn_abs_error,
            "PINN Error (%)": pinn_error_percent,
            "NN Vel (m/s)": nn_pred_vel,
            "NN Abs Error (m/s)": nn_abs_error,
            "NN Error (%)": nn_error_percent,
        })

    df_results = pd.DataFrame(results)
    print(df_results.round(4).to_string(index=False))

    # --- Plotting Comparison ---
    plt.figure(figsize=(12, 7))
    x_pos = np.arange(len(df_results))
    width = 0.25

    cfd_vals = df_results["CFD Vel (m/s)"]
    pinn_vals = df_results["PINN Vel (m/s)"]
    nn_vals = df_results["NN Vel (m/s)"]
    labels = df_results["Sensor Name"]

    plt.bar(x_pos - width, cfd_vals, width, label="CFD", alpha=0.8)
    plt.bar(x_pos, pinn_vals, width, label="PINN", alpha=0.9)
    plt.bar(x_pos + width, nn_vals, width, label="Traditional NN", alpha=0.9)

    plt.title(f"Velocity Comparison at Sensor Locations (Fan = {args.fan_speed})")
    plt.xlabel("Sensor Locations")
    plt.ylabel("Velocity Magnitude (m/s)")
    plt.xticks(x_pos, labels)
    plt.legend(loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"model_comparison_sensor_fan{args.fan_speed}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(f"model_comparison_sensor_fan{args.fan_speed}.eps", format="eps", bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
