from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pinn_utils import DEFAULT_SENSORS, load_model_with_weights, load_normalization_stats, load_training_arrays, nearest_node_value, normalize_inputs, predict_velocity_components, read_cfd_csv, velocity_magnitude


def parse_args():
    parser = argparse.ArgumentParser(description="Validate PINN predictions at specified sensor locations.")
    parser.add_argument("--data", default="combined_pinn_data.csv", help="Training-domain combined CSV")
    parser.add_argument("--weights", default="parametrized_pinn_model.weights.h5")
    parser.add_argument("--norm", default="normalization_stats.npz")
    parser.add_argument("--fan-speed", type=float, required=True, help="Fan speed used for prediction input")
    parser.add_argument("--unseen-cfd", default=None, help="Optional CFD CSV for unseen validation")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model_with_weights(args.weights)
    X_min, X_range = load_normalization_stats(args.norm)

    if args.unseen_cfd:
        df = read_cfd_csv(args.unseen_cfd)
        x_coords = df["x-coordinate"].values
        y_coords = df["y-coordinate"].values
        z_coords = df["z-coordinate"].values
        u_vel = df["x-velocity"].values
        v_vel = df["y-velocity"].values
        w_vel = df["z-velocity"].values
        X_cfd = np.stack([x_coords, y_coords, z_coords], axis=1)
        vel_cfd = velocity_magnitude(u_vel, v_vel, w_vel)
        title = f"Validation: CFD vs PINN at fan speed {args.fan_speed}"
        fig_base = f"validation_sensor_fan{args.fan_speed}"
    else:
        arrays = load_training_arrays(args.data)
        X_raw = arrays["X_raw"]
        U_raw = arrays["U_raw"].flatten()
        V_raw = arrays["V_raw"].flatten()
        W_raw = arrays["W_raw"].flatten()
        idx_fan = np.where(np.isclose(X_raw[:, 3], args.fan_speed))[0]
        if len(idx_fan) == 0:
            raise ValueError(f"No rows found for fan speed {args.fan_speed} in training-domain data.")
        X_cfd = X_raw[idx_fan][:, :3]
        vel_cfd = velocity_magnitude(U_raw[idx_fan], V_raw[idx_fan], W_raw[idx_fan])
        title = f"Velocity Comparison at Sensor Locations (Fan = {args.fan_speed})"
        fig_base = f"sensor_comparison_fan{args.fan_speed}"

    results = []
    for name, coords in DEFAULT_SENSORS.items():
        actual_vel, distance_offset, _ = nearest_node_value(X_cfd, coords, vel_cfd)
        sensor_point = np.array([[coords[0], coords[1], coords[2], args.fan_speed]], dtype=np.float32)
        sensor_norm = normalize_inputs(sensor_point, X_min, X_range)
        pred = predict_velocity_components(model, sensor_norm, batch_size=1)
        pred_vel = velocity_magnitude(pred[:, 0], pred[:, 1], pred[:, 2])[0]
        abs_error = abs(pred_vel - actual_vel)
        error_percent = (abs_error / actual_vel) * 100 if actual_vel > 0.01 else 0.0
        results.append({
            "Sensor Name": name,
            "Target (X,Y,Z)": coords,
            "Dist to CFD Node (m)": distance_offset,
            "CFD Vel (m/s)": actual_vel,
            "PINN Vel (m/s)": pred_vel,
            "Abs Error (m/s)": abs_error,
            "Error (%)": error_percent,
        })

    df_results = pd.DataFrame(results)
    print(df_results.round(4).to_string(index=False))

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(df_results))
    width = 0.35
    cfd_vals = df_results["CFD Vel (m/s)"]
    pinn_vals = df_results["PINN Vel (m/s)"]
    labels = df_results["Sensor Name"]

    bars1 = plt.bar(x_pos - width / 2, cfd_vals, width, label="CFD", alpha=0.8)
    bars2 = plt.bar(x_pos + width / 2, pinn_vals, width, label="PINN", alpha=0.9)
    plt.title(title)
    plt.xlabel("Sensor Locations")
    plt.ylabel("Velocity Magnitude (m/s)")
    plt.xticks(x_pos, labels)
    plt.legend(loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=10)

    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=10)
        err_pct = df_results["Error (%)"].iloc[i]
        max_h = max(cfd_vals.iloc[i], pinn_vals.iloc[i])
        plt.text(x_pos[i], max_h + (max_h * 0.1 if max_h > 0 else 0.05), f"Err: {err_pct:.1f}%", ha="center", color="red", fontweight="bold")

    plt.ylim(0, max(max(cfd_vals), max(pinn_vals)) * 1.3)
    plt.tight_layout()
    plt.savefig(f"{fig_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fig_base}.eps", format="eps", bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
