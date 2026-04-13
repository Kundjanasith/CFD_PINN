from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from pinn_utils import load_model_with_weights, load_normalization_stats, load_training_arrays, normalize_inputs, predict_velocity_components, velocity_magnitude


def parse_args():
    parser = argparse.ArgumentParser(description="Compare node-by-node CFD vs PINN velocities for one fan speed.")
    parser.add_argument("--data", default="combined_pinn_data.csv")
    parser.add_argument("--weights", default="parametrized_pinn_model.weights.h5")
    parser.add_argument("--norm", default="normalization_stats.npz")
    parser.add_argument("--fan-speed", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    arrays = load_training_arrays(args.data)
    model = load_model_with_weights(args.weights)
    X_min, X_range = load_normalization_stats(args.norm)

    X_raw = arrays["X_raw"]
    U_raw = arrays["U_raw"].flatten()
    V_raw = arrays["V_raw"].flatten()
    W_raw = arrays["W_raw"].flatten()

    available_fans = np.unique(X_raw[:, 3])
    fan_target = available_fans[-1] if args.fan_speed is None else args.fan_speed
    print(f"Available fan speeds: {available_fans}")
    print(f"Using fan speed: {fan_target}")

    idx = np.where(np.isclose(X_raw[:, 3], fan_target))[0]
    if len(idx) == 0:
        raise ValueError(f"No rows found for fan speed {fan_target}")

    X_nodes = X_raw[idx]
    vel_true = velocity_magnitude(U_raw[idx], V_raw[idx], W_raw[idx])
    X_nodes_norm = normalize_inputs(X_nodes, X_min, X_range)
    preds = predict_velocity_components(model, X_nodes_norm, batch_size=args.batch_size)
    vel_pred = velocity_magnitude(preds[:, 0], preds[:, 1], preds[:, 2])

    plt.figure(figsize=(8, 8))
    plt.scatter(vel_true, vel_pred, alpha=0.3, edgecolors="none", label="Nodes")
    min_val = min(np.min(vel_true), np.min(vel_pred))
    max_val = max(np.max(vel_true), np.max(vel_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    plt.title(f"Node-by-Node Comparison (Fan Speed = {fan_target})")
    plt.xlabel("CFD Actual Velocity (m/s)")
    plt.ylabel("PINN Predicted Velocity (m/s)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_base = f"node_comparison_fan{fan_target}"
    plt.savefig(f"{fig_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_base}.eps", format="eps", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
