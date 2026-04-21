from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from pinn_utils import load_model_with_weights, load_normalization_stats, load_training_arrays, normalize_inputs, predict_velocity_components, velocity_magnitude


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CFD vs PINN on a ZX plane at a target Y coordinate.")
    parser.add_argument("--data", default="combined_pinn_data.csv")
    parser.add_argument("--weights", default="parametrized_pinn_model.weights.h5")
    parser.add_argument("--norm", default="normalization_stats.npz")
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
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

    idx_fan = np.where(np.isclose(X_raw[:, 3], fan_target))[0]
    X_cfd = X_raw[idx_fan]
    vel_cfd = velocity_magnitude(U_raw[idx_fan], V_raw[idx_fan], W_raw[idx_fan])

    idx_plane = np.where(np.abs(X_cfd[:, 1] - args.target_y) <= args.tolerance)[0]
    if len(idx_plane) < 10:
        raise ValueError(f"Only {len(idx_plane)} CFD points found. Increase tolerance.")

    X_plane = X_cfd[idx_plane]
    x_points = X_plane[:, 0]
    z_points = X_plane[:, 2]
    vel_cfd_plane = vel_cfd[idx_plane]

    X_plane_norm = normalize_inputs(X_plane, X_min, X_range)
    preds = predict_velocity_components(model, X_plane_norm, batch_size=args.batch_size)
    vel_pred_plane = velocity_magnitude(preds[:, 0], preds[:, 1], preds[:, 2])

    vmin_shared = min(np.min(vel_cfd_plane), np.min(vel_pred_plane))
    vmax_shared = max(np.max(vel_cfd_plane), np.max(vel_pred_plane))

    fig, axes = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
    x_min_lim, x_max_lim = np.min(X_raw[:, 0]), np.max(X_raw[:, 0])
    z_min_lim, z_max_lim = np.min(X_raw[:, 2]), np.max(X_raw[:, 2])

    sc1 = axes[0].scatter(z_points, x_points, c=vel_cfd_plane, cmap="turbo", s=15, alpha=0.9, edgecolors="none", vmin=vmin_shared, vmax=vmax_shared)
    axes[0].set_title(f"Ansys Simulation: CFD Nodes\n(Y ≈ {args.target_y} m)")
    axes[0].set_xlabel("Z-Coordinate (Length, m)")
    axes[0].set_ylabel("X-Coordinate (Width, m)")
    axes[0].set_xlim(z_min_lim, z_max_lim)
    axes[0].set_ylim(x_min_lim, x_max_lim)
    axes[0].set_facecolor("#f0f0f0")
    axes[0].grid(color="white", linestyle="-", linewidth=1, alpha=0.7)

    sc2 = axes[1].scatter(z_points, x_points, c=vel_pred_plane, cmap="turbo", s=15, alpha=0.9, edgecolors="none", vmin=vmin_shared, vmax=vmax_shared)
    axes[1].set_title(f"Prediction: PINN Model\n(Y ≈ {args.target_y} m)")
    axes[1].set_xlabel("Z-Coordinate (Length, m)")
    axes[1].set_xlim(z_min_lim, z_max_lim)
    axes[1].set_ylim(x_min_lim, x_max_lim)
    axes[1].set_facecolor("#f0f0f0")
    axes[1].grid(color="white", linestyle="-", linewidth=1, alpha=0.7)

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), fraction=0.015, pad=0.03)
    cbar.set_label("Velocity Magnitude (m/s)")
    plt.suptitle(f"CFD vs PINN Velocity Comparison at ZX Plane (Fan = {fan_target})", y=1.05)
    plt.tight_layout()

    fig_base = f"cfd_vs_pinn_y{args.target_y}_fan{fan_target}"
    plt.savefig(f"{fig_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{fig_base}.eps", format="eps", bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
