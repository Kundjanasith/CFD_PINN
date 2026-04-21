from __future__ import annotations

import argparse
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from pinn_utils import (
    load_training_arrays,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_fourier_model,
)

RHO = 1.225
NU = 1.5e-5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NN, PINN v1, and PINN v2 on the same dataset."
    )
    parser.add_argument("--data", default="../Train/combined_pinn_data.csv",
                        help="CSV used for evaluation.")
    parser.add_argument("--batch-size", type=int, default=4096)

    parser.add_argument("--nn-weights", default="traditional_nn_model.weights.h5")
    parser.add_argument("--pinn1-weights", default="parametrized_pinn_model.weights.h5")
    parser.add_argument("--pinn2-weights", default="pinn_v2_model.weights.h5")

    parser.add_argument("--eval-split", choices=["all", "tail20"], default="tail20",
                        help="all = evaluate whole CSV, tail20 = last 20 percent only.")
    parser.add_argument("--save-prefix", default="model_eval")
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def build_nn_model(hidden_sizes=(128, 128, 128, 128, 64)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(len(FEATURE_COLUMNS),), name="inputs")
    x = inputs
    for i, units in enumerate(hidden_sizes):
        x = tf.keras.layers.Dense(units, activation="swish", name=f"dense_{i+1}")(x)
    outputs = tf.keras.layers.Dense(len(TARGET_COLUMNS), name="outputs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="traditional_nn")


def build_pinn_v2_model(hidden_sizes=(128, 128, 128, 128, 64)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(len(FEATURE_COLUMNS),), name="inputs")
    x = inputs
    for i, units in enumerate(hidden_sizes):
        x = tf.keras.layers.Dense(units, activation="swish", name=f"dense_{i+1}")(x)
    outputs = tf.keras.layers.Dense(len(TARGET_COLUMNS), name="outputs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="pinn_v2")


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-8):
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def relative_l2(y_true, y_pred, eps=1e-12):
    num = np.sqrt(np.sum((y_true - y_pred) ** 2))
    den = np.sqrt(np.sum(y_true ** 2)) + eps
    return float(num / den)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    names = ["u", "v", "w", "p"]
    rows = []
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append({
            "target": name,
            "MAE": mae(yt, yp),
            "RMSE": rmse(yt, yp),
            "R2": r2_score_manual(yt, yp),
            "MAPE_percent": mape(yt, yp),
            "SMAPE_percent": smape(yt, yp),
            "RelL2": relative_l2(yt, yp),
        })

    speed_true = np.sqrt(np.sum(y_true[:, 0:3] ** 2, axis=1))
    speed_pred = np.sqrt(np.sum(y_pred[:, 0:3] ** 2, axis=1))
    rows.append({
        "target": "speed_mag",
        "MAE": mae(speed_true, speed_pred),
        "RMSE": rmse(speed_true, speed_pred),
        "R2": r2_score_manual(speed_true, speed_pred),
        "MAPE_percent": mape(speed_true, speed_pred),
        "SMAPE_percent": smape(speed_true, speed_pred),
        "RelL2": relative_l2(speed_true, speed_pred),
    })
    return rows


def load_eval_arrays(data_path: str, eval_split: str):
    arrays = load_training_arrays(data_path)

    X = arrays["X_norm"].numpy() if hasattr(arrays["X_norm"], "numpy") else np.asarray(arrays["X_norm"])
    U = arrays["U_raw"].numpy() if hasattr(arrays["U_raw"], "numpy") else np.asarray(arrays["U_raw"])
    V = arrays["V_raw"].numpy() if hasattr(arrays["V_raw"], "numpy") else np.asarray(arrays["V_raw"])
    W = arrays["W_raw"].numpy() if hasattr(arrays["W_raw"], "numpy") else np.asarray(arrays["W_raw"])
    P = arrays["P_raw"].numpy() if hasattr(arrays["P_raw"], "numpy") else np.asarray(arrays["P_raw"])
    X_range = arrays["X_range"].numpy() if hasattr(arrays["X_range"], "numpy") else np.asarray(arrays["X_range"])

    Y = np.concatenate([U, V, W, P], axis=1)

    n = len(X)
    if eval_split == "tail20":
        start = int(0.8 * n)
        X = X[start:]
        Y = Y[start:]

    return X.astype(np.float32), Y.astype(np.float32), X_range.astype(np.float32)


def predict_model(model: tf.keras.Model, X: np.ndarray, batch_size: int):
    preds = model.predict(X, batch_size=batch_size, verbose=0)
    return np.asarray(preds, dtype=np.float32)


def safe_load_weights(model: tf.keras.Model, path: str):
    if not path or not os.path.exists(path):
        print(f"[WARN] Weights not found, skipping: {path}")
        return False
    try:
        model.load_weights(path)
        print(f"[OK] Loaded weights: {path}")
        return True
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return False


def compute_physics_residual_summary(model: tf.keras.Model, X: np.ndarray, X_range: np.ndarray, batch_size: int = 2048):
    """
    Compute mean squared residual summaries:
      continuity, ns_x, ns_y, ns_z
    """
    inv_range_x = tf.constant(1.0 / X_range[0], dtype=tf.float32)
    inv_range_y = tf.constant(1.0 / X_range[1], dtype=tf.float32)
    inv_range_z = tf.constant(1.0 / X_range[2], dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)

    cont_vals = []
    mx_vals = []
    my_vals = []
    mz_vals = []

    for X_batch in ds:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_batch)
            preds = model(X_batch, training=False)

            u = preds[:, 0:1]
            v = preds[:, 1:2]
            w = preds[:, 2:3]
            p = preds[:, 3:4]

            du = tape.gradient(u, X_batch)
            dv = tape.gradient(v, X_batch)
            dw = tape.gradient(w, X_batch)
            dp = tape.gradient(p, X_batch)

            u_x = du[:, 0:1] * inv_range_x
            u_y = du[:, 1:2] * inv_range_y
            u_z = du[:, 2:3] * inv_range_z

            v_x = dv[:, 0:1] * inv_range_x
            v_y = dv[:, 1:2] * inv_range_y
            v_z = dv[:, 2:3] * inv_range_z

            w_x = dw[:, 0:1] * inv_range_x
            w_y = dw[:, 1:2] * inv_range_y
            w_z = dw[:, 2:3] * inv_range_z

            p_x = dp[:, 0:1] * inv_range_x
            p_y = dp[:, 1:2] * inv_range_y
            p_z = dp[:, 2:3] * inv_range_z

            u_xx = tape.gradient(du[:, 0:1], X_batch)[:, 0:1] * (inv_range_x ** 2)
            u_yy = tape.gradient(du[:, 1:2], X_batch)[:, 1:2] * (inv_range_y ** 2)
            u_zz = tape.gradient(du[:, 2:3], X_batch)[:, 2:3] * (inv_range_z ** 2)

            v_xx = tape.gradient(dv[:, 0:1], X_batch)[:, 0:1] * (inv_range_x ** 2)
            v_yy = tape.gradient(dv[:, 1:2], X_batch)[:, 1:2] * (inv_range_y ** 2)
            v_zz = tape.gradient(dv[:, 2:3], X_batch)[:, 2:3] * (inv_range_z ** 2)

            w_xx = tape.gradient(dw[:, 0:1], X_batch)[:, 0:1] * (inv_range_x ** 2)
            w_yy = tape.gradient(dw[:, 1:2], X_batch)[:, 1:2] * (inv_range_y ** 2)
            w_zz = tape.gradient(dw[:, 2:3], X_batch)[:, 2:3] * (inv_range_z ** 2)

            res_cont = u_x + v_y + w_z
            res_ns_x = (u * u_x + v * u_y + w * u_z) + (1.0 / RHO) * p_x - NU * (u_xx + u_yy + u_zz)
            res_ns_y = (u * v_x + v * v_y + w * v_z) + (1.0 / RHO) * p_y - NU * (v_xx + v_yy + v_zz)
            res_ns_z = (u * w_x + v * w_y + w * w_z) + (1.0 / RHO) * p_z - NU * (w_xx + w_yy + w_zz)

        cont_vals.append(float(tf.reduce_mean(tf.square(res_cont)).numpy()))
        mx_vals.append(float(tf.reduce_mean(tf.square(res_ns_x)).numpy()))
        my_vals.append(float(tf.reduce_mean(tf.square(res_ns_y)).numpy()))
        mz_vals.append(float(tf.reduce_mean(tf.square(res_ns_z)).numpy()))
        del tape

    return {
        "continuity_mse": float(np.mean(cont_vals)),
        "ns_x_mse": float(np.mean(mx_vals)),
        "ns_y_mse": float(np.mean(my_vals)),
        "ns_z_mse": float(np.mean(mz_vals)),
    }


def save_metrics_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    header = f"{'Model':<12} {'Target':<10} {'MAE':>12} {'RMSE':>12} {'R2':>12} {'MAPE%':>12} {'SMAPE%':>12} {'RelL2':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<12} {r['target']:<10} "
            f"{r['MAE']:>12.6f} {r['RMSE']:>12.6f} {r['R2']:>12.6f} "
            f"{r['MAPE_percent']:>12.4f} {r['SMAPE_percent']:>12.4f} {r['RelL2']:>12.6f}"
        )


def print_physics_table(rows, title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    header = f"{'Model':<12} {'Cont MSE':>16} {'NS-x MSE':>16} {'NS-y MSE':>16} {'NS-z MSE':>16}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<12} "
            f"{r['continuity_mse']:>16.6e} {r['ns_x_mse']:>16.6e} "
            f"{r['ns_y_mse']:>16.6e} {r['ns_z_mse']:>16.6e}"
        )


def make_bar_plot(summary_rows, prefix):
    """
    Make one plot per target for MAE comparison.
    """
    targets = ["u", "v", "w", "p", "speed_mag"]
    for target in targets:
        rows = [r for r in summary_rows if r["target"] == target]
        if not rows:
            continue
        labels = [r["model"] for r in rows]
        vals = [r["MAE"] for r in rows]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, vals)
        plt.title(f"MAE Comparison - {target}")
        plt.xlabel("Model")
        plt.ylabel("MAE")
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{prefix}_mae_{target}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{prefix}_mae_{target}.eps", format="eps", bbox_inches="tight")
        plt.close()


def make_scatter_plot(y_true, pred_dict, prefix):
    """
    Make parity plots for each target and each model.
    """
    targets = ["u", "v", "w", "p"]
    for i, target in enumerate(targets):
        for model_name, y_pred in pred_dict.items():
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true[:, i], y_pred[:, i], s=6, alpha=0.4)
            lo = min(np.min(y_true[:, i]), np.min(y_pred[:, i]))
            hi = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
            plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.title(f"Parity Plot - {model_name} - {target}")
            plt.xlabel(f"True {target}")
            plt.ylabel(f"Predicted {target}")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(f"{prefix}_parity_{model_name}_{target}.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{prefix}_parity_{model_name}_{target}.eps", format="eps", bbox_inches="tight")
            plt.close()


def main():
    args = parse_args()

    X_eval, Y_eval, X_range = load_eval_arrays(args.data, args.eval_split)
    print(f"Evaluation samples: {len(X_eval)}")
    print(f"Input shape: {X_eval.shape}")
    print(f"Target shape: {Y_eval.shape}")

    models = {}

    # NN
    nn_model = build_nn_model()
    _ = nn_model(tf.zeros((1, len(FEATURE_COLUMNS)), dtype=tf.float32))
    if safe_load_weights(nn_model, args.nn_weights):
        models["NN"] = nn_model

    # PINN v1
    pinn1_model = build_fourier_model()
    _ = pinn1_model(tf.zeros((1, len(FEATURE_COLUMNS)), dtype=tf.float32))
    if safe_load_weights(pinn1_model, args.pinn1_weights):
        models["PINN_v1"] = pinn1_model

    # PINN v2
    pinn2_model = build_pinn_v2_model()
    _ = pinn2_model(tf.zeros((1, len(FEATURE_COLUMNS)), dtype=tf.float32))
    if safe_load_weights(pinn2_model, args.pinn2_weights):
        models["PINN_v2"] = pinn2_model

    if not models:
        print("No model weights could be loaded. Exiting.")
        return

    metric_rows_all = []
    physics_rows_all = []
    pred_dict = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} ...")
        Y_pred = predict_model(model, X_eval, batch_size=args.batch_size)
        pred_dict[model_name] = Y_pred

        metric_rows = compute_metrics(Y_eval, Y_pred)
        for row in metric_rows:
            row["model"] = model_name
            metric_rows_all.append(row)

        physics_summary = compute_physics_residual_summary(
            model=model,
            X=X_eval,
            X_range=X_range,
            batch_size=min(args.batch_size, 2048),
        )
        physics_summary["model"] = model_name
        physics_rows_all.append(physics_summary)

    metric_csv = f"{args.save_prefix}_metrics.csv"
    physics_csv = f"{args.save_prefix}_physics.csv"
    save_metrics_csv(metric_rows_all, metric_csv)
    save_metrics_csv(physics_rows_all, physics_csv)

    print_table(metric_rows_all, "Prediction Metrics")
    print_physics_table(physics_rows_all, "Physics Residual Metrics")

    print(f"\nSaved metrics CSV: {metric_csv}")
    print(f"Saved physics CSV: {physics_csv}")

    if args.make_plots:
        make_bar_plot(metric_rows_all, args.save_prefix)
        make_scatter_plot(Y_eval, pred_dict, args.save_prefix)
        print(f"Saved plots with prefix: {args.save_prefix}")


if __name__ == "__main__":
    main()
