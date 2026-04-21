from __future__ import annotations

import argparse
import time
import os
import tqdm
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import matplotlib.pyplot as plt
import tensorflow as tf

from pinn_utils import build_fourier_model, load_training_arrays, save_normalization_stats


RHO = 1.225
NU = 1.5e-5


def parse_args():
    parser = argparse.ArgumentParser(description="Train the parametrized Fourier-feature PINN model.")
    parser.add_argument("--data", default="../Train/combined_pinn_data.csv")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--weights", default="parametrized_pinn_model.weights.h5")
    parser.add_argument("--norm", default="normalization_pinn_stats.npz")
    parser.add_argument("--initial-lr", type=float, default=0.002)
    parser.add_argument("--decay-steps", type=int, default=50000)
    parser.add_argument("--decay-rate", type=float, default=0.90)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--weight-data", type=float, default=3.0)
    parser.add_argument("--weight-physics", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    arrays = load_training_arrays(args.data)

    X_train_norm = arrays["X_norm"]
    U_raw = arrays["U_raw"]
    V_raw = arrays["V_raw"]
    W_raw = arrays["W_raw"]
    X_range = tf.constant(arrays["X_range"], dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X_train_norm, U_raw, V_raw, W_raw))
    dataset = dataset.shuffle(buffer_size=50000).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)

    model = build_fourier_model()
    _ = model(tf.zeros((1, 4), dtype=tf.float32))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.initial_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=False,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=args.clipnorm)

    @tf.function
    def train_step(X_batch_norm, u_true, v_true, w_true):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_batch_norm)
            preds = model(X_batch_norm)
            u, v, w, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], preds[:, 3:4]

            du_dX_norm = tape.gradient(u, X_batch_norm)
            dv_dX_norm = tape.gradient(v, X_batch_norm)
            dw_dX_norm = tape.gradient(w, X_batch_norm)
            dp_dX_norm = tape.gradient(p, X_batch_norm)

            inv_range_x = 1.0 / X_range[0]
            inv_range_y = 1.0 / X_range[1]
            inv_range_z = 1.0 / X_range[2]

            u_x = du_dX_norm[:, 0:1] * inv_range_x
            u_y = du_dX_norm[:, 1:2] * inv_range_y
            u_z = du_dX_norm[:, 2:3] * inv_range_z
            v_x = dv_dX_norm[:, 0:1] * inv_range_x
            v_y = dv_dX_norm[:, 1:2] * inv_range_y
            v_z = dv_dX_norm[:, 2:3] * inv_range_z
            w_x = dw_dX_norm[:, 0:1] * inv_range_x
            w_y = dw_dX_norm[:, 1:2] * inv_range_y
            w_z = dw_dX_norm[:, 2:3] * inv_range_z
            p_x = dp_dX_norm[:, 0:1] * inv_range_x
            p_y = dp_dX_norm[:, 1:2] * inv_range_y
            p_z = dp_dX_norm[:, 2:3] * inv_range_z

            u_xx = tape.gradient(du_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x ** 2)
            u_yy = tape.gradient(du_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y ** 2)
            u_zz = tape.gradient(du_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z ** 2)
            v_xx = tape.gradient(dv_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x ** 2)
            v_yy = tape.gradient(dv_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y ** 2)
            v_zz = tape.gradient(dv_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z ** 2)
            w_xx = tape.gradient(dw_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x ** 2)
            w_yy = tape.gradient(dw_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y ** 2)
            w_zz = tape.gradient(dw_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z ** 2)

            res_continuity = u_x + v_y + w_z
            res_ns_x = (u * u_x + v * u_y + w * u_z) + (1.0 / RHO) * p_x - NU * (u_xx + u_yy + u_zz)
            res_ns_y = (u * v_x + v * v_y + w * v_z) + (1.0 / RHO) * p_y - NU * (v_xx + v_yy + v_zz)
            res_ns_z = (u * w_x + v * w_y + w * w_z) + (1.0 / RHO) * p_z - NU * (w_xx + w_yy + w_zz)

            loss_data = (
                tf.reduce_mean(tf.square(u_true - u)) +
                tf.reduce_mean(tf.square(v_true - v)) +
                tf.reduce_mean(tf.square(w_true - w))
            )
            loss_physics = (
                tf.reduce_mean(tf.square(res_continuity)) +
                tf.reduce_mean(tf.square(res_ns_x)) +
                tf.reduce_mean(tf.square(res_ns_y)) +
                tf.reduce_mean(tf.square(res_ns_z))
            )
            total_loss = args.weight_data * loss_data + args.weight_physics * loss_physics

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        del tape
        return total_loss, loss_data, loss_physics

    total_hist, data_hist, phys_hist = [], [], []
    start_time = time.time()

    try:
        for epoch in tqdm.tqdm(range(args.epochs)):
            epoch_loss_total = 0.0
            epoch_loss_data = 0.0
            epoch_loss_phys = 0.0
            steps = 0

            for X_batch, u_b, v_b, w_b in dataset:
                l_total, l_data, l_phys = train_step(X_batch, u_b, v_b, w_b)
                epoch_loss_total += float(l_total.numpy())
                epoch_loss_data += float(l_data.numpy())
                epoch_loss_phys += float(l_phys.numpy())
                steps += 1

            avg_total = epoch_loss_total / steps
            avg_data = epoch_loss_data / steps
            avg_phys = epoch_loss_phys / steps
            total_hist.append(avg_total)
            data_hist.append(avg_data)
            phys_hist.append(avg_phys)

            if (epoch + 1) % args.print_freq == 0:
                # current_lr = float(optimizer.learning_rate(optimizer.iterations).numpy())
                current_lr = optimizer.learning_rate.numpy()
                print(
                    f"Epoch {epoch + 1:4d}/{args.epochs} | LR: {current_lr:.6f} | "
                    f"Total: {avg_total:.6f} | Data: {avg_data:.6f} | Phys: {avg_phys:.6f}"
                )

            if (epoch + 1) % args.save_freq == 0:
                model.save_weights(args.weights)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving latest weights...")

    model.save_weights(args.weights)
    save_normalization_stats(args.norm, arrays["X_min"], arrays["X_range"])

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"Saved weights: {args.weights}")
    print(f"Saved normalization stats: {args.norm}")
    print(f"Training time: {elapsed_min:.2f} minutes")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(total_hist) + 1), total_hist, label="Total Loss", linewidth=2)
    plt.plot(range(1, len(data_hist) + 1), data_hist, label="Data Loss", linestyle="--")
    plt.plot(range(1, len(phys_hist) + 1), phys_hist, label="Physics Loss", linestyle="-.")
    plt.yscale("log")
    plt.title("Training Loss Breakdown (Fourier Features Enabled)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    fig_base = f"training_loss_fourier_epoch{len(total_hist)}"
    plt.savefig(f"{fig_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_base}.eps", format="eps", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
