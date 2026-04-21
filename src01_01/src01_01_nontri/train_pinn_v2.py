from __future__ import annotations

import argparse
import time
import os
import tqdm
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib.pyplot as plt
import tensorflow as tf

from pinn_utils import (
    load_training_arrays,
    save_normalization_stats,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)

RHO = 1.225
NU = 1.5e-5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PINN v2: supervised u,v,w,p + staged physics regularization."
    )
    parser.add_argument("--data", default="../Train/combined_pinn_data.csv")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--weights", default="pinn_v2_model.weights.h5")
    parser.add_argument("--norm", default="normalization_pinn_v2_stats.npz")

    parser.add_argument("--initial-lr", type=float, default=0.002)
    parser.add_argument("--decay-steps", type=int, default=50000)
    parser.add_argument("--decay-rate", type=float, default=0.90)
    parser.add_argument("--clipnorm", type=float, default=1.0)

    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128, 128, 128, 64])

    # staged training schedule
    parser.add_argument("--pretrain-epochs", type=int, default=1000,
                        help="Pure data-fitting phase.")
    parser.add_argument("--ramp-epochs", type=int, default=1000,
                        help="Small physics weight phase after pretraining.")

    parser.add_argument("--phys-weight-ramp", type=float, default=1e-4)
    parser.add_argument("--phys-weight-final", type=float, default=1e-3)

    # residual weights
    parser.add_argument("--lambda-cont", type=float, default=1.0)
    parser.add_argument("--lambda-mx", type=float, default=0.1)
    parser.add_argument("--lambda-my", type=float, default=0.1)
    parser.add_argument("--lambda-mz", type=float, default=0.1)

    # target weights
    parser.add_argument("--weight-u", type=float, default=1.0)
    parser.add_argument("--weight-v", type=float, default=1.0)
    parser.add_argument("--weight-w", type=float, default=1.0)
    parser.add_argument("--weight-p", type=float, default=1.0)

    # optional warm start from NN
    parser.add_argument("--init-from-nn", default="",
                        help="Optional path to trained NN weights for warm start.")
    return parser.parse_args()


def build_pinn_v2_model(hidden_sizes=(128, 128, 128, 128, 64)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(len(FEATURE_COLUMNS),), name="inputs")
    x = inputs
    for i, units in enumerate(hidden_sizes):
        x = tf.keras.layers.Dense(units, activation="swish", name=f"dense_{i+1}")(x)
    outputs = tf.keras.layers.Dense(len(TARGET_COLUMNS), name="outputs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="pinn_v2")


def get_physics_weight(epoch: int, pretrain_epochs: int, ramp_epochs: int,
                       phys_weight_ramp: float, phys_weight_final: float) -> float:
    if epoch < pretrain_epochs:
        return 0.0
    if epoch < pretrain_epochs + ramp_epochs:
        return phys_weight_ramp
    return phys_weight_final


def main():
    args = parse_args()
    arrays = load_training_arrays(args.data)

    X_train_norm = arrays["X_norm"]
    U_raw = arrays["U_raw"]
    V_raw = arrays["V_raw"]
    W_raw = arrays["W_raw"]
    P_raw = arrays["P_raw"]
    X_range = tf.constant(arrays["X_range"], dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (X_train_norm, U_raw, V_raw, W_raw, P_raw)
    )
    dataset = dataset.shuffle(buffer_size=50000).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)

    model = build_pinn_v2_model(tuple(args.hidden_sizes))
    _ = model(tf.zeros((1, len(FEATURE_COLUMNS)), dtype=tf.float32))

    if args.init_from_nn:
        try:
            model.load_weights(args.init_from_nn)
            print(f"Warm-started from NN weights: {args.init_from_nn}")
        except Exception as e:
            print(f"Warning: could not load init weights from {args.init_from_nn}")
            print(f"Reason: {e}")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.initial_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=False,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=args.clipnorm
    )

    @tf.function
    def train_step(X_batch_norm, u_true, v_true, w_true, p_true, phys_weight):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X_batch_norm)
            preds = model(X_batch_norm)

            u = preds[:, 0:1]
            v = preds[:, 1:2]
            w = preds[:, 2:3]
            p = preds[:, 3:4]

            # First derivatives wrt normalized inputs
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

            # Second derivatives
            u_xx = tape.gradient(du_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x ** 2)
            u_yy = tape.gradient(du_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y ** 2)
            u_zz = tape.gradient(du_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z ** 2)

            v_xx = tape.gradient(dv_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x ** 2)
            v_yy = tape.gradient(dv_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y ** 2)
            v_zz = tape.gradient(dv_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z ** 2)

            w_xx = tape.gradient(dw_dX_norm[:, 0:1], X_batch_norm)[:, 0:1] * (inv_range_x ** 2)
            w_yy = tape.gradient(dw_dX_norm[:, 1:2], X_batch_norm)[:, 1:2] * (inv_range_y ** 2)
            w_zz = tape.gradient(dw_dX_norm[:, 2:3], X_batch_norm)[:, 2:3] * (inv_range_z ** 2)

            # PDE residuals
            res_cont = u_x + v_y + w_z

            res_ns_x = (u * u_x + v * u_y + w * u_z) + (1.0 / RHO) * p_x - NU * (u_xx + u_yy + u_zz)
            res_ns_y = (u * v_x + v * v_y + w * v_z) + (1.0 / RHO) * p_y - NU * (v_xx + v_yy + v_zz)
            res_ns_z = (u * w_x + v * w_y + w * w_z) + (1.0 / RHO) * p_z - NU * (w_xx + w_yy + w_zz)

            # supervised data loss on all 4 outputs
            loss_u = tf.reduce_mean(tf.square(u_true - u))
            loss_v = tf.reduce_mean(tf.square(v_true - v))
            loss_w = tf.reduce_mean(tf.square(w_true - w))
            loss_p = tf.reduce_mean(tf.square(p_true - p))

            loss_data = (
                args.weight_u * loss_u +
                args.weight_v * loss_v +
                args.weight_w * loss_w +
                args.weight_p * loss_p
            )

            # physics loss
            loss_cont = tf.reduce_mean(tf.square(res_cont))
            loss_mx = tf.reduce_mean(tf.square(res_ns_x))
            loss_my = tf.reduce_mean(tf.square(res_ns_y))
            loss_mz = tf.reduce_mean(tf.square(res_ns_z))

            loss_phys = (
                args.lambda_cont * loss_cont +
                args.lambda_mx * loss_mx +
                args.lambda_my * loss_my +
                args.lambda_mz * loss_mz
            )

            total_loss = loss_data + phys_weight * loss_phys

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        del tape

        return (
            total_loss,
            loss_data,
            loss_phys,
            loss_u, loss_v, loss_w, loss_p,
            loss_cont, loss_mx, loss_my, loss_mz
        )

    total_hist = []
    data_hist = []
    phys_hist = []
    phys_weight_hist = []

    start_time = time.time()

    try:
        for epoch in tqdm.tqdm(range(args.epochs)):
            phys_weight = tf.constant(
                get_physics_weight(
                    epoch=epoch,
                    pretrain_epochs=args.pretrain_epochs,
                    ramp_epochs=args.ramp_epochs,
                    phys_weight_ramp=args.phys_weight_ramp,
                    phys_weight_final=args.phys_weight_final,
                ),
                dtype=tf.float32
            )

            epoch_total = 0.0
            epoch_data = 0.0
            epoch_phys = 0.0
            epoch_u = 0.0
            epoch_v = 0.0
            epoch_w = 0.0
            epoch_p = 0.0
            epoch_cont = 0.0
            epoch_mx = 0.0
            epoch_my = 0.0
            epoch_mz = 0.0
            steps = 0

            for X_batch, u_b, v_b, w_b, p_b in dataset:
                (
                    l_total, l_data, l_phys,
                    l_u, l_v, l_w, l_p,
                    l_cont, l_mx, l_my, l_mz
                ) = train_step(X_batch, u_b, v_b, w_b, p_b, phys_weight)

                epoch_total += float(l_total.numpy())
                epoch_data += float(l_data.numpy())
                epoch_phys += float(l_phys.numpy())
                epoch_u += float(l_u.numpy())
                epoch_v += float(l_v.numpy())
                epoch_w += float(l_w.numpy())
                epoch_p += float(l_p.numpy())
                epoch_cont += float(l_cont.numpy())
                epoch_mx += float(l_mx.numpy())
                epoch_my += float(l_my.numpy())
                epoch_mz += float(l_mz.numpy())
                steps += 1

            avg_total = epoch_total / steps
            avg_data = epoch_data / steps
            avg_phys = epoch_phys / steps
            avg_u = epoch_u / steps
            avg_v = epoch_v / steps
            avg_w = epoch_w / steps
            avg_p = epoch_p / steps
            avg_cont = epoch_cont / steps
            avg_mx = epoch_mx / steps
            avg_my = epoch_my / steps
            avg_mz = epoch_mz / steps

            total_hist.append(avg_total)
            data_hist.append(avg_data)
            phys_hist.append(avg_phys)
            phys_weight_hist.append(float(phys_weight.numpy()))

            if (epoch + 1) % args.print_freq == 0:
                current_lr = optimizer.learning_rate.numpy()
                print(
                    f"Epoch {epoch + 1:4d}/{args.epochs} | "
                    f"LR: {current_lr:.6f} | "
                    f"w_phys: {float(phys_weight.numpy()):.6e} | "
                    f"Total: {avg_total:.6f} | "
                    f"Data: {avg_data:.6f} | "
                    f"Phys: {avg_phys:.6f} | "
                    f"u: {avg_u:.6f} | v: {avg_v:.6f} | w: {avg_w:.6f} | p: {avg_p:.6f} | "
                    f"cont: {avg_cont:.6f} | mx: {avg_mx:.6f} | my: {avg_my:.6f} | mz: {avg_mz:.6f}"
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

    # Plot 1: total/data/physics loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(total_hist) + 1), total_hist, label="Total Loss", linewidth=2)
    plt.plot(range(1, len(data_hist) + 1), data_hist, label="Data Loss", linestyle="--")
    plt.plot(range(1, len(phys_hist) + 1), phys_hist, label="Physics Loss", linestyle="-.")
    plt.yscale("log")
    plt.title("PINN v2 Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("training_loss_pinn_v2.png", dpi=300, bbox_inches="tight")
    plt.savefig("training_loss_pinn_v2.eps", format="eps", bbox_inches="tight")
    plt.show()

    # Plot 2: physics weight schedule
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(phys_weight_hist) + 1), phys_weight_hist, linewidth=2)
    plt.yscale("log")
    plt.title("PINN v2 Physics Weight Schedule")
    plt.xlabel("Epochs")
    plt.ylabel("Physics Weight")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("physics_weight_schedule_pinn_v2.png", dpi=300, bbox_inches="tight")
    plt.savefig("physics_weight_schedule_pinn_v2.eps", format="eps", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
