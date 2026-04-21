from __future__ import annotations

import argparse
import time
import os
import tqdm
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import matplotlib.pyplot as plt
import tensorflow as tf

from pinn_utils import load_training_arrays, save_normalization_stats, FEATURE_COLUMNS, TARGET_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description="Train a traditional Neural Network model for CFD prediction.")
    parser.add_argument("--data", default="../Train/combined_pinn_data.csv")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--weights", default="traditional_nn_model.weights.h5")
    parser.add_argument("--norm", default="normalization_nn_stats.npz")
    parser.add_argument("--initial-lr", type=float, default=0.002)
    parser.add_argument("--decay-steps", type=int, default=50000)
    parser.add_argument("--decay-rate", type=float, default=0.90)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    return parser.parse_args()


def build_nn_model(hidden_sizes=(128, 128, 128, 128, 64)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(len(FEATURE_COLUMNS),), name="inputs")
    x = inputs
    for units in hidden_sizes:
        x = tf.keras.layers.Dense(units, activation="swish")(x)
    outputs = tf.keras.layers.Dense(len(TARGET_COLUMNS), name="outputs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="traditional_nn")


def main():
    args = parse_args()
    arrays = load_training_arrays(args.data)

    X_train_norm = arrays["X_norm"]
    Y_train_raw = tf.concat([arrays["U_raw"], arrays["V_raw"], arrays["W_raw"], arrays["P_raw"]], axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((X_train_norm, Y_train_raw))
    dataset = dataset.shuffle(buffer_size=50000).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE)

    model = build_nn_model()
    _ = model(tf.zeros((1, len(FEATURE_COLUMNS)), dtype=tf.float32))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.initial_lr,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=False,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=args.clipnorm)

    @tf.function
    def train_step(X_batch_norm, Y_true):
        with tf.GradientTape() as tape:
            preds = model(X_batch_norm)
            loss = tf.reduce_mean(tf.square(Y_true - preds)) # Standard MSE loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    total_hist = []
    start_time = time.time()

    try:
        for epoch in tqdm.tqdm(range(args.epochs)):
            epoch_loss_total = 0.0
            steps = 0

            for X_batch, Y_batch in dataset:
                l_total = train_step(X_batch, Y_batch)
                epoch_loss_total += float(l_total.numpy())
                steps += 1

            avg_total = epoch_loss_total / steps
            total_hist.append(avg_total)

            if (epoch + 1) % args.print_freq == 0:
                current_lr = optimizer.learning_rate.numpy()
                print(
                    f"Epoch {epoch + 1:4d}/{args.epochs} | LR: {current_lr:.6f} | "
                    f"Total Loss: {avg_total:.6f}"
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
    plt.yscale("log")
    plt.title("Traditional NN Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    fig_base = f"training_loss_traditional_nn_epoch{len(total_hist)}"
    plt.savefig(f"{fig_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_base}.eps", format="eps", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()


# I have created two new Python scripts:


#    1. train_nn.py: This script implements and trains a traditional Neural Network (your "original" method) to predict velocity and pressure based on the
#       input coordinates and fan speed, without incorporating physics constraints.
#    2. compare_models.py: This script will load both the trained traditional NN and your existing PINN model, evaluate them, and provide a comparison,
#       including sensor-specific validation and a bar chart visualization.


#   To proceed with the comparison, please follow these steps:

#   Step 1: Train the Traditional Neural Network


#   Run the train_nn.py script to train the traditional NN. This will create a traditional_nn_model.weights.h5 file and update normalization_stats.npz.



#    1 python3 train_nn.py --epochs 3000 --batch-size 1024 --initial-lr 0.002


#   Step 2: Ensure the PINN Model is Trained

#   You should already have a trained PINN model (parametrized_pinn_model.weights.h5) from previous runs. If not, you would need to train it using
#   train_pinn.py:



#    1 python3 train_pinn.py --epochs 3000 --batch-size 1024 --initial-lr 0.002


#   Step 3: Run the Model Comparison


#   Once both models are trained (or their weights files exist), run the compare_models.py script:


#    1 python3 compare_models.py --fan-speed 1.0


#   This will print a table comparing the sensor validation results for both models and generate a bar chart (model_comparison_sensor_fan1.0.png and .eps)
#   visualizing the velocity predictions at sensor locations.


#   Please let me know when you have completed these steps, and I can help interpret the results or make further adjustments.
