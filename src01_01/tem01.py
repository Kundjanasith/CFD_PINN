# prepare_data.py
# Combines multiple CFD CSV files into one training file and normalizes input features.

# train_pinn.py
# Trains the Fourier-feature PINN model and saves weights plus normalization metadata.

# evaluate_nodes.py
# Compares CFD node-by-node velocity magnitude against PINN predictions for one fan speed.

# compare_plane.py
# Compares CFD vs PINN on a ZX plane at a target Y coordinate and saves side-by-side figures.

# validate_sensors.py
# Evaluates three sensor locations against training-domain CFD data or an unseen CFD file.

# pinn_utils.py
# Shared utility functions used by the other scripts.


python evaluate_nodes.py \
  --data ../Train/combined_pinn_data.csv \
  --weights ../src01/models/model_checkpoint_epoch1.weights.h5 \
  --norm normalization_stats.npz \
  --fan-speed 1.0

# python evaluate_nodes.py \
#   --data combined_pinn_data.csv \
#   --weights parametrized_pinn_model.weights.h5 \
#   --norm normalization_stats.npz \
#   --fan-speed 1.0

# import os 
# for i in range(1,101):
#     os.system(f"python evaluate_nodes.py --data combined_pinn_data.csv --weights parametrized_pinn_model.weights.h5 --norm normalization_stats.npz --fan-speed {i/100:.2f}")