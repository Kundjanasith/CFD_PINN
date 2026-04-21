# PINN-CFD Python Scripts

This folder is a cleaned Python conversion of the uploaded notebook `PINN-CFD-AT.ipynb`.

## Files

- `prepare_data.py`  
  Combines multiple CFD CSV files into one training file and normalizes input features.

- `train_pinn.py`  
  Trains the Fourier-feature PINN model and saves weights plus normalization metadata.

- `evaluate_nodes.py`  
  Compares CFD node-by-node velocity magnitude against PINN predictions for one fan speed.

- `compare_plane.py`  
  Compares CFD vs PINN on a ZX plane at a target Y coordinate and saves side-by-side figures.

- `validate_sensors.py`  
  Evaluates three sensor locations against training-domain CFD data or an unseen CFD file.

- `pinn_utils.py`  
  Shared utility functions used by the other scripts.

## Typical workflow

1. Prepare combined data
2. Train model
3. Run one or more evaluation scripts

## Example

```bash
python prepare_data.py \
  --file-100 fluent_data_100.csv \
  --file-060 fluent_data_060.csv \
  --file-030 fluent_data_030.csv \
  --output combined_pinn_data.csv

python3 train_pinn.py \
  --data ../Train/combined_pinn_data.csv \
  --epochs 100 \
  --batch-size 32 \
  --weights parametrized_pinn_model.weights.h5 \
  --norm normalization_stats.npz

python3 evaluate_nodes.py \
  --data ../Train/combined_pinn_data.csv \
  --weights parametrized_pinn_model.weights.h5 \
  --norm normalization_stats.npz \
  --fan-speed 1.0
```

## Notes

- The notebook mixed fan-speed conventions such as `1.0`, `0.6`, `0.3` and also `80.0` in some cells.  
  The converted scripts keep fan speed explicit through command-line arguments so you can choose the scale you actually use.
- The original notebook depended on variables defined in earlier cells. These scripts remove that dependency by loading what each script needs explicitly.

python prepare_data.py --file-100 fluent_data_100.csv --file-060 fluent_data_060.csv --file-030 fluent_data_030.csv

python train_pinn.py --data combined_pinn_data.csv --epochs 3000

python evaluate_nodes.py --data combined_pinn_data.csv --fan-speed 1.0

python compare_plane.py --data ../Train/combined_pinn_data.csv --target-y 0.4554 --fan-speed 1.0

python validate_sensors.py --data ../Train/combined_pinn_data.csv --fan-speed 1.0

python validate_sensors.py --weights parametrized_pinn_model.weights.h5 --norm normalization_stats.npz --fan-speed 0.8 --unseen-cfd ansys_results_fan80.csv