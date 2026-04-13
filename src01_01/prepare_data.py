from __future__ import annotations

import argparse

from pinn_utils import combine_cases, load_training_arrays


def parse_args():
    parser = argparse.ArgumentParser(description="Combine multiple CFD cases into one PINN training CSV.")
    parser.add_argument("--file-100", required=True, help="CSV for the 100%% fan-speed case")
    parser.add_argument("--file-060", required=True, help="CSV for the 60%% fan-speed case")
    parser.add_argument("--file-030", required=True, help="CSV for the 30%% fan-speed case")
    parser.add_argument("--speed-100", type=float, default=1.0)
    parser.add_argument("--speed-060", type=float, default=0.6)
    parser.add_argument("--speed-030", type=float, default=0.3)
    parser.add_argument("--output", default="combined_pinn_data.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    case_map = {
        args.file_100: args.speed_100,
        args.file_060: args.speed_060,
        args.file_030: args.speed_030,
    }
    df = combine_cases(case_map, args.output)

    arrays = load_training_arrays(args.output)
    print("--- Combined data ready ---")
    print(f"Saved CSV: {args.output}")
    print(f"Rows: {len(df):,}")
    print(f"Input matrix shape: {arrays['X_norm'].shape}")
    print(f"Output shape (u/v/w/p each): {arrays['U_raw'].shape}")
    print(f"X_min: {arrays['X_min']}")
    print(f"X_range: {arrays['X_range']}")


if __name__ == "__main__":
    main()
