"""
Wrapper for pt and ft data generation from mimic-III
"""
import argparse

from create_mimic_pt_dataset import create_mimic_pt
from create_mimic_ft_dataset import create_mimic_ft


def main(args):
    print(f"Running create_mimic_dataset.py to generate pretraining and finetuning data for MIMIC-III")
    # Create pretraining data
    create_mimic_pt(args.random_seed, args.data_dir, args.base_path, args.num_patients, args.extension)
    # Create finetuning data
    create_mimic_ft(args.data_dir, args.base_path, args.task, args.outcome_var, args.num_timesteps, args.lookahead, args.window_method)
    print("Done converting MIMIC-III to pretraining and finetuning datasets")


if __name__ == "__main__":
    # shared args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Input and output data directory", required=True)
    parser.add_argument(
        "--base_path",
        type=str,
        default="mimic",
        help="Base path for naming output files",
    )
    # pt only args
    parser.add_argument(
        "--num_patients", type=int, help="Only process this many patients then break"
    )
    parser.add_argument("--random_seed", default=3, help="Consistent seed for RNG")
    parser.add_argument("--extension", type=str, help='File extension: csv | pkl', default='csv')
    # ft only args
    parser.add_argument("--task", default="readmission", type=str)
    parser.add_argument("--outcome_var", default="outcome_readmission")
    parser.add_argument("--num_timesteps", default=2, type=int, help='Fixed history length')
    parser.add_argument("--lookahead", default=1, type=int)
    parser.add_argument("--window_method", default="first", type=str, help='all | first | last')
    args = parser.parse_args()
    print(f"{args}", flush=True)
    main(args)
