import json
import sys
from collections import OrderedDict
from pathlib import Path


def generate_patient_split_path(ft_base_path, run, patient_splits_dir="patient_splits", ft_splits_base_path="finetune_patient_split"):
    if not ft_base_path:
        print(f"ft_base_path is required for patient splits path. Exiting...")
        sys.exit(1)
    if not run:
        print(f"run is required for patient splits path. Exiting...")
        sys.exit(1)
    patient_split_path = f"{patient_splits_dir}/{ft_base_path}/{ft_splits_base_path}_run{run}.json"
    return patient_split_path


def load_patient_split(path):
    print(f"Loading patient split from {path}")
    try:
        with open(path, 'r') as f:
            patient_split = json.load(f)
    except FileNotFoundError:
        print(f"No patient split file at this path, returning empty")
        patient_split = None
    return patient_split


def generate_patient_split(ft_base_path, run, rep_patient_ids, val_split, test_split, patient_splits_dir="patient_splits", ft_splits_base_path="finetune_patient_split"):
    train_split = 1.0 - (test_split + val_split)

    run_patient_splits_path = generate_patient_split_path(ft_base_path, run, patient_splits_dir=patient_splits_dir, ft_splits_base_path=ft_splits_base_path)

    print(f"patient_splits_dir: {patient_splits_dir}")
    print(f"run_patient_splits_path: {run_patient_splits_path}")

    print(f"Generating patient split for run: {run}", flush=True)
    unique_patientids = list(OrderedDict.fromkeys(rep_patient_ids))
    num_total_patients = len(unique_patientids)
    num_train_patients = int(num_total_patients * train_split)
    num_val_patients = int(num_total_patients * val_split)
    num_test_patients = int(num_total_patients * test_split)
    train_patient_ids = unique_patientids[:num_train_patients]
    val_patient_ids = unique_patientids[
        num_train_patients : num_train_patients + num_val_patients
    ]
    test_patient_ids = unique_patientids[num_train_patients + num_val_patients :]

    # Make sure patient splits directory exists for finetuning base path
    Path(f"{patient_splits_dir}/{ft_base_path}").mkdir(parents=True, exist_ok=True)
    patient_split = {
        "ft_train": train_patient_ids,
        "ft_val": val_patient_ids,
        "ft_test": test_patient_ids,
    }
    print(f"Saving patient split to {run_patient_splits_path}")
    with open(run_patient_splits_path, 'w') as f:
        json.dump(patient_split, f)
    return patient_split
