import argparse
import pickle as pkl
import sys

import pandas as pd


def load_data(path):
    data = pd.read_pickle(
        path
    )
    return data


def get_windows_mimic(row):
    windows = []
    ts = row.timestep
    num_admissions = len(ts)
    window = ts[0:2]

    windows.append(window)
    return windows


def create_windows(data, outcome_cols, window_method):
    global timesteps
    # Output a dataframe with one row per sample.
    # Columns: pid_vid, input_timesteps
    timesteps = data.groupby("pid_vid").agg({"timestep": list}).reset_index()
    timesteps["timestep"] = timesteps["timestep"].apply(lambda ts: sorted(ts))

    if window_method == 'all':
        timesteps["input_timesteps"] = timesteps.apply(get_windows_mimic, axis=1)
    elif window_method == 'first':
        # FIXME
        timesteps["input_timesteps"] = timesteps.apply(get_windows_mimic, axis=1)
    else:
        print(f"Unhandled window_method: {window_method}, exiting...")
        sys.exit(1)

    rows = []
    for row in timesteps.itertuples():
        rows += [[row.pid_vid, w] for w in row.input_timesteps]
    timesteps = pd.DataFrame(rows, columns=["pid_vid", "input_timesteps"])

    timesteps["timestep"] = timesteps["input_timesteps"].apply(max)
    merge_columns = ["pid_vid", "timestep"]
    merge_columns.extend(outcome_cols)
    timesteps = timesteps.merge(
        data[merge_columns],
        on=["pid_vid", "timestep"],
        how="left",
    )
    timesteps.drop(columns=["timestep"], inplace=True)
    return timesteps


def cast_codes_to_str(codes):
    str_codes = [str(code) for code in codes]
    return str_codes


def main(args):
    task = args.task
    lookahead = args.lookahead
    num_timesteps = args.num_timesteps
    outcome_var = args.outcome_var
    base_path = args.base_path
    data_dir = args.data_dir
    window_method = args.window_method

    data_path = f"{data_dir}/{base_path}.pkl"
    print(f"Loading data from: {data_path}")
    data = load_data(data_path)
    # cast all codes to strings
    data.conditions = data.conditions.apply(cast_codes_to_str)
    data.drugs = data.drugs.apply(cast_codes_to_str)
    data.procedures = data.procedures.apply(cast_codes_to_str)
    data.measurements = data.measurements.apply(cast_codes_to_str)
    # might need to handle data.measurement_values in the future

    if task == 'readmission':
        task_data = data
        if 'mimic' in base_path:
            outcome_columns = ["outcome_readmission"]
        else:
            outcome_columns = ["outcome_readmission"]
    else:
        print(f"Task: {task} not handled, exiting...")
        sys.exit(1)

    print(f"task_data.shape: {task_data.shape}", flush=True)
    print(f"task_data.pid_vid.nunique(): {task_data.pid_vid.nunique()}")
    print(f"Num cases: {task_data[task_data[outcome_var] == True].pid_vid.nunique()}")
    print(f"Num controls: {task_data[task_data[outcome_var] == False].pid_vid.nunique()}")


    # drop patients that only have 1 visit
    # patients with 2 visits are controls for no readmission
    # patients with 3 or more visists are cases for readmission
    admissions_per_patient = task_data.groupby("pid_vid").size()
    patients_to_drop = admissions_per_patient[admissions_per_patient < 2].index.tolist()
    items_to_drop = task_data[task_data.pid_vid.isin(patients_to_drop)].index
    print(f"Dropping {len(items_to_drop)} patients with only 1 visit")
    print(f"task_data.shape: {task_data.shape}, num_patients: {task_data.pid_vid.nunique()}")
    task_data.drop(items_to_drop, inplace=True)
    print(f"task_data.shape: {task_data.shape}, num_patients: {task_data.pid_vid.nunique()}")

    task_windows = create_windows(task_data, outcome_columns, window_method)
    num_samples = len(task_windows)
    #pos_outcome = 100 * task_windows[outcome_var].sum() / len(task_windows)
    #pos_outcome = 100 *
    num_patients = task_windows.pid_vid.nunique()
    num_cases = task_windows[task_windows[outcome_var] == True].pid_vid.nunique()
    num_controls = task_windows[task_windows[outcome_var] == False].pid_vid.nunique()
    pos_outcome = 100 * num_cases / num_patients

    print(f"# samples: {num_samples}")
    print(f"lookahead: {lookahead}")
    print(f"num_timesteps: {num_timesteps}")
    print(f"num_patients: {num_patients}")
    print(f"num_cases: {num_cases}")
    print(f"num_controls: {num_controls}")
    print(f"% pos outcome: {pos_outcome}")
    # aggregation interval is fixed for now
    #output_path = f"{data_dir}/{task}_{base_path}_aggr24_lookahead{lookahead}_numts{num_timesteps}.pkl"
    output_path = f"{data_dir}/{task}_{base_path}_lookahead{lookahead}_numts{num_timesteps}.pkl"
    print(f"Dumping finetuning data to {output_path}")
    with open(output_path, 'wb') as f:
        pkl.dump([task_data, task_windows], f)


    # load pickle for testing the file was generated properly
    with open(output_path, 'rb') as f:
        data, windows = pkl.load(f)
    print(f"data.shape: {data.shape}, windows.shape: {windows.shape}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="readmission", type=str)
    parser.add_argument("--outcome_var", default="outcome_readmission")
    parser.add_argument("--num_timesteps", default=2, type=int, help='Fixed history length')
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--window_method", default="first", type=str, help='all | first | last')
    parser.add_argument("--lookahead", default=1, type=int)
    args = parser.parse_args()
    print(f"args: {args}")
    main(args)
