import pickle as pkl
import pickle5
import sys

import numpy as np
import pandas as pd


def load_dataframe(data_dir, infile, model, features, lab_codes):
    if "pretrain" in model:
        pretrain_data_path = f"{data_dir}/{infile}"
        print(f"Loading pretraining data from {pretrain_data_path}")
        data = pd.read_pickle(pretrain_data_path)
        windows = None
        try:
            data = pd.read_pickle(pretrain_data_path)
        except ValueError:
            data = pickle5.load(open(pretrain_data_path, 'rb'))

    elif "finetune" in model:
        finetune_data_path = f"{data_dir}/{infile}"
        print(f"Loading finetuning data from {finetune_data_path}")
        with open(finetune_data_path, 'rb') as f:
            data, windows = pkl.load(f)
    else:
        print(f"Unhandled model: {model}")
        print("Accepted model options: ['pretrain', 'finetune']")
        print("Exiting...")
        sys.exit(1)

    print("loaded input dataframes")

    data['codes'] = [[] for i in range(len(data))]
    if 'dx' in features:
        data['codes'] = data['codes'] + data['conditions']
    if 'pr' in features:
        data['codes'] = data['codes'] + data['procedures']
    if 'rx' in features:
        data['codes'] = data['codes'] + data['drugs']
    if 'mea' in features:
        data['codes'] = data['codes'] + data['measurements']

    data.drop(columns=['conditions', 'procedures', 'drugs', 'measurements'])
    data['codes'] = data['codes'].apply(lambda x: [str(c) for c in x])

    lab_code_to_ind = {lab_codes[i]: i for i in range(len(lab_codes))}
    data['measurement_values'] = data['measurement_values'].apply(lambda d: {k: v for (k, v) in d.items() if k in lab_codes})

    def mea_vec(d):
        vec = np.array([np.nan for i in range(len(lab_codes))])
        for k, v in d.items():
            vec[lab_code_to_ind[k]] = v
        return vec
    data['measurement_values'] = data['measurement_values'].apply(mea_vec)
    lab_means = np.nanmean(np.array(list(data['measurement_values'])), axis=0)
    print(f'labs codes {lab_codes}')
    print(f'lab means {lab_means}')
    data['measurement_values'] = data['measurement_values'].apply(lambda x: np.nan_to_num(x - lab_means))
    lab_stds = np.std(np.array(list(data['measurement_values'])), axis=0)
    print('lab stds', lab_stds)
    data['measurement_values'] = data['measurement_values'].apply(lambda x: x / lab_stds)

    if 'pretrain' in model:
        data = data.loc[data.codes.apply(len) >= 3]
    return data, windows


def get_lab_codes_for_dataset(infile):
    lab_codes = []
    if 'covid' in infile:
        lab_codes = ['3013650', '3037511', '3010457', '3004327', '3006923', '3020460']
    return lab_codes
