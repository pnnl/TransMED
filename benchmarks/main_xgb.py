import argparse
import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from tqdm import tqdm
from xgboost import XGBClassifier


def inv_list(l, start=0):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d


def get_data(aggregation_interval, num_timesteps, lookahead, run, features, data_dir, infile, task, base_path, outcome_var, force_binary_outcome=False):
    global data, codes, mea_values, demo , inputs, all_mea
    ft_data_path = f"{data_dir}/{infile}"
    print(f"Loading data from: {ft_data_path}")
    with open(ft_data_path, 'rb') as f :
        data, windows = pkl.load(f)

    if force_binary_outcome:
        print(f"Forcing two output classes for outcome: {outcome_var}")
        # Any frequency outcome variable greater than or equal to 1 is set to 1
        windows.loc[windows[outcome_var] >= 1, outcome_var] = 1

    data['codes'] = [[] for i in range(len(data))]
    if 'dx' in features:
        data['codes'] = data['codes'] + data['conditions']
    if 'pr' in features:
        data['codes'] = data['codes'] + data['procedures']
    if 'rx' in features:
        data['codes'] = data['codes'] + data['drugs']
    if 'mea' in features:
        data['codes'] = data['codes'] + data['measurements']
    data = data[['pid_vid', 'timestep', 'sex', 'race', 'ethnicity', 'age', 'codes',
                 'measurement_values', outcome_var]]

    # Convert codes to indices.
    all_codes = sorted(list(set(list(np.concatenate(list(data['codes']))))))
    code2idx = inv_list(all_codes, start=0)
    data['codes'] = data['codes'].apply(lambda x: [code2idx[c] for c in x])

    # Convert measurements to indices.
    all_mea = sorted(list(set(list(np.concatenate(list(
        data['measurement_values'].apply(lambda x:list(x.keys()))))))))
    mea2idx = inv_list(all_mea, start=0)
    data['measurement_values'] = data['measurement_values'].apply(
                                lambda x: {mea2idx[k]:v for k,v in x.items()})
    data['measurement_values'] = data['measurement_values'].apply(lambda x:list(x.items()))

    # Convert sex to binary, race and ethnicity to one hot.
    #data['sex'] = data['sex'].map({'MALE':0, 'FEMALE':1})
    #data['sex'] = data['sex'].astype(int)
    data['sex'] = data['sex'].map({'M':0, 'F':1})
    data = pd.get_dummies(data, columns=['race', 'ethnicity'], dummy_na=False)

    # Create deomgraphics vector.
    demo_columns = [col for col in data.columns if (('sex' in col) or ('age' in col) or ('race' in col) or ('ethnicity' in col))]
    data['demo'] = data[demo_columns].agg(np.array, axis=1)
    data = data[['pid_vid', 'timestep', 'demo', 'codes', 'measurement_values', outcome_var]]

    pid_vids = [] # Ns
    input_timesteps = [] # Ns x num_timesteps
    demo = []
    risk_factors = []
    codes = []
    mea_values = []
    outcomes = [] # Ns

    data = data.set_index(['pid_vid', 'timestep'])
    for sample in windows.itertuples():
        outcomes.append(getattr(sample, outcome_var))
        pid_vids.append(sample.pid_vid)
        input_timesteps.append(sample.input_timesteps)
        curr_codes = []
        curr_mea_values = []
        for timestep in sample.input_timesteps:
            row = data.loc[sample.pid_vid, timestep]
            curr_codes += row.codes
            curr_mea_values += row.measurement_values
        codes.append(curr_codes)
        mea_values.append(curr_mea_values)
        demo.append(row.demo)
        #risk_factors.append(row.risk_factors)

    def one_hot_codes(x):
        o = [0 for i in range(len(all_codes))]
        for c in x:
            assert (c==int(c))
            o[int(c)] = 1
        return o
    def mea_vec(l):
        d = {k[0]:[] for k in l}
        for k in l:
            d[k[0]].append(k[1])
        d = {k:np.mean(v) for k,v in d.items()}
        o = [np.nan for i in range(len(all_mea))]
        for k,v in d.items():
            o[k] = v
        return o
    inputs = []
    for i in tqdm(range(len(demo))):
        #inputs.append(list(demo[i])+one_hot_codes(codes[i])+mea_vec(mea_values[i]))
        curr_list = []
        if 'demo' in features:
            curr_list += list(demo[i])
        if 'risk_factors' in features or 'nlp' in features:
            curr_list += list(risk_factors[i])
        if ('dx' in features) or ('rx' in features) or ('pr' in features) or ('mea' in features):
            curr_list += one_hot_codes(codes[i])
        if 'val' in features:
            curr_list += mea_vec(mea_values[i])
        inputs.append(curr_list)

    inputs = np.array(inputs)

    # # Mean fill missing measurements.
    # start = len(inputs[0]) - len(all_mea)
    # for k in range(start, len(inputs[0])):
    #     mean_value = np.nanmean(inputs[:,k])
    #     inputs[:,k] = np.nan_to_num(inputs[:,k], mean_value)

    # if 'demo' not in features:
    #     inputs = inputs[:, len(demo[0]):]
    # if 'val' not in features:
    #     inputs = inputs[:, :len(inputs[0])-len(all_mea)]
    if 'val' in features:
        start = len(inputs[0]) - len(all_mea)
        for k in range(start, len(inputs[0])):
            mean_value = np.nanmean(inputs[:,k])
            inputs[:,k] = np.nan_to_num(inputs[:,k], mean_value)

    # Normalize inputs.
    means = inputs.mean(axis=0, keepdims=True)
    stds = inputs.std(axis=0, keepdims=True)
    mask = (stds==0)
    stds = 1*mask + (1-mask)*stds
    inputs = (inputs-means)/stds

    print ('# samples', len(outcomes))
    print ('input dim', len(inputs[0]))

    patient_splits_path = f'patient_splits/{base_path}/finetune_patient_split_run{run}.json'
    split = json.load(open(patient_splits_path, 'r'))
    train_ids, valid_ids, test_ids = split['ft_train'], split['ft_val'], split['ft_test']
    train_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in train_ids])
    valid_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in valid_ids])
    test_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in test_ids])
    print ('# train, valid, test', len(train_idx), len(valid_idx), len(test_idx))

    return inputs, np.array(outcomes), train_idx, valid_idx, test_idx, pid_vids, input_timesteps


def get_results(y_true, y_pred):
    results = {}
    results['auroc'] = roc_auc_score(y_true, y_pred[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred[:, 1])
    results['auprc'] = auc(recall, precision)
    y_pred_cls = (y_pred[:,1]>y_pred[:,0]).astype(int)
    results['binary-f1'] = f1_score(y_true, y_pred_cls, average='binary')
    results['micro-f1'] = f1_score(y_true, y_pred_cls, average='micro')
    results['macro-f1'] = f1_score(y_true, y_pred_cls, average='macro')
    results['cm'] = confusion_matrix(y_true, y_pred_cls)
    results['precision_score'] = precision_score(y_true, y_pred_cls)
    results['recall_score'] = recall_score(y_true, y_pred_cls)
    return results


def write_scores(test_idx, pid_vids, input_timesteps, y_pred, y_true, scores_path):
    y_pred_cls = (y_pred[:,1]>y_pred[:,0]).astype(int)
    test_pid_vids = [pid_vids[i] for i in test_idx]
    test_input_timesteps = [input_timesteps[i] for i in test_idx]
    res = {}
    num_timesteps = len(test_input_timesteps[0])
    for i in range(len(test_idx)):
        #key = ';'.join([pid_vids[i]+';'+str(ts[i][t]) for t in range(num_timesteps)])+';'
        key = ';'.join([f"{test_pid_vids[i]};{test_input_timesteps[i][t]}" for t in range(num_timesteps)])+';'
        res[key] = [int(y_pred_cls[i]), int(y_true[i]), float(y_pred[i, 1])]
    print(f"Dumping patient scores dict to {scores_path}")
    with open(scores_path, 'w') as f:
        json.dump(res, f)

def main(args):

    np.random.seed(args.random_seed)
    inputs, outcomes, train_idx, valid_idx, test_idx, pid_vids, input_timesteps = \
        get_data(args.aggregation_interval, args.num_timesteps, args.lookahead, args.run, args.features, args.data_dir, args.infile, args.task, args.base_path, args.outcome_var, args.force_binary_outcome)

    # fit on training data, we aren't using the validation set for this model
    print("Fitting XGBClasifier from the training data set")
    clf = XGBClassifier(random_state=0).fit(inputs[train_idx], outcomes[train_idx])

    print("Testing classifier on the test set")
    score = clf.score(inputs[test_idx], outcomes[test_idx])
    y_pred = clf.predict_proba(inputs[test_idx])
    y_true = outcomes[test_idx]

    print("Generating unique base path based on argument values")
    include_in_path = ['run', 'features', 'num_timesteps', 'lookahead']
    unique_base_path = f'{args.task}'
    for a in include_in_path:
        unique_base_path += '_'+a+'_'+str(getattr(args, a))
    print(f"unique_base_path: {unique_base_path}")
    scores_path = f"{args.results_dir}/{unique_base_path}.json"

    print(f"Making sure results_dir {args.results_dir} exists")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    write_scores(test_idx, pid_vids, input_timesteps, y_pred, y_true, scores_path)

    # metrics
    results = get_results(y_true, y_pred)

    print(f"results: {results}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='dx_pr_rx_mea_demo', type=str)
    parser.add_argument('--aggregation_interval', default=24, type=int)
    parser.add_argument('--num_timesteps', default=8, type=int)
    parser.add_argument('--lookahead', default=1, type=int)
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--task', type=str, default='mace')
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--outcome_var', type=str, default='outcome_mace')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--force_binary_outcome', action='store_true')
    args = parser.parse_args()
    main(args)
