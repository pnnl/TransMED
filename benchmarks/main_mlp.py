import argparse
import json
import os
import pickle as pkl
import sys
from functools import partialmethod
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import tensorflow.keras.backend as K
from gensim.models.poincare import PoincareKeyedVectors
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def load_dynamic_features(path):
    print(f"Loading dynamic features from {path}")
    with open(path) as f:
        dynamic_features = f.read().splitlines()
    print(f"Loaded {len(dynamic_features)} dynamic features from {path}")
    return dynamic_features


def load_poincare_embeddings(path):
    poincare_embeddings = PoincareKeyedVectors.load_word2vec_format(path, binary=True)
    return poincare_embeddings



def inv_list(l, start=0):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d


def one_hot_codes(x, all_codes):
    o = [0 for i in range(len(all_codes))]
    for c in x:
        assert (c==int(c))
        o[int(c)] = 1
    return o

def poincare_codes(x, all_codes, poincare_embeddings, poincare_embed_dim, idx2code):
    # set up empty vector to fill
    o = [np.zeros(poincare_embed_dim) for i in range(len(all_codes))]
    for c in x:
        try:
            o[int(c)] = poincare_embeddings[idx2code[c]]
        except KeyError:
            print(f"Code not present in poincare_embeddings: {idx2code[c]}")
    return o


def mea_vec(l, all_mea):
    d = {k[0]:[] for k in l}
    for k in l:
        d[k[0]].append(k[1])
    d = {k:np.mean(v) for k,v in d.items()}
    o = [np.nan for i in range(len(all_mea))]
    for k,v in d.items():
        o[k] = v
    return o


def get_inputs_one_hot(all_codes, all_mea, features, demo, risk_factors, codes, mea_values):
    inputs = []
    for i in tqdm(range(len(demo))):
        #inputs.append(list(demo[i])+one_hot_codes(codes[i])+mea_vec(mea_values[i]))
        curr_list = []
        if 'demo' in features:
            curr_list += list(demo[i])
        if 'risk_factors' in features or 'nlp' in features:
            curr_list += list(risk_factors[i])
        if ('dx' in features) or ('rx' in features) or ('pr' in features) or ('mea' in features):
            curr_list += one_hot_codes(codes[i], all_codes)
        if 'val' in features:
            curr_list += mea_vec(mea_values[i], all_mea)
        inputs.append(curr_list)

    inputs = np.array(inputs)
    return inputs

def get_inputs_poincare(all_codes, all_mea, features, demo, risk_factors, codes, mea_values, poincare_embeddings, poincare_embed_dim, idx2code):
    inputs = []
    for i in tqdm(range(len(demo))):
        curr_list = []
        # We will handle demo the same as for one_hot because they don't exist in poincare embeddings
        if 'demo' in features:
            demo_items = list(demo[i])
            demo_array = []
            for di, demo_item in enumerate(demo_items):
                item_val = np.zeros(poincare_embed_dim)
                # handling age
                if di == 1:
                    if demo_item > (poincare_embed_dim - 1):
                        demo_item = poincare_embed_dim - 1
                    item_val[demo_item] = 1
                # Handling bools
                else:
                    if demo_item == 1:
                        item_val = np.ones(poincare_embed_dim)
                demo_array.append(item_val)
            curr_list += demo_array
        # We will handle risk_factors the same as for one_hot because they don't exist in poincare embeddings
        if 'risk_factors' in features or 'nlp' in features:
            curr_list += list(risk_factors[i])
        # We will handle features differently
        if ('dx' in features) or ('rx' in features) or ('pr' in features) or ('mea' in features):
            #curr_list += one_hot_codes(codes[i], all_codes)
            curr_list += poincare_codes(codes[i], all_codes, poincare_embeddings, poincare_embed_dim, idx2code)
        if 'val' in features:
            # We don't have measurement values with the feature_exposure dataset so we will treat the same as one_hot
            curr_list += mea_vec(mea_values[i], all_mea)

        # flatten list
        curr_array = np.asarray(curr_list)
        curr_list_flat = curr_array.flatten()
        inputs.append(curr_list_flat)

    inputs = np.array(inputs)
    return inputs


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
    idx2code =  {v: k for k, v in code2idx.items()}
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



    if args.embedding_method == 'poincare':
        print("Using poincare embeddings")
        poincare_embeddings = load_poincare_embeddings(args.poincare_path)
        pe_set = set(poincare_embeddings.index_to_key)
        dynamic_features = load_dynamic_features(args.dynamic_features_path)
        df_set = set(dynamic_features)
        set_diff = df_set.difference(pe_set)
        print(f"set_diff: {set_diff}")
        inputs = get_inputs_poincare(all_codes, all_mea, features, demo, risk_factors, codes, mea_values, poincare_embeddings, args.poincare_embed_dim, idx2code)
    elif args.embedding_method == 'one_hot':
        print("Using one hot embeddings")
        inputs = get_inputs_one_hot(all_codes, all_mea, features, demo, risk_factors, codes, mea_values)
    else:
        print(f"Unhandled embedding method {args.embedding_method}, exiting...")
        sys.exit(0)

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

    with open(f'patient_splits/{base_path}/finetune_patient_split_run{run}.json', 'r') as f:
        split = json.load(f)

    train_ids, valid_ids, test_ids = split['ft_train'], split['ft_val'], split['ft_test']
    train_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in train_ids])
    valid_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in valid_ids])
    test_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in test_ids])
    print ('# train, valid, test', len(train_idx), len(valid_idx), len(test_idx))

    return inputs, np.array(outcomes), train_idx, valid_idx, test_idx, pid_vids, input_timesteps


def LogisticRegression(input_shape):
    inp = Input(shape=input_shape)
    op = Dense(1, activation='sigmoid')(inp)
    model = Model(inp, op)
    print (model.summary())
    return model


class Trainer():
    def __init__(self, model, inputs, outcomes, train_idx, valid_idx, test_idx, pid_vids, input_timesteps, batch_size, lr, epochs, patience, patience_threshold, weights_path):
        self.model = model
        self.inputs, self.outcomes = inputs, outcomes
        self.train_idx, self.valid_idx, self.test_idx = train_idx, valid_idx, test_idx
        self.pid_vids, self.input_timesteps = pid_vids, input_timesteps
        self.batch_size, self.lr, self.epochs, self.patience = batch_size, lr, epochs, patience
        self.model.compile(optimizer=Adam(lr), loss='binary_crossentropy')
        self.class_weights = self.compute_class_weights()
        self.patience = patience
        self.weights_path = weights_path
        self.last_threshold = 0.5
        self.patience_threshold = patience_threshold

    def compute_class_weights(self):
        N = len(self.outcomes)
        num_pos = self.outcomes.sum()
        num_neg = N - num_pos
        return {0:N/(2*num_neg), 1:N/(2*num_pos)}

    def train_epoch(self):
        np.random.shuffle(self.train_idx)
        epoch_loss = 0
        b = 0
        pbar = tqdm(range(0, len(self.train_idx), self.batch_size))
        for batch_start in pbar:
            batch_end = min(batch_start+self.batch_size, len(self.train_idx))
            ind = self.train_idx[batch_start:batch_end]
            batch_loss = self.model.train_on_batch(self.inputs[ind],
                                                   self.outcomes[ind],
                                                   class_weight=self.class_weights)
            epoch_loss += batch_loss
            b += 1
            pbar.set_description('batch_loss: '+str(round(epoch_loss/b, 5)))
        return epoch_loss/b

    def train(self):
        wait = self.patience
        #best_val_met = -np.inf
        best_loss = np.inf
        for e in range(self.epochs):
            loss = self.train_epoch()
            valid_res = self.test(self.valid_idx)
            print(f"\nEpoch: {e}, loss: {loss}, valid_res: {valid_res}")
            # if valid_res['best_f1']>best_val_met:
            #     best_val_met = valid_res['best_f1']
            #     wait = self.patience
            #     self.model.save_weights(self.weights_path)
            # else:
            #     wait -= 1
            #     if wait==0:
            #         break
            best_loss_to_beat = best_loss * (1 - self.patience_threshold)
            if loss < best_loss_to_beat:
                best_loss = loss
                wait = self.patience
                self.model.save_weights(self.weights_path)
            else:
                wait -= 1
                if wait==0:
                    break
            print ('Wait for', wait, 'more epochs.')
        self.model.load_weights(self.weights_path)
        self.valid_res = self.test(self.valid_idx)
        self.test_res = self.test(self.test_idx, final_test=True)
        print ('Test res:', self.test_res)
        self.write_scores(self.test_idx)

    def test(self, idx, final_test=False):
        res = {}
        ypred = self.model.predict(self.inputs[idx]).flatten()
        ytrue = self.outcomes[idx]
        res['auroc'] = roc_auc_score(ytrue, ypred)
        precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
        res['auprc'] = auc(recall, precision)
        res['loss'] = np.mean(K.eval(K.binary_crossentropy(K.constant(ytrue), K.constant(ypred))))
        #ind = np.argwhere(recall+precision).flatten()
        #all_f1s = 2*recall*precision/(recall+precision)
        # best_bp_ind = np.nanargmax(all_f1s)
        # res['best_f1'] = np.nanmax(all_f1s)
        # best_f1_threshold = thresholds[best_bp_ind]

        cmh_all_f1s = []
        for threshold in thresholds:
            item_f1 = sklearn.metrics.f1_score(ytrue, ypred>=threshold, average='binary')
            cmh_all_f1s.append(item_f1)
        best_bp_ind = np.nanargmax(cmh_all_f1s)
        best_f1_threshold = thresholds[best_bp_ind]

        # If we are performing the final test, use the last validation threshold
        if final_test:
            threshold = self.last_threshold
            res['best_f1'] = sklearn.metrics.f1_score(ytrue, ypred>=threshold, average='binary')
        # If train or validation, use the best f1 threshold and save
        else:
            threshold = best_f1_threshold
            self.last_threshold = threshold
            res['best_f1'] = np.nanmax(cmh_all_f1s)

        res['cm'] = sklearn.metrics.confusion_matrix(ytrue, ypred>=threshold)
        res['precision'] = sklearn.metrics.precision_score(ytrue, ypred>=threshold)
        res['recall'] = sklearn.metrics.recall_score(ytrue, ypred>=threshold)
        return res

    def write_scores(self, idx):
        #y_scores = self.model.predict(self.inputs[idx]).flatten()
        y_scores = self.model.predict(self.inputs[idx]).flatten()
        #y_true = self.outcomes[idx]
        y_true = self.outcomes[idx]
        y_pred = (y_scores>=self.last_threshold).astype(int)
        pid_vids = [self.pid_vids[i] for i in idx]
        ts = [self.input_timesteps[i] for i in idx]
        json_path = self.weights_path[:-2].replace('weights', 'scores') + 'json'
        res = {}
        num_timesteps = len(ts[0])
        for i in range(len(idx)):
            #key = ';'.join([pid_vids[i] + ';' + str(ts[i][t]) for t in range(num_timesteps)]) + ';'
            key = ';'.join([f"{pid_vids[i]};{ts[i][t]}" for t in range(num_timesteps)]) + ';'
            #old_format = [-1, int(y_true[i]), float(y_scores[i])]
            res[key] = [int(y_pred[i]), int(y_true[i]), float(y_scores[i])]
        #print(res)
        with open(json_path, 'w') as f:
            json.dump(res, f)


def main(args):
    all_res = []
    print('='*100)
    print(args.num_timesteps, args.lookahead, args.task, args.features)

    np.random.seed(args.random_seed)
    inputs, outcomes, train_idx, valid_idx, test_idx, pid_vids, input_timesteps = \
        get_data(args.aggregation_interval, args.num_timesteps, args.lookahead, args.run, args.features, args.data_dir, args.infile, args.task, args.base_path, args.outcome_var, args.force_binary_outcome)
    # FIX how this is calculated
    num_features = len(inputs[0])
    if args.embedding_method == 'poincare':
        #input_shape = (num_features, args.poincare_embed_dim)
        input_shape = (num_features,)
    else:
        input_shape = (num_features,)
    print(f'input_shape: {input_shape}')
    model = LogisticRegression(input_shape)
    include_in_path = ['run', 'features', 'lr', 'batch_size', 'num_timesteps', 'lookahead']
    # make sure results directory is made
    print(f"Make sure results directory exists: {args.results_dir}")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    weights_path = f'{args.results_dir}/{args.task}'
    for a in include_in_path:
        weights_path += f'_{a}_{getattr(args, a)}'
    weights_path += '.h5'
    trainer = Trainer(model, inputs, outcomes, train_idx, valid_idx, test_idx, pid_vids, input_timesteps, args.batch_size, args.lr, args.epochs, args.patience, args.patience_threshold, weights_path)
    trainer.train()
    #all_res.append(trainer.test_res)

    # Log the input arguments and results.
    logfile = f'{args.results_dir}/log.csv'
    args_dict = vars(args)
    args_dict['num_input_dim'] = len(inputs[0])
    if not(os.path.exists(logfile)):
        with open(logfile, 'w') as f:
            #f.write(','.join(list(args_dict.keys())) + ',test_auroc,test_auprc,test_best_f1,test_cm,valid_auroc,valid_auprc,valid_best_f1,valid_cm\n')
            f.write(','.join(list(args_dict.keys())) + ',test_auroc,test_auprc,test_best_f1,test_precision,test_recall,test_cm,valid_auroc,valid_auprc,valid_best_f1,valid_precision,valid_recall,valid_cm\n')


    #for m in ['auroc', 'auprc', 'best_f1']:
    for m in ['auroc', 'auprc', 'best_f1', 'precision', 'recall']:
        #vals = [res[m] for res in all_res]
        #print (m, str(round(np.mean(vals), 3))+', '+str(round(np.std(vals), 3)))
        args_dict[f'test_{m}'] = trainer.test_res[m]
        args_dict[f'valid_{m}'] = trainer.valid_res[m]

    test_cm = trainer.test_res['cm']
    args_dict['test_cm'] = f"{test_cm[0,0]} {test_cm[0,1]} {test_cm[1,0]} {test_cm[1,1]}"

    test_cm = trainer.valid_res['cm']
    args_dict['valid_cm'] = f"{test_cm[0,0]} {test_cm[0,1]} {test_cm[1,0]} {test_cm[1,1]}"

    log = pd.read_csv(logfile)
    log = pd.concat([log, pd.DataFrame({k: [v] for k,v in args_dict.items()})])
    log.to_csv(logfile, index=False)

    #print ('# input dimensions', len(inputs[0]))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='dx_pr_rx_mea_demo', type=str)
    parser.add_argument('--aggregation_interval', default=24, type=int)
    parser.add_argument('--num_timesteps', default=8, type=int)
    parser.add_argument('--lookahead', default=1, type=int)
    parser.add_argument('--run', default=9, type=int)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--outcome_var', type=str, default='outcome_mace')
    parser.add_argument('--force_binary_outcome', action='store_true')

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--task', type=str, default='mace')
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--patience_threshold', default=0.0001, type=float, help='Minimum relative loss decrease')

    # poincare
    parser.add_argument('--embedding_method', type=str, default='one_hot')
    parser.add_argument('--poincare_path', type=str)
    parser.add_argument('--poincare_embed_dim', type=int, default=100)
    parser.add_argument('--dynamic_features_path', type=str)

    parser.add_argument('--random_seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
