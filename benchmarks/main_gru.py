import argparse
import json
import os
import pickle as pkl
from functools import partialmethod
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from tensorflow.keras.layers import GRU, Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def inv_list(l, start=0):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d

def get_data(task, aggregation_interval, num_timesteps, lookahead, run, features, data_dir, infile, base_path, outcome_var, force_binary_outcome=False):
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
    #data = data[['pid_vid', 'timestep','codes', 'outcome_'+task, 'sex', 'race', 'ethnicity', 'age', 'risk_factors', 'measurement_values']]
    data = data[['pid_vid', 'timestep','codes', outcome_var, 'sex', 'race', 'ethnicity', 'age', 'measurement_values']]

    # Convert codes to multihot vectors.
    all_codes = sorted(list(set(list(np.concatenate(list(data['codes']))))))
    code2idx = inv_list(all_codes, start=0)
    data['codes'] = data['codes'].apply(lambda x: [code2idx[c] for c in x])
    def multihot(l):
        l2 = [0 for i in range(len(code2idx))]
        for cind in l:
            l2[cind] = 1
        return l2
    data['codes'] = data['codes'].apply(multihot)

    # Convert sex to binary, race and ethnicity to one hot.
    #data['sex'] = data['sex'].map({'MALE':0, 'FEMALE':1})
    #data['sex'] = data['sex'].astype(int)
    data['sex'] = data['sex'].map({'M':0, 'F':1})
    data = pd.get_dummies(data, columns=['race', 'ethnicity'], dummy_na=False)

    # Create deomgraphics vector.
    demo_columns = [col for col in data.columns if (('sex' in col) or ('age' in col) or ('race' in col) or ('ethnicity' in col))]
    data['demo'] = data[demo_columns].agg(np.array, axis=1)
    #data = data[['pid_vid', 'timestep', 'demo', 'codes', 'risk_factors', 'measurement_values', 'outcome_'+task]]
    data = data[['pid_vid', 'timestep', 'demo', 'codes', 'measurement_values', outcome_var]]

    # Create labs vector.
    lab_codes = ['3013650', '3037511', '3010457', '3004327', '3006923', '3020460']
    lab_code_to_ind = {lab_codes[i]:i for i in range(len(lab_codes))}
    data['measurement_values'] = data['measurement_values'].apply(lambda d:{k:v for (k,v) in d.items() if k in lab_codes})
    def mea_vec(d):
        vec = np.array([np.nan for i in range(len(lab_codes))])
        for k,v in d.items():
            vec[lab_code_to_ind[k]] = v
        return vec
    data['measurement_values'] = data['measurement_values'].apply(mea_vec)
    lab_means = np.nanmean(np.array(list(data['measurement_values'])), axis=0)
    print ('labs codes', lab_codes)
    print ('lab means', lab_means)
    data['measurement_values'] = data['measurement_values'].apply(lambda x:np.nan_to_num(x-lab_means))
    lab_stds = np.std(np.array(list(data['measurement_values'])), axis=0)
    print ('lab stds', lab_stds)
    data['measurement_values'] = data['measurement_values'].apply(lambda x:x/lab_stds)

    pid_vids = [] # Ns
    input_timesteps = [] # Ns x num_timesteps
    codes = [] # Ns x num_timesteps x codes
    demos = [] # Ns x demo_length
    rfs = [] # Ns x num_rfs
    labs = [] # Ns x num_timesteps x num_labs
    outcomes = [] # Ns

    data = data.set_index(['pid_vid', 'timestep'])
    for sample in windows.itertuples():
        outcomes.append(getattr(sample, outcome_var))
        pid_vids.append(sample.pid_vid)
        input_timesteps.append(sample.input_timesteps)
        curr_codes = []
        curr_labs = []
        for timestep in sample.input_timesteps:
            row = data.loc[sample.pid_vid, timestep]
            curr_codes.append(list(row.codes))
            curr_labs.append(list(row.measurement_values))
        codes.append(curr_codes)
        labs.append(curr_labs)
        demos.append(list(row.demo))
        #rfs.append(list(row.risk_factors))
        rfs.append(np.empty(6))

    codes = np.array(codes) # N x T x num_codes
    labs = np.array(labs) # N x T x num_labs
    demos = np.array(demos) # N x num_demo_var
    rfs = np.array(rfs) # N x num_rf

    print ('# samples', len(outcomes))
    split = json.load(open(f'patient_splits/{base_path}/finetune_patient_split_run{run}.json', 'r'))
    train_ids, valid_ids, test_ids = split['ft_train'], split['ft_val'], split['ft_test']
    train_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in train_ids])
    valid_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in valid_ids])
    test_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in test_ids])
    print ('# train, valid, test', len(train_idx), len(valid_idx), len(test_idx))

    max_input_len = max([len(x) for x in codes])
    print ('max_input_len', max_input_len)
    return codes, labs, demos, rfs, np.array(outcomes), train_idx, valid_idx, test_idx, pid_vids, input_timesteps

def MyGRU(num_timesteps, input_dim, hidden_dim, recurrent_dropout, bi, features, num_codes, num_labs, num_demo, num_rf):
    codes = Input(shape=(num_timesteps, num_codes))
    labs = Input(shape=(num_timesteps, num_labs))
    demo = Input(shape=(num_demo,))
    rf = Input(shape=(num_rf,))

    if ('dx' in features) or ('rx' in features) or ('pr' in features) or ('mea' in features):
        if 'val' in features:
            gru_inputs = Concatenate(axis=-1)([codes, labs]) # b, num_timesteps, num_codes+num_labs
        else:
            gru_inputs = codes # b, num_timesteps, num_codes
    elif 'val' in features:
        gru_inputs = labs # b, num_timesteps, num_labs
    gru_inputs = Dense(input_dim, activation='relu')(gru_inputs) # b, num_timesteps, input_dim

    if ('demo' in features) and ('nlp' in features):
        initial_state = Concatenate(axis=-1)([demo, rf])
        initial_state = Dense(hidden_dim)(initial_state)
    elif 'demo' in features:
        initial_state = demo
        initial_state = Dense(hidden_dim)(initial_state)
    elif 'nlp' in features:
        initial_state = rf
        initial_state = Dense(hidden_dim)(initial_state)
    else:
        initial_state=None

    gru_output_forward = GRU(units=hidden_dim, recurrent_dropout=recurrent_dropout)(gru_inputs, initial_state=initial_state)

    if bi:
        gru_output_backward = GRU(units=hidden_dim, recurrent_dropout=recurrent_dropout, go_backwards=True)(gru_inputs, initial_state=initial_state)
        gru_outputs = Concatenate(axis=1)([gru_output_forward, gru_output_backward])
    else:
        gru_outputs = gru_output_forward # b, hidden_dim

    output = Dense(2, activation='softmax')(gru_outputs) # b, 2
    model = Model([codes, labs, demo, rf], output)
    print (model.summary())
    return model

class Trainer():
    def __init__(self, model, codes, labs, demos, rfs, outcomes, train_idx, valid_idx, test_idx, pid_vids, input_timesteps, batch_size, lr, epochs, patience, patience_threshold, weights_path, scores_path):
        self.model = model
        self.codes, self.labs, self.demos, self.rfs, self.outcomes = codes, labs, demos, rfs, outcomes
        self.train_idx, self.valid_idx, self.test_idx = train_idx, valid_idx, test_idx
        self.pid_vids, self.input_timesteps = pid_vids, input_timesteps
        self.batch_size, self.lr, self.epochs, self.patience = batch_size, lr, epochs, patience
        self.model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy')
        self.class_weights = self.compute_class_weights
        self.patience = patience
        self.weights_path = weights_path
        self.scores_path = scores_path
        self.patience_threshold = patience_threshold

    def compute_class_weights(self):
        N = len(self.outcomes)
        num_pos = self.outcomes.sum()
        num_neg = N - num_pos
        return {0:N/(2*num_neg), 1:N/(2*num_pos)}

    def get_batch_io(self, idx):
        codes = self.codes[idx]
        labs = self.labs[idx]
        demo = self.demos[idx]
        rf = self.rfs[idx]
        outcomes = self.outcomes[idx]
        return [codes, labs, demo, rf], outcomes
        #return [codes, labs, demo], outcomes

    def train_epoch(self):
        np.random.shuffle(self.train_idx)
        epoch_loss = 0
        b = 0
        pbar = tqdm(range(0, len(self.train_idx), self.batch_size))
        for batch_start in pbar:
            batch_end = min(batch_start+self.batch_size, len(train_idx))
            inputs, output = self.get_batch_io(train_idx[batch_start:batch_end])
            class_weights = self.class_weights()
            # Had to pull class_weights before train_on_batch call due to error
            #batch_loss = self.model.train_on_batch(inputs, output, class_weight=self.class_weights)
            batch_loss = self.model.train_on_batch(inputs, output, class_weight=class_weights)
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
            print(f"Epoch: {e}, loss: {loss}, valid_res: {valid_res}")
            # if valid_res['f1']>best_val_met:
            #     best_val_met = valid_res['f1']
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
        self.test_res = self.test(self.test_idx)
        print ('Test res:', self.test_res)
        self.write_scores(self.test_idx)

    def test(self, idx):
        ypred = []
        ytrue = []
        res = {}
        pbar = tqdm(range(0, len(idx), self.batch_size))
        for batch_start in pbar:
            batch_end = min(batch_start+self.batch_size, len(idx))
            inputs, output = self.get_batch_io(idx[batch_start:batch_end])
            ytrue += list(output)
            ypred += list(self.model.predict(inputs))

        ypred = np.array(ypred)
        ytrue = np.array(ytrue)
        res['auroc'] = roc_auc_score(ytrue, ypred[:,1])
        precision, recall, thresholds = precision_recall_curve(ytrue, ypred[:,1])
        res['auprc'] = auc(recall, precision)

        ypred_cls = (ypred[:,1]>ypred[:,0]).astype(int)
        res['f1'] = f1_score(ytrue, ypred_cls, average='binary')
        res['cm'] = sklearn.metrics.confusion_matrix(ytrue, ypred_cls)
        res['precision'] = sklearn.metrics.precision_score(ytrue, ypred_cls)
        res['recall'] = sklearn.metrics.recall_score(ytrue, ypred_cls)
        return res

    def write_scores(self, idx):
        ypred = []
        ytrue = []
        pbar = tqdm(range(0, len(idx), self.batch_size))
        for batch_start in pbar:
            batch_end = min(batch_start+self.batch_size, len(idx))
            inputs, output = self.get_batch_io(idx[batch_start:batch_end])
            ytrue += list(output)
            ypred += list(self.model.predict(inputs))
        ypred = np.array(ypred)
        ypred_cls = (ypred[:,1]>ypred[:,0]).astype(int)
        pid_vids = [self.pid_vids[i] for i in idx]
        ts = [self.input_timesteps[i] for i in idx]
        #json_path = self.weights_path[:-2].replace('weights', 'scores').replace('gru_results', 'baselines_results/gru')+'json'
        res = {}
        num_timesteps = len(ts[0])
        for i in range(len(idx)):
            #key = ';'.join([pid_vids[i]+';'+str(ts[i][t]) for t in range(num_timesteps)])+';'
            key = ';'.join([f"{pid_vids[i]};{ts[i][t]}" for t in range(num_timesteps)])+';'
            res[key] = [int(ypred_cls[i]), int(ytrue[i]), float(ypred[i, 1])]
        with open(self.scores_path, 'w') as f:
            json.dump(res, f)

if __name__=='__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='los', type=str)
    parser.add_argument('--features', default='dx_pr_rx_mea_val_demo_nlp', type=str)
    parser.add_argument('--aggregation_interval', default=24, type=int)
    parser.add_argument('--num_timesteps', default=2, type=int)
    parser.add_argument('--lookahead', default=2, type=int)
    parser.add_argument('--run', default=9, type=int)

    parser.add_argument('--input_dim', default=32, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--recurrent_dropout', default=0.3, type=float)
    parser.add_argument('--bi', default=True, type=str2bool)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--patience', default=10, type=int)

    # VA specific arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--patience_threshold', default=0.0001, type=float, help='Minimum relative loss decrease')
    # new arguments
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--outcome_var', type=str, default='outcome_mace')
    parser.add_argument('--force_binary_outcome', action='store_true')
    args = parser.parse_args()
    print(f"args: {args}")


    # for num_timesteps in [2]: # 248
    #     for lookahead in [7, 3]: # 237
    #         for task in ['los', 'ventilation']:
    #             for bi in [False]:
    #                 for features in ['dx', 'rx', 'pr', 'dx_rx_pr']:
    #num_timesteps = args.num_timesteps
    #lookahead = args.lookahead
    #task = args.task
    #bi = args.bi
    print ('='*100)
    #print (num_timesteps, lookahead, task, bi, features)
    print(f"{args.num_timesteps}, {args.lookahead}, {args.task}, {args.bi}, {args.features}")
    #args.task = task
    #args.bi = bi
    #args.features = features
    #args.num_timesteps = num_timesteps
    #args.lookahead = lookahead

    np.random.seed(args.random_seed)
    # make sure gru_results directory is made
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    # make sure baselines_results/gru directory is made
    #Path('baselines_results/gru').mkdir(parents=True, exist_ok=True)


    codes, labs, demos, rfs, outcomes, train_idx, valid_idx, test_idx, pid_vids, input_timesteps = \
        get_data(args.task, args.aggregation_interval, args.num_timesteps, args.lookahead, args.run, args.features, args.data_dir, args.infile, args.base_path, args.outcome_var, args.force_binary_outcome)

    model = MyGRU(args.num_timesteps, args.input_dim, args.hidden_dim, args.recurrent_dropout, args.bi, args.features, len(codes[0,0]), len(labs[0,0]), len(demos[0]), len(rfs[0]))

    include_in_path = ['run', 'features', 'input_dim', 'hidden_dim', 'recurrent_dropout', 'bi', 'num_timesteps', 'lookahead']
    #weights_path = 'gru_results/'+args.task
    unique_base_path = f'{args.task}'
    for a in include_in_path:
        unique_base_path += '_'+a+'_'+str(getattr(args, a))
    weights_path = f"{args.results_dir}/{unique_base_path}.h5"
    scores_path = f"{args.results_dir}/{unique_base_path}.json"

    trainer = Trainer(model, codes, labs, demos, rfs, outcomes, train_idx, valid_idx, test_idx, pid_vids, input_timesteps, args.batch_size, args.lr, args.epochs, args.patience, args.patience_threshold, weights_path, scores_path)
    trainer.train()

    # Log the input arguments and result.
    logfile = f'{args.results_dir}/log.csv'
    args_dict = vars(args)
    if not(os.path.exists(logfile)):
        with open(logfile, 'w') as f:
            #f.write( ','.join(list(args_dict.keys()))+',test_auroc,test_auprc,test_f1,test_cm,valid_auroc,valid_auprc,valid_f1,valid_cm\n')
            f.write( ','.join(list(args_dict.keys()))+',test_auroc,test_auprc,test_f1,test_precision,test_recall,test_cm,valid_auroc,valid_auprc,valid_f1,valid_precision,valid_recall,valid_cm\n')

    #for m in ['auroc', 'auprc', 'f1']:
    for m in ['auroc', 'auprc', 'f1', 'precision', 'recall']:
        args_dict['test_'+m] = trainer.test_res[m]
        args_dict['valid_'+m] = trainer.valid_res[m]
    test_cm = trainer.test_res['cm']
    args_dict['test_cm'] = str(test_cm[0,0])+' '+str(test_cm[0,1])+' '+str(test_cm[1,0])+' '+str(test_cm[1,1])
    test_cm = trainer.valid_res['cm']
    args_dict['valid_cm'] = str(test_cm[0,0])+' '+str(test_cm[0,1])+' '+str(test_cm[1,0])+' '+str(test_cm[1,1])

    log = pd.read_csv(logfile)
    log = pd.concat([log, pd.DataFrame({k:[v] for k,v in args_dict.items()})])
    log.to_csv(logfile, index=False)
