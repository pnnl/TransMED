import argparse
import json
import math
import os
import pickle as pkl

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import (auc, balanced_accuracy_score,
                             precision_recall_curve, roc_auc_score)
from tensorflow.keras.layers import Add, Dense, Embedding, Input, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from benchmarks.keras_transformer import Transformer

PAD, CLS, SEP, MASK = 0, 1, 2, 3


def inv_list(l, start=0):
    return {l[i]:i+start for i in range(len(l))}


def get_data(data_dir, pretrain_infile, finetune_infile, task, aggregation_interval, num_timesteps, lookahead, run, features, outcome_var, force_binary_outcome=False):
    if task=='pretrain':
        task = 'los'
    pdata = pkl.load(open(f'{data_dir}/{pretrain_infile}', 'rb'))
    data, windows = pkl.load(open(f'{data_dir}/{finetune_infile}', 'rb'))

    if force_binary_outcome:
        print(f"Forcing two output classes for outcome: {outcome_var}")
        # Any frequency outcome variable greater than or equal to 1 is set to 1
        windows.loc[windows[outcome_var] >= 1, outcome_var] = 1

    data['codes'] = [[] for i in range(len(data))]
    pdata['codes'] = [[] for i in range(len(pdata))]
    if 'dx' in features:
        data['codes'] = data['codes'] + data['conditions'].apply(lambda l:[str(c) for c in l])
        pdata['codes'] = pdata['codes'] + pdata['conditions'].apply(lambda l:[str(c) for c in l])
    if 'pr' in features:
        data['codes'] = data['codes'] + data['procedures'].apply(lambda l:[str(c) for c in l])
        pdata['codes'] = pdata['codes'] + pdata['procedures'].apply(lambda l:[str(c) for c in l])
    if 'rx' in features:
        data['codes'] = data['codes'] + data['drugs'].apply(lambda l:[str(c) for c in l])
        pdata['codes'] = pdata['codes'] + pdata['drugs'].apply(lambda l:[str(c) for c in l])
    if 'mea' in features:
        data['codes'] = data['codes'] + data['measurements'].apply(lambda l:[str(c) for c in l])
        pdata['codes'] = pdata['codes'] + pdata['measurements'].apply(lambda l:[str(c) for c in l])
    data = data[['pid_vid', 'timestep', 'age', 'codes', 'outcome_'+task]]
    pdata = pdata[['pid_vid', 'timestep', 'age', 'codes']]
    data['age'] = data['age'].apply(lambda x:int(round(x)))
    pdata['age'] = pdata['age'].apply(lambda x:int(round(x)))

    # Convert codes to indices.
    all_codes = sorted(list(set(list(np.concatenate(list(data['codes'])))+list(np.concatenate(list(pdata['codes']))))))
    code2idx = inv_list(all_codes, start=4) # PAD, CLS, SEP, MASK
    data['codes'] = data['codes'].apply(lambda x: [code2idx[c] for c in x])
    pdata['codes'] = pdata['codes'].apply(lambda x: [code2idx[c] for c in x])

    pid_vids = [] # Ns
    input_timesteps = [] # Ns x num_timesteps
    codes = [] # Ns x max_codes
    ages = [] # Ns
    positions = [] # Ns x max_codes
    outcome_los = [] # Ns

    data = data.set_index(['pid_vid', 'timestep'])
    for sample in windows.itertuples():
        outcome_los.append(getattr(sample, 'outcome_'+task))
        pid_vids.append(sample.pid_vid)
        input_timesteps.append(sample.input_timesteps)
        curr_codes = []
        curr_positions = []
        t = 0
        for timestep in sample.input_timesteps:
            row = data.loc[sample.pid_vid, timestep]
            curr_codes += row.codes + [SEP]
            curr_positions += [t]*(len(row.codes) + 1)
            t += 1
        ages.append(row.age)
        codes.append([CLS] + curr_codes[:-1]) # No SEP for last timestep
        positions.append([0] + curr_positions[:-1])

    print ('# samples', len(outcome_los))
    split = json.load(open('patient_splits/finetune_patient_split'+str(run)+'.json', 'r'))
    train_ids, valid_ids, test_ids = split['ft_train'], split['ft_val'], split['ft_test']
    train_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in train_ids])
    valid_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in valid_ids])
    test_idx = np.array([i for i in range(len(pid_vids)) if pid_vids[i] in test_ids])
    print ('# train, valid, test', len(train_idx), len(valid_idx), len(test_idx))

    pcodes = [] # Ns x max_codes
    pages = [] # Ns
    ppositions = [] # Ns x max_codes

    pdata = pdata.sort_values(by=['pid_vid', 'timestep'])
    timesteps = pdata.groupby('pid_vid').agg({'timestep':list}).reset_index()
    pwindows = []
    for row in timesteps.itertuples():
        for i in range(len(row.timestep)-num_timesteps+1):
            pwindows.append([row.pid_vid, row.timestep[i:i+num_timesteps]])
    pwindows = pd.DataFrame(pwindows, columns=['pid_vid', 'input_timesteps'])

    pdata = pdata.set_index(['pid_vid', 'timestep'])
    for sample in tqdm(pwindows.itertuples()):
        curr_codes = []
        curr_positions = []
        t = 0
        for timestep in sample.input_timesteps:
            row = pdata.loc[sample.pid_vid, timestep]
            curr_codes += row.codes + [SEP]
            curr_positions += [t]*(len(row.codes) + 1 )
            t += 1
        pages.append(row.age)
        pcodes.append([CLS] + curr_codes[:-1]) # No SEP for last timestep
        ppositions.append([0] + curr_positions[:-1])

    pats = pwindows.pid_vid.unique()
    np.random.shuffle(pats)
    bp = int(0.9*len(pats))
    ptrain_idx = pats[:bp]
    pvalid_idx = pats[bp:]
    ids = list(pwindows.pid_vid)
    ptrain_idx = np.array([i for i in range(len(ids)) if ids[i] in ptrain_idx])
    pvalid_idx = np.array([i for i in range(len(ids)) if ids[i] in pvalid_idx])
    print ('# ptrain, pvalid', len(ptrain_idx), len(pvalid_idx))

    max_input_len = max([len(x) for x in codes])
    print ('max_input_len', max_input_len)
    return len(all_codes), codes, np.array(ages), positions, np.array(outcome_los), train_idx, valid_idx, test_idx, max_input_len, pid_vids, input_timesteps, pcodes, pages, ppositions, ptrain_idx, pvalid_idx

# Borrowed from https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L741
def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  position = tf.to_float(tf.range(length) + start_index)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      tf.maximum(tf.to_float(num_timescales) - 1, 1))
  tf.range(num_timescales)
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  # Please note that this slightly differs from the published paper.
  # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal

class PosEnc(Layer):
    def __init__(self, max_input_len, d):
        self.max_input_len = max_input_len
        self.d = d
        super(PosEnc, self).__init__()
    def build(self, input_shape):
        self.pe = get_timing_signal_1d(self.max_input_len, self.d)[0, :, :]
        super(PosEnc, self).build(input_shape)
    def call(self, x):
        return tf.nn.embedding_lookup(self.pe, tf.cast(x, tf.int32))
    def compute_output_shape(self, input_shape):
        return input_shape + (self.d,)

def BEHRT(max_input_len, num_codes, d, N, h, dk, dv, dff, dropout):
    cod = Input(shape=(None,))
    age = Input(shape=(1,))
    pos = Input(shape=(None,))
    seg = Input(shape=(None,))
    pt_outputs = Input(shape=(None,))

    cod_emb = Embedding(num_codes+4, d)(cod) # b, T, d
    age_emb = Embedding(99, d)(age) # b, 1, d
    pos_emb = PosEnc(max_input_len, d)(pos)
    seg_emb = Embedding(2, d)(seg) # b, T, d
    comb_emb = Add()([cod_emb, age_emb, pos_emb, seg_emb]) # b, T, d

    mask = Lambda(lambda x:K.clip(x,0,1))(cod)
    cont_emb = Transformer(N, h, dk, dv, dff, dropout)(comb_emb, mask=mask) # b, T, d
    cls_emb = Lambda(lambda x:x[:,0:1,:])(cont_emb) # b, d
    op = Lambda(lambda x:K.sigmoid(K.max(x, axis=-1, keepdims=True)))(cls_emb)

    model = Model([cod, age, pos, seg], op)
    print (model.summary())

    pt_op = Dense(num_codes, activation='softmax')(cont_emb) # b, T, num_codes
    pt_mask = Lambda(lambda x:K.cast(x>=0, 'float32'))(pt_outputs)
    pt_outputs2 = Lambda(lambda x:K.clip(x, 0, num_codes+100))(pt_outputs)
    pt_loss1 = Lambda(lambda x:K.sparse_categorical_crossentropy(x[0], x[1]))([pt_outputs2, pt_op]) # b, T
    pt_loss = Lambda(lambda x:K.sum(pt_mask*pt_loss1, axis=-1))([mask, pt_loss1])

    pt_model = Model([cod, age, pos, seg, pt_outputs], [pt_loss, pt_op])
    return model, pt_model

class Trainer():
    def __init__(self, model, pt_model, codes, ages, positions, outcomes, train_idx, valid_idx, test_idx, pid_vids, input_timesteps, pcodes, pages, ppositions, ptrain_idx, pvalid_idx, batch_size, lr, epochs, patience, weights_path, pweights_path, num_codes):
        self.model, self.pt_model = model, pt_model
        self.codes, self.ages, self.positions, self.outcomes = codes, ages, positions, outcomes
        self.pcodes, self.pages, self.ppositions = pcodes, pages, ppositions
        self.ages = np.expand_dims(self.ages, -1)
        self.pages = np.expand_dims(self.pages, -1)
        self.train_idx, self.valid_idx, self.test_idx = train_idx, valid_idx, test_idx
        self.ptrain_idx, self.pvalid_idx = ptrain_idx, pvalid_idx
        self.pid_vids, self.input_timesteps = pid_vids, input_timesteps
        self.batch_size, self.lr, self.epochs, self.patience = batch_size, lr, epochs, patience
        self.model.compile(optimizer=Adam(lr), loss='binary_crossentropy')
        self.pt_model.compile(optimizer=Adam(lr), loss=self.pt_get_loss())
        self.class_weights = self.compute_class_weights
        self.patience = patience
        self.weights_path = weights_path
        self.num_codes = num_codes

    def compute_class_weights(self):
        N = len(self.outcomes)
        num_pos = self.outcomes.sum()
        num_neg = N - num_pos
        return {0:N/(2*num_neg), 1:N/(2*num_pos)}

    def pt_get_loss(self):
#         def masked_bce(y_true, y_pred): # y_true - b, T; # y_pred - b, T, num_codes
#             mask = K.cast(y_true>=0, 'float32')
#             return K.sum(mask*K.sparse_categorical_crossentropy(y_true, y_pred), axis=-1)
        def identity(y_true, y_pred): # y_true - b, T; # y_pred - b, T, num_codes
            return y_pred
        return identity

    """When training the network and specifcally, the embeddings for the MLM task, we lef 86.5% of the disease
    words unchanged; 12% of the words were replaced with [mask]; and the remaining 1.5% of words, were replaced
    with randomly-chosen disease words."""
    def pt_get_batch_io(self, idx):
        codes = [self.pcodes[i] for i in idx]
        ages = self.pages[idx]
        positions = [self.ppositions[i] for i in idx]
        max_len_for_batch = np.max([len(x) for x in codes])
        codes = np.array([x+[PAD]*(max_len_for_batch-len(x)) for x in codes])
        positions = np.array([x+[PAD]*(max_len_for_batch-len(x)) for x in positions])
        segments = positions%2
        prediction_mask = np.random.choice([0, 1, 2], codes.shape, p=[0.865, 0.12, 0.015]) # Mask tokens. b, T
        prediction_mask *= (codes>=4)  # Unmask special tokens.
        random_codes = np.random.randint(size=codes.shape, low=4, high=self.num_codes+4)
        masked_codes = (prediction_mask==0)*codes + (prediction_mask==1)*MASK + (prediction_mask==2)*random_codes
        outputs = (prediction_mask>=1)*(codes-4) + (prediction_mask==0)*(-1)
        return masked_codes, ages, positions, segments, outputs

    def pretrain_epoch(self):
        np.random.shuffle(self.ptrain_idx)
        epoch_loss = 0
        b = 0
        pbar = tqdm(range(0, len(self.ptrain_idx), self.batch_size))
        for batch_start in pbar:
            batch_end = min(batch_start+self.batch_size, len(ptrain_idx))
            inputs = self.pt_get_batch_io(ptrain_idx[batch_start:batch_end])
            batch_loss = self.pt_model.train_on_batch(inputs, [np.zeros((len(inputs[0]), 1)), np.zeros((len(inputs[0]), 1))])
            epoch_loss += batch_loss[1]
            b += 1
            pbar.set_description('batch_loss: '+str(round(epoch_loss/b, 5)))
        return epoch_loss/b

    def pretrain(self):
        print (self.pt_model.summary())
        wait = self.patience
        best_val_met = -np.inf
        weights_path = self.weights_path.split('_')
        weights_path = weights_path[0]+'_'+weights_path[1]
        for e in range(self.epochs):
            loss = self.pretrain_epoch()
            val_res = self.pt_test(self.pvalid_idx)
            print ('Epoch', e, 'loss', loss, 'val res', val_res)
            if val_res['bal_acc'] > best_val_met:
                best_val_met = val_res['bal_acc']
                wait = self.patience
                self.pt_model.save_weights(pweights_path)
            else:
                wait -= 1
                if wait==0:
                    break
            print ('Wait for', wait, 'more epochs.')
        self.pt_best_val_met = best_val_met
        self.pt_model.load_weights(pweights_path)

    def pt_test(self, idx):
        ypred = []
        ytrue = []
        res = {}
        pbar = tqdm(range(0, len(idx), self.batch_size))
        for batch_start in pbar:
            batch_end = min(batch_start+self.batch_size, len(idx))
            for _ in range(3): # Try 2 masks per sample.
                inputs = self.pt_get_batch_io(idx[batch_start:batch_end])
                curr_ypred = self.pt_model.predict(inputs)[1] # b, T, num_codes
                curr_ypred = np.argmax(curr_ypred, axis=-1).flatten() # b x T
                curr_ytrue = inputs[-1].flatten()
                ii = np.argwhere(curr_ytrue>=0).flatten()
                ypred += list(curr_ypred[ii])
                ytrue += list(curr_ytrue[ii])
        ypred = np.array(ypred)
        ytrue = np.array(ytrue)
        res['bal_acc'] = balanced_accuracy_score(ytrue, ypred)
        return res

    def get_batch_io(self, idx):
        codes = [self.codes[i] for i in idx]
        ages = self.ages[idx]
        positions = [self.positions[i] for i in idx]
        outcomes = self.outcomes[idx]
        max_len_for_batch = np.max([len(x) for x in codes])
        codes = np.array([x+[PAD]*(max_len_for_batch-len(x)) for x in codes])
        positions = np.array([x+[PAD]*(max_len_for_batch-len(x)) for x in positions])
        segments = positions%2
        return [codes, ages, positions, segments], outcomes

    def train_epoch(self):
        np.random.shuffle(self.train_idx)
        epoch_loss = 0
        b = 0
        pbar = tqdm(range(0, len(self.train_idx), self.batch_size))
        for batch_start in pbar:
            batch_end = min(batch_start+self.batch_size, len(train_idx))
            inputs, output = self.get_batch_io(train_idx[batch_start:batch_end])
            batch_loss = self.model.train_on_batch(inputs, output, class_weight=self.class_weights)
            epoch_loss += batch_loss
            b += 1
            pbar.set_description('batch_loss: '+str(round(epoch_loss/b, 5)))
        return epoch_loss/b

    def train(self, pretrain):
        if pretrain:
            self.pt_model.load_weights(pweights_path)
        wait = self.patience
        best_val_met = -np.inf
        for e in range(self.epochs):
            loss = self.train_epoch()
            valid_res = self.test(self.valid_idx)
            print ('Epoch', e, 'loss', loss, 'valid_res:', valid_res)
            if valid_res['best_f1']>best_val_met:
                best_val_met = valid_res['best_f1']
                wait = self.patience
                self.model.save_weights(self.weights_path)
            else:
                wait -= 1
                if wait==0:
                    break
            print ('Wait for', wait, 'more epochs.')
        self.model.load_weights(self.weights_path)
        self.best_valid_met = best_val_met
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
            ypred += list(self.model.predict(inputs).flatten())
        ypred = np.array(ypred)
        ytrue = np.array(ytrue)
        res['auroc'] = roc_auc_score(ytrue, ypred)
        precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
        res['auprc'] = auc(recall, precision)
        res['loss'] = np.mean(K.eval(K.binary_crossentropy(K.constant(ytrue), K.constant(ypred))))
        ind = np.argwhere(recall+precision).flatten()
        all_f1s = 2*recall*precision/(recall+precision)
        best_bp_ind = np.nanargmax(all_f1s)
        res['best_f1'] = np.nanmax(all_f1s)
        res['cm'] = sklearn.metrics.confusion_matrix(ytrue, ypred>=thresholds[best_bp_ind])
        return res

    def write_scores(self, idx):
        ypred = []
        ytrue = []
        pbar = tqdm(range(0, len(idx), self.batch_size))
        for batch_start in pbar:
            batch_end = min(batch_start+self.batch_size, len(idx))
            inputs, output = self.get_batch_io(idx[batch_start:batch_end])
            ytrue += list(output)
            ypred += list(self.model.predict(inputs).flatten())
        pid_vids = [self.pid_vids[i] for i in idx]
        ts = [self.input_timesteps[i] for i in idx]
        json_path = self.weights_path[:-2].replace('weights', 'scores')+'json'
        res = {}
        num_timesteps = len(ts[0])
        for i in range(len(idx)):
            key = ';'.join([pid_vids[i]+';'+str(ts[i][t]) for t in range(num_timesteps)])+';'
            res[key] = [-1, int(ytrue[i]), float(ypred[i])]
        with open(json_path, 'w') as f:
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
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--pretrain_infile', type=str)
    parser.add_argument('--finetune_infile',type=str)
    parser.add_argument('--task', default='los', type=str)
    parser.add_argument('--features', default='dx_pr_rx_mea', type=str)
    parser.add_argument('--aggregation_interval', default=24, type=int)
    parser.add_argument('--num_timesteps', default=2, type=int)
    parser.add_argument('--lookahead', default=3, type=int)
    parser.add_argument('--run', default=9, type=int)

    parser.add_argument('--d', default=64, type=int)
    parser.add_argument('--N', default=4, type=int)
    parser.add_argument('--h', default=4, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--pretrain', default=False, type=str2bool)
    parser.add_argument('--outcome_var', type=str, default='outcome_mace')
    parser.add_argument('--force_binary_outcome', action='store_true', type=bool)
    args = parser.parse_args()

    np.random.seed(0)
    num_codes, codes, ages, positions, outcome_los, train_idx, valid_idx, test_idx, max_input_len, pid_vids, input_timesteps, pcodes, pages, ppositions, ptrain_idx, pvalid_idx = \
        get_data(args.data_dir, args.pretrain_infile, args.finetune_infile, args.task, args.aggregation_interval, args.num_timesteps, args.lookahead, args.run, args.features, args.outcome_var, args.force_binary_outcome)

    for task in ['ventilation']:
        for lookahead in [7]:
            for features in ['pr', 'mea']:
                            args.task = task
                            args.lookahead = lookahead
                            args.features = features
                            args.dk = args.d // args.h
                            args.dv = args.d // args.h
                            args.dff = args.d*2
                            print (task, args.lookahead, features)

                            model, pt_model = BEHRT(max_input_len, num_codes, args.d, args.N, args.h, args.dk, args.dv, args.dff, args.dropout)

                            include_in_path = ['run', 'features', 'N', 'h', 'd', 'pretrain', 'num_timesteps', 'lookahead']
                            weights_path = 'behrt_results/'+args.task
                            for a in include_in_path:
                                weights_path += '_'+a+'_'+str(getattr(args, a))
                                weights_path += '.h5'
                            include_in_path = ['features', 'N', 'h', 'd', 'num_timesteps']
                            pweights_path = 'behrt_results/pretraining'
                            for a in include_in_path:
                                pweights_path += '_'+a+'_'+str(getattr(args, a))
                                pweights_path += '.h5'

                            trainer = Trainer(model, pt_model, codes, ages, positions, outcome_los, train_idx, valid_idx, test_idx, pid_vids, input_timesteps, pcodes, pages, ppositions, ptrain_idx, pvalid_idx, args.batch_size, args.lr, args.epochs, args.patience, weights_path, pweights_path, num_codes)
                            if args.task=='pretrain':
                                trainer.pretrain()
                            else:
                                trainer.train(args.pretrain)

                            # Log the input arguments and result.
                            logfile = 'behrt_results/log.csv'
                            args_dict = vars(args)
                            if not(os.path.exists(logfile)):
                                with open(logfile, 'w') as f:
                                    f.write(','.join(list(args_dict.keys())) + ',test_auroc,test_auprc,test_best_f1,test_cm,valid_auroc,valid_auprc,valid_best_f1,valid_cm,valid_pt_metrics\n')

                            if args.task!='pretrain':
                                for m in ['auroc', 'auprc', 'best_f1']:
                                    args_dict['test_'+m] = trainer.test_res[m]
                                    args_dict['valid_'+m] = trainer.valid_res[m]
                                test_cm = trainer.test_res['cm']
                                args_dict['test_cm'] = str(test_cm[0,0])+' '+str(test_cm[0,1])+' '+str(test_cm[1,0])+' '+str(test_cm[1,1])
                                test_cm = trainer.valid_res['cm']
                                args_dict['valid_cm'] = str(test_cm[0,0])+' '+str(test_cm[0,1])+' '+str(test_cm[1,0])+' '+str(test_cm[1,1])
                            else:
                                args_dict['valid_pt_metric'] = trainer.pt_best_val_met
                            log = pd.read_csv(logfile)
                            log = pd.concat([log, pd.DataFrame({k:[v] for k,v in args_dict.items()})])
                            log.to_csv(logfile, index=False)
