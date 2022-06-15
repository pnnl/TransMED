import json
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

#from src.clinical_groups import ClinicalGroupMapper
from src.demo_mapper import DemoMapper, DemoMapper_MACE, DemoMapper_MIMIC
from src.ssl import (PatientFineTuneBERTModel, PatientFineTuneGRUModel,
                     PatientFineTuneModel)


class FineTuneData:
    def __init__(self, patient_events, patient_outcomes, patient_event_attention_masks, static_embeddings, labvalues, masked_posn, position_ids, patient_id_with_timestep, num_time_steps):

        self.x = patient_events
        self.y = patient_outcomes
        # self.patient_ids = patient_ids
        self.mask = patient_event_attention_masks
        self.static_embeddings = static_embeddings
        self.labvalues = labvalues

        # generate fake mask and get position tensor for each patient event
        self.masked_posn = masked_posn
        self.position_ids = position_ids
        self.patient_id_with_timestep = patient_id_with_timestep
        self.num_time_steps = num_time_steps
        self.shape = len(self.x[0])
        # if sampling is not None:
        #    self.sample(sampling, sampling_ratio)

    def copy(self):
        newobj = FineTuneData(self.x, self.y, self.mask,  self.static_embeddings,  self.labvalues,
              self.masked_posn,  self.position_ids,  self.patient_id_with_timestep,  self.num_time_steps)
        return newobj

    def __len__(self):
        return len(self.x[0])

    def sample(self, sampling, sampling_ratio):
        print(f"Sampling:{sampling}", flush=True)
        num_pats = len(self.x) // self.num_time_steps
        y_np = np.squeeze(self.y.numpy())
        outcomes = [y_np[i * self.num_time_steps] for i in range(num_pats)]
        c = Counter(outcomes)
        majority_class = max(c, key=lambda x: c[x])
        minority_class = min(c, key=lambda x: c[x])
        majority_indices = [i for i, v in enumerate(outcomes) if v == majority_class]
        minority_indices = [i for i, v in enumerate(outcomes) if v == minority_class]
        all_indices = list(range(len(outcomes)))
        if len(c) != 2:
            print(f"Cannot Sample since outcomes have {c} distinct values", flush=True)
        elif sampling == "undersampling":
            chosen_indices = random.sample(
                majority_indices, k=int(sampling_ratio * c[minority_class])
            )
            all_indices = chosen_indices + minority_indices
        elif sampling == "oversampling":
            extra_count_required = (
                int(c[majority_class] / sampling_ratio) - c[minority_class]
            )
            if extra_count_required <= 0:
                print(
                    "no oversampling required, majority/minority < sampling_ratio",
                    flush=True,
                )
            else:
                extra_indices = random.choices(minority_indices, k=extra_count_required)
                all_indices += extra_indices
        tensor_indices = sum(
            [
                list(range(i * self.num_time_steps, (i + 1) * self.num_time_steps))
                for i in all_indices
            ],
            [],
        )
        self.choose_indices(tensor_indices)

    def choose_indices(self, indices):
        self.x = self.x[indices]
        self.y = self.y[indices]
        self.mask = self.mask[indices]
        self.masked_posn = self.masked_posn[indices]
        self.position_ids = self.position_ids[indices]
        self.static_embeddings = self.static_embeddings[indices]
        self.patient_id_with_timestep = self.patient_id_with_timestep[indices]

    def size(self):
        return self.x.size(0)

    def resize(self, size):
        self.x = self.x[0:size]
        self.y = self.y[0:size]
        self.mask = self.mask[0:size]
        self.masked_posn = self.masked_posn[0:size]
        self.position_ids = self.position_ids[0:size]
        self.static_embeddings = self.static_embeddings[0:size]
        self.labvalues = self.labvalues[0:size]
        self.patient_id_with_timestep = self.patient_id_with_timestep[0:size]
        return


def get_pos_id_uneven_bins(timestep):
    pos = 0
    if(timestep == 0):
        pos = 2
    elif(timestep == 1):
        pos = 4
    elif(timestep == 2):
        pos = 5
    elif(timestep == 3):
        pos = 6
    elif(timestep == 4):
        pos = 6.5
    elif(timestep == 5):
        pos = 7
    elif(timestep == 6):
        pos = 7.25
    elif(timestep == 7):
        pos = 7.5
    return pos * 4


def generate_position_and_mask(events, num_time_steps, timesteps,
                               use_uneven_time_bins=False,
                               nmask=1):
    index = 0
    masked_posn = []
    pos_ids = []
    max_seq_len = len(events[0])

    for event in events:
        mask_idx = random.sample(range(1, max_seq_len), nmask)
        masked_posn.append(mask_idx)
        if use_uneven_time_bins:
            pos_id = get_pos_id_uneven_bins(timesteps[index])
        pos_id = index % num_time_steps
        position_id = [pos_id] * max_seq_len
        pos_ids.append(position_id)
        index += 1
    return masked_posn, pos_ids


def pad_patient_timestamps(patient_timestamps, num_time_steps):
    last_timestamp_data = patient_timestamps[-1]
    for i in range(len(patient_timestamps), num_time_steps):
        patient_timestamps.append(last_timestamp_data)
    return patient_timestamps


def get_outcome_weights(windows, args):
    outcome = args.outcome_var

    print(f"Training for outcome: {outcome}", flush=True)
    num_out_classes = int(windows[outcome].nunique())

    counts = windows[outcome].value_counts(normalize=True)

    print("number of output class and their weights", num_out_classes, counts)
    counts_c0 = float(counts[0])
    counts_c1 = float(counts[1])
    weights = torch.tensor([counts_c1, counts_c0])
    return outcome, num_out_classes, weights


def prepare_finetuning_input(data, windows, tokenizer, args):
    # Prepare demo vector.
    if 'feature_exposure' in args.infile:
        demographics_mapper = DemoMapper_MACE(args.bin_age)
    elif 'mimic' in args.infile:
        demographics_mapper = DemoMapper_MIMIC(args.bin_age)
    else:
        demographics_mapper = DemoMapper(args.bin_age)

    # generic class name printing
    print(f"Using demographics mapper: {demographics_mapper.__class__.__name__}")
    # group_mapper = ClinicalGroupMapper(args.group_filename)
    embed_size = args.hidden_dim

    # get patient demographics vector
    #values = {"race": "unknown", "age": 1, "sex": "unknown", "ethnicity": "unknown"}
    #data = data.fillna(value=values)
    #get patient demographics vector
    def f_demo(row):
        return demographics_mapper.get_demographics_embed(row, embed_size, args)

    print("Mapping demographics data")
    data["demo"] = data.apply(f_demo, axis=1)

    #get patient clinical risk groups vector
    #def f_groups(row):
    #    return group_mapper.get_group_embed(row, embed_size, args)
    #data["groups"] = data.apply(f_groups, axis=1)

    # Add NLP as a separate attribute to the data dictionary (PROBABLY simpler to add a switch to get_demographics_embed for now)
    # nlp_mapper = NLPMapper()
    # def f(row):
    #    return nlp_mapper.get_nlp_embed(row, embed_size, args)
    # data['nlp'] = data.apply(f, axis=1)

    tokenizer.set_max_codes_in_visit(data.codes.apply(len).max())
    #data = data[['pid_vid', 'timestep', 'codes', 'demo', 'measurement_values']]

    selected_columns = ['pid_vid', 'timestep', 'codes', 'demo', 'measurement_values']
    # I think this code was due to VA memory constraints, dropping is lightweight
    # columns = set(data.columns.tolist())
    # columns_to_drop = columns.difference(set(selected_columns))
    # columns_to_drop = list(columns_to_drop)
    # print(f"Dropping {len(columns_to_drop)} columns", flush=True)
    # data.drop(columns_to_drop, axis=1, inplace=True)

    print("Tokenizing all symbolic data")
    data = data[selected_columns]

    def f(codes):
        return tokenizer.tokenize([codes])

    print("Running tokenizer to get inputs_ids and attention_masks columns", flush=True)
    data[['input_ids', 'attention_masks']] = data["codes"].apply(f)
    # This was crashing OOM, testing multiple return apply function
    #data = data.merge(data["codes"].apply(f), left_index=True, right_index=True)
    #print(f"{data}", flush=True)
    all_patient_events = []
    all_patient_event_attention_masks = []
    all_patient_outcomes = []
    all_patient_demographics = []
    windows = windows.sample(frac=1)  # shuffle samples
    num_time_steps = args.num_time_steps
    embed_size = args.hidden_dim
    rep_patient_ids = []
    patient_id_with_timestep = []
    all_patient_labvalues = []
    timesteps = []
    outcome = args.outcome_var

    # data cannot be shuffled after this
    # For each patient, we create 'a bert input for each timestep'
    # we then keep track of number of time step for each patient in order
    # During finetuning, bert output is concatenated to get a patient's event
    # embeddings for each time step
    print("Iterating through samples in windows and timesteps", flush=True)

    data = data.set_index(["pid_vid", "timestep"])
    print("Generating tensors")

    for sample in windows.itertuples():
        for timestep in sample.input_timesteps:
            data_row = data.loc[sample.pid_vid, timestep]
            all_patient_events.append(data_row.input_ids[0])
            rep_patient_ids.append(sample.pid_vid)
            patient_id_with_timestep.append(f"{sample.pid_vid};{timestep}")
            timesteps.append(timestep)
            all_patient_event_attention_masks.append(data_row.attention_masks[0])
            all_patient_outcomes.append([getattr(sample, outcome)])
            all_patient_demographics.append(data_row.demo)
            all_patient_labvalues.append(data_row.measurement_values)

    #print(
    #    f"Number of (instances, unique) missed codes during patient encoding from pretraining\
    #     data: {len(tokenizer.get_missed_codes()),len(set(tokenizer.get_missed_codes()))}",
    #   flush=True,
    #)
    # note that the len all tensors is num_patients*num_time_steps
    # The data for outcomes and patient demographics is repeated for num_time_steps (for size consistency, to create consistent batches)
    masked_posn, pos_ids = generate_position_and_mask(
        all_patient_events, num_time_steps, timesteps,
        use_uneven_time_bins=args.use_uneven_time_bins
    )
    # print ([len(i) for i in all_patient_events])
    all_patient_events = torch.tensor(all_patient_events)

    all_patient_outcomes = torch.tensor(all_patient_outcomes)
    all_patient_event_attention_masks = torch.tensor(all_patient_event_attention_masks)
    #print(
    #    f"{all_patient_events.shape}, {all_patient_outcomes.shape}, {all_patient_event_attention_masks.shape}",
    #    flush=True,
    #)

    all_patient_demographics = torch.cat(all_patient_demographics).reshape(
        -1, embed_size
    )
    masked_posn = torch.tensor(masked_posn)
    pos_ids = torch.tensor(pos_ids)
    all_patient_labvalues = torch.tensor(all_patient_labvalues)

    return rep_patient_ids, all_patient_events, all_patient_outcomes, all_patient_event_attention_masks, all_patient_demographics, all_patient_labvalues, masked_posn, pos_ids, patient_id_with_timestep, args, num_time_steps


def split_train_val_test_finetune(rep_patient_ids, patient_split, x, y, attention_mask,
                                  static_embeddings, labvalues, masked_posn, pos_ids,
                                  patient_id_with_timestep, num_time_steps,
                                  finetune_dir):

    train_patient_ids = patient_split["ft_train"]
    val_patient_ids = patient_split["ft_val"]
    test_patient_ids = patient_split["ft_test"]

    train_ind = [
        i for i in range(len(rep_patient_ids)) if rep_patient_ids[i] in train_patient_ids
    ]
    val_ind = [
        i for i in range(len(rep_patient_ids)) if rep_patient_ids[i] in val_patient_ids
    ]
    test_ind = [
        i for i in range(len(rep_patient_ids)) if rep_patient_ids[i] in test_patient_ids
    ]

    print(
        f"Number of indices in train: {len(train_ind)}, val: {len(val_ind)}, "
        f"test: {len(test_ind)}",
        flush=True,
    )

    time.sleep(2)

    training_patient_id_with_timestep = [patient_id_with_timestep[i] for i in train_ind]
    val_patient_id_with_timestep = [patient_id_with_timestep[i] for i in val_ind]
    test_patient_id_with_timestep = [patient_id_with_timestep[i] for i in test_ind]

    training_data = FineTuneData(x[train_ind, :],
                                 y[train_ind, :],
                                 attention_mask[train_ind, :],
                                 static_embeddings[train_ind, :],
                                 labvalues[train_ind, :],
                                 masked_posn[train_ind, :],
                                 pos_ids[train_ind, :],
                                 training_patient_id_with_timestep,
                                 num_time_steps)

    val_data = FineTuneData(x[val_ind, :],
                            y[val_ind, :],
                            attention_mask[val_ind, :],
                            static_embeddings[val_ind, :],
                            labvalues[val_ind, :],
                            masked_posn[val_ind, :],
                            pos_ids[val_ind, :],
                            val_patient_id_with_timestep,
                            num_time_steps)

    test_data = FineTuneData(x[test_ind, :],
                             y[test_ind, :],
                             attention_mask[test_ind, :],
                             static_embeddings[test_ind, :],
                             labvalues[test_ind, :],
                             masked_posn[test_ind, :],
                             pos_ids[test_ind, :],
                             test_patient_id_with_timestep,
                             num_time_steps)

    print(f"saved fine tuning data splits at {finetune_dir}", flush=True)
    torch.save(
        training_data,
        f"{finetune_dir}/ft_train.pt",
    )
    torch.save(
        test_data,
        f"{finetune_dir}/ft_test.pt",
    )
    torch.save(
        val_data,
        f"{finetune_dir}/ft_val.pt",
    )

    return training_data, val_data, test_data


def save_finetune_config(
    best_epoch, num_out_class, num_static_and_temporal_steps, precision, recall, avg_fscore, support, f1, auc, args, finetune_dir
):
    finetune_config = {
        "features": args.features,
        "outcome_var": args.outcome_var,
        "best_epoch": best_epoch,
        "d_ffn": args.d_ffn,
        "n_class": num_out_class,
        "precision": list(precision),
        "recall": list(recall),
        "avg_fscore": list(avg_fscore),
        "support": str(support),
        "f1": f1,
        "auc": auc,
        "timesteps": args.num_time_steps,
        "timesteps_including_static": num_static_and_temporal_steps,
        "lookahead": args.lookahead,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
    }
    with open(f"{finetune_dir}/finetune_results.json", 'a') as f:
        result = json.dumps(finetune_config)
        f.write(f"{result}\n")


def get_patient_finetune_model(ft_layer, hidden_dim, num_static_and_temporal_steps, d_ffn, num_out_classes, batch_size, dropout):
    if ft_layer == "bert":
        print("USING BERT FOR FINETUNING")
        patient_finetune_model = PatientFineTuneBERTModel(
            hidden_dim,
            num_static_and_temporal_steps,
            d_ffn,
            num_out_classes,
            batch_size,
        )
    elif ft_layer == "gru":
        print("USING GRU FINETUNING")
        patient_finetune_model = PatientFineTuneGRUModel(
            hidden_dim,
            num_static_and_temporal_steps,
            d_ffn,
            num_out_classes,
            batch_size,
            dropout,
        )
    elif ft_layer == "dense":
        print("USING DENSE FINETUNING")
        patient_finetune_model = PatientFineTuneModel(
            hidden_dim,
            num_static_and_temporal_steps,
            d_ffn,
            num_out_classes,
            dropout,
        )
    else:
        print(f"Unhandled ft_layer: {ft_layer}, exiting...")
        sys.exit(1)

    return patient_finetune_model
