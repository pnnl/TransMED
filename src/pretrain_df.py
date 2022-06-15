import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel

from src.ssl import PatientSSLModel
from src.tokenizer import MedTokenizer, load_tokenizer


class PretrainData:
    def __init__(self, x, y, masked_pos, attention_mask):
        self.x = x
        self.y = y
        self.masked_pos = masked_pos
        self.mask = attention_mask

    def take_slice(self, beg, end):
        self.x = self.x[beg:end, :]
        self.y = self.y[beg:end, :]
        self.masked_pos = self.masked_pos[beg:end, :]
        self.mask = self.mask[beg:end, :]


def masking_function(events, attention_masks, PADDING, MASK, args):
    """
    take patient events and performs masking for each event, generating
    args.nmask samples per event
    """
    masked_encoded_events = []
    masked_attention_masks = []
    masked_labels = []
    masked_positions = []
    for i in range(int(args.nmask)):
        for event, attention_mask in zip(events, attention_masks):
        # Sample the mask index from valid codes(ignoring padded tokens) in the event
        # Create more masked samples per event - Khushbu

            try:
                num_features_in_event = event.index(PADDING)
                mask_pos = random.sample(range(1, num_features_in_event), 1)
            except ValueError:
                mask_pos = random.sample(range(1, len(event)), 1)
            #event_label = []  # [-100] * len(event)
            # print("selected mask at id, masked", mask_pos, event[mask_pos])
            mask_pos = mask_pos[0]
            event_label = [event[mask_pos]]
            event[mask_pos] = MASK
            attention_mask[mask_pos] = 0
            masked_encoded_events.append(event)
            masked_attention_masks.append(attention_mask)
            masked_labels.append(event_label)
            masked_positions.append([mask_pos])
    return masked_encoded_events, masked_attention_masks, masked_labels, masked_positions



def prepare_pretrain_inputs(data, args, mask_type="random"):

    all_coded_events = list(data["codes"])
    random.shuffle(all_coded_events)

    # all_codes = sorted(list(set(np.concatenate(all_coded_events))))
    # effecient way to find all codes
    all_codes = set()
    for codes in all_coded_events:
        all_codes.update(codes)

    all_codes = sorted(all_codes)
    code_to_ix = {all_codes[i]: i for i in range(len(all_codes))}
    ix_to_code = {v: k for k, v in code_to_ix.items()}
    visit_vocab_size = len(all_codes)

    PADDING = visit_vocab_size
    MASK = visit_vocab_size + 1
    CLS = visit_vocab_size + 2
    max_num_codes_in_visit = data["codes"].apply(len).max()
    tokenizer = MedTokenizer(
        code_to_ix, ix_to_code, max_num_codes_in_visit, PADDING, MASK, CLS
    )
    events, attention_masks = tokenizer.tokenize(all_coded_events)

    print(
        f"Shape of input id: {np.array(events).shape}"
        f" and attention mask: {np.array(attention_masks).shape} after tokenize()",
        flush=True,
    )


    (masked_encoded_events,
     masked_attention_masks,
     masked_labels,
    masked_positions) = masking_function(events, attention_masks, PADDING, MASK, args)


    print(
        f"shape of events: {np.array(masked_encoded_events).shape}, "
        f"attention mask: {np.array(masked_attention_masks).shape}, "
        f"masked_labels: {np.array(masked_labels).shape}, "
        f"masked_positions: {np.array(masked_positions).shape}",
        flush=True,
    )

    x = torch.tensor(masked_encoded_events)
    attention_mask = torch.tensor(masked_attention_masks)
    y = torch.tensor(masked_labels)
    masked_positions = torch.tensor(masked_positions)
    return x, attention_mask, y, masked_positions, tokenizer, visit_vocab_size


def split_train_val_test(x, attention_mask, y, masked_positions, args):
    val_split = args.val_split
    test_split = args.test_split
    train_split = 1.0 - (test_split + val_split)
    num_total_visits = int(x.size(0))
    num_train = int(num_total_visits * train_split)
    num_val = int(num_total_visits * val_split)
    num_test = int(num_total_visits * test_split)

    print(
        f"Number of train: {num_train}, val: {num_val}, test: {num_test} ", flush=True
    )
    training_data = PretrainData(x, y, masked_positions,
                                 attention_mask)
    training_data.take_slice(0, num_train)
    val_data = PretrainData(x, y, masked_positions,
                            attention_mask)
    val_data.take_slice(num_train, num_train+num_val)
    test_data = PretrainData(x, y, masked_positions,
                             attention_mask)
    test_data.take_slice(num_train+num_val, num_total_visits)
    return training_data, val_data, test_data


def split_train_val_test_VA(x, attention_mask, y, masked_positions, args):
    val_split = args.val_split
    test_split = args.test_split
    train_split = 1.0 - (test_split + val_split)
    num_total_visits = int(x.size(0))
    num_train = int(num_total_visits * train_split)
    num_val = int(num_total_visits * val_split)
    num_test = int(num_total_visits * test_split)

    print(
        f"Number of train: {num_train}, val: {num_val}, test: {num_test} ", flush=True
    )
    x_train = x[0:num_train, :]
    y_train = y[0:num_train, :]
    mask_train = attention_mask[0:num_train, :]
    masked_pos_train = masked_positions[0:num_train, :]
    training_data = PretrainData(x_train, y_train, masked_pos_train, mask_train)

    x_val = x[num_train : num_train + num_val, :]
    y_val = y[num_train : num_train + num_val, :]
    mask_val = attention_mask[num_train : num_train + num_val, :]
    masked_pos_val = masked_positions[num_train : num_train + num_val, :]
    val_data = PretrainData(x_val, y_val, masked_pos_val, mask_val)

    x_test = x[num_train + num_val :, :]
    y_test = y[num_train + num_val :, :]
    mask_test = attention_mask[num_train + num_val :, :]
    masked_pos_test = masked_positions[num_train + num_val :, :]
    test_data = PretrainData(x_test, y_test, masked_pos_test, mask_test)

    print(f"{x_train.size()}, {y_train.size()}, {mask_train.size()}", flush=True)
    print(f"{x_val.size()}, {y_val.size()}, {mask_val.size()}", flush=True)
    print(f"{x_test.size()}, {y_test.size()}, {mask_test.size()}", flush=True)

    return training_data, val_data, test_data


def load_pretrain_model(pretrain_dir):
    checkpoints_dir = f"{pretrain_dir}/checkpoints"
    print(f"Loading pretrain model from {checkpoints_dir}")
    try:
        with open(f"{pretrain_dir}/pretrain_config.json") as f:
            pretrain_config = json.load(f)
        tokenizer = load_tokenizer(f"{pretrain_dir}/tokenizer.json")
        config = BertConfig(
            vocab_size=pretrain_config["vocab_size"],
            hidden_size=pretrain_config["hidden_size"],
            num_hidden_layers=pretrain_config["num_hidden_layers"],
            num_attention_heads=pretrain_config["num_attention_heads"],
            max_position_embeddings=pretrain_config["max_position_embeddings"],
            pad_token_id=pretrain_config["pad_token_id"],
            intermediate_size=pretrain_config["intermediate_size"],
        )

        base_bert_model = BertModel(config)
        d_model = config.hidden_size
        best_epoch = pretrain_config["best_epoch"]

        # Note that the BERT vocab_size includes PADDING and MASKING token,
        # However, the vocab_size used in
        # PatientSSLModel = bert_vocab_size-2 (does not include PADDING and MASKING)
        patient_pretrain_model = PatientSSLModel(
            base_bert_model, d_model, pretrain_config["vocab_size"] - 3
        )
        patient_pretrain_model.load_state_dict(
            torch.load(
                f"{checkpoints_dir}/pretrain_model.{best_epoch}.ckpt",
                map_location="cpu",
            )
        )
        patient_pretrain_model.set_finetune(True)
        print(
            f"Loaded pretrain model and configuration from {checkpoints_dir}",
            flush=True,
        )
        return patient_pretrain_model, tokenizer, pretrain_config
    except IOError:
        print(
            f"Could not find pretrain_configuration or tokenizer file in {checkpoints_dir}",
            flush=True,
        )
        sys.exit(1)


def save_pretrain_model(tokenizer, config, best_epoch, best_model, pretrain_dir):
    tokenizer.save(f"{pretrain_dir}/tokenizer.json")
    # make sure checkpoints directory exists
    checkpoints_dir = f"{pretrain_dir}/checkpoints"
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    # save checkpoint to checkpoints_dir
    torch.save(
        best_model.state_dict(),
        f"{checkpoints_dir}/pretrain_model.{best_epoch}.ckpt",
    )

    pretrain_config = {
        "best_epoch": best_epoch,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "pad_token_id": config.pad_token_id,
        "intermediate_size": config.intermediate_size,
    }
    with open(f"{pretrain_dir}/pretrain_config.json", "w") as f:
        json.dump(pretrain_config, f)


def modify_pretrain_dir(pretrain_dir, ft_features, sep='_'):
    ft_only_features = ['demo', 'nlp', 'val']
    # make sure features are sorted
    ft_features_list = ft_features.split(sep)
    ft_features_list.sort()
    # drop features that don't matter for pretraining
    pt_features_list = [i for i in ft_features_list if i not in ft_only_features]
    # make sure modified features are sorted
    pt_features_list.sort()
    pretrain_features_string = sep.join(pt_features_list)
    # combine pretrain_dir with features subdir
    modified_pretrain_dir = f"{pretrain_dir}/{pretrain_features_string}"
    return modified_pretrain_dir
