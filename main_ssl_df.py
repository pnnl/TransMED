import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import BertConfig, BertModel

from src.config import parse_arguments
from src.data_loader import get_lab_codes_for_dataset, load_dataframe
from src.finetune_df import (get_outcome_weights, get_patient_finetune_model,
                             prepare_finetuning_input, save_finetune_config,
                             split_train_val_test_finetune)
from src.patient_splits import (generate_patient_split,
                                generate_patient_split_path,
                                load_patient_split)
from src.pretrain_df import (load_pretrain_model, modify_pretrain_dir,
                             prepare_pretrain_inputs, save_pretrain_model,
                             split_train_val_test)
from src.ssl import PatientSSLModel
from src.ssl_finetune_runner import SSLFineTuneRunner
from src.ssl_model_runner import SSLRunner


def sort_features(features, sep='_'):
    print(f"Sorting features for consistency")
    print(f"features: {features}")
    features_list = features.split(sep)
    features_list.sort()
    sorted_features = sep.join(features_list)
    print(f"sorted_features: {sorted_features}")
    return sorted_features


def main(args):
    global data, windows
    # sort features for consistent input/output paths
    args.features = sort_features(args.features)

    # specify lab codes we are looking for for each dataset in this funciton
    lab_codes = get_lab_codes_for_dataset(args.infile)

    # load data
    data, windows = load_dataframe(args.data_dir, args.infile, args.model, args.features, lab_codes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}", flush=True)

    """
    take patient data dictionary as input and create
    1) Input tensor of size [number_of_patients *max_number_visits *max_features_in_visit]
    Each visit tensor is coded as [1, 0, ....1, 0], where
    1/0 represent presence/absence of a feature in visit
    2) output tensor of size  [number_of_patients * len(outcome_vars)]
    """
    if args.model.lower() == "pretrain":
        """
        use patient data to create following for SSL
        1) x = Embedding(set of features observed at same time')
        2) y = id of feature that was observed but is masked in x
        3) attention_mask is a boolean_tensor of length = length(x), where
        attention_mask[i] = True if 'i' is a masked position and False otherwise
        """
        print(f"pretrain_dir: {args.pretrain_dir}")
        if args.create_uniq_dirs:
            # set pretrain_dir to feature specific path
            args.pretrain_dir = f"{args.pretrain_dir}/{args.features}"
            print(f"features specific pretrain_dir: {args.pretrain_dir}")
        # make sure pretrain_dir exists
        Path(args.pretrain_dir).mkdir(parents=True, exist_ok=True)

        (
            x,
            attention_mask,
            y,
            masked_positions,
            tokenizer,
            visit_vocab_size,
        ) = prepare_pretrain_inputs(data, args)

        # the vA unven time bins are multiplied by 4 to provide
        # unique integers for 3 month period bins
        max_positions = 32 if args.use_uneven_time_bins else args.num_time_steps + 1
        print("using max positions", max_positions)
        config = BertConfig(
            vocab_size=visit_vocab_size + 3,  # FOR cls, masking and padding
            hidden_size=args.hidden_dim,
            num_hidden_layers=args.nlayers,
            num_attention_heads=args.nheads,
            max_position_embeddings=max_positions,
            pad_token_id=tokenizer.PADDING,
            intermediate_size=4 * args.hidden_dim
        )

        base_bert_model = BertModel(config)
        d_model = config.hidden_size
        patient_model = PatientSSLModel(base_bert_model, d_model, visit_vocab_size)
        model_runner = SSLRunner(
            patient_model,
            args.num_epochs,
            args.batch_size,
            args.pt_patience,
            args.pt_patience_threshold,
            args.pretrain_dir,
            args,
            device,
        )

        print("Using split_train_val_test")
        training_data, val_data, test_data = split_train_val_test(
            x, attention_mask, y, masked_positions, args
        )

        best_model, best_epoch = model_runner.run_train(
            training_data.x,
            training_data.y,
            training_data.mask,
            training_data.masked_pos,
            val_data.x,
            val_data.y,
            val_data.mask,
            val_data.masked_pos,
        )
        test_accuracy = model_runner.run_test(
            best_model, test_data.x, test_data.y, test_data.mask, test_data.masked_pos
        )
        save_pretrain_model(tokenizer, config, best_epoch, best_model, args.pretrain_dir)

        logfile = os.path.join(args.pretrain_dir, "pretrain_log.csv")
        args_dict = vars(args)
        if not (os.path.exists(logfile)):
            with open(logfile, "w") as f:
                f.write(",".join(list(args_dict.keys())) + ",accuracy\n")
        args_dict.update({"accuracy": test_accuracy})
        log = pd.read_csv(logfile)
        log = pd.concat([log, pd.DataFrame({k: [v] for k, v in args_dict.items()})])
        log.to_csv(logfile, index=False)

    elif args.model.lower() == "finetune":

        ft_uniq_path = f"ts{args.num_time_steps}_l{args.lookahead}_{args.features}"
        print(f"ft_uniq_path: {ft_uniq_path}")

        if args.create_uniq_dirs:
            args.finetune_dir = f"{args.finetune_dir}/{args.outcome_var}/run_{args.run}/{ft_uniq_path}"

        print(f"finetune_dir: {args.finetune_dir}")
        Path(args.finetune_dir).mkdir(parents=True, exist_ok=True)
        print(f" time binning enabled {args.bin_age}")

        if args.force_binary_outcome:
            print(f"Forcing two output classes for outcome: {args.outcome_var}")
            # Any frequency outcome variable greater than or equal to 1 is set to 1
            windows.loc[windows[args.outcome_var] >= 1, args.outcome_var] = 1

        outcome, num_out_classes, weights = get_outcome_weights(windows, args)
        weights = weights.to(device)
        print(f"weights: {weights}", flush=True)

        if args.modify_pretrain_dir:
            # drop finetuning specific features
            # this must be removed or modified if you wish to use pretraining with different features
            args.pretrain_dir = modify_pretrain_dir(args.pretrain_dir, args.features)
        patient_pretrain_model, tokenizer, pretrain_config = load_pretrain_model(args.pretrain_dir)
        args.hidden_dim = pretrain_config["hidden_size"]
        print(f"using hidden dimension/embed_size ={args.hidden_dim}")
        # add an extra timestep for demo or nlp if using empty demo
        add_time_step = "demo" in args.features or "nlp" in args.features
        num_static_and_temporal_steps = (
            args.num_time_steps + 1 if add_time_step else args.num_time_steps
        )

        if "val" in args.features:
            args.hidden_dim += 6  # Add no. of lab codes

        print(
            f"number of time steps considered in finetuning {num_static_and_temporal_steps}",
            flush=True,
        )
        patient_finetune_model = get_patient_finetune_model(args.ft_layer, args.hidden_dim, num_static_and_temporal_steps, args.d_ffn, num_out_classes, args.batch_size, args.dropout)

        args.batch_size = args.num_time_steps * args.batch_size
        # returns a sequence of events, ordered by patient
        # patient_event_seq =  [
        # patient1_encoded_event_1,
        # patient1_encoded_event_2,
        # ..
        # patient1_encoded_event_n,
        # patient2_encoded_event_1,
        # ..
        # patient2_encoded_event_n]
        #
        # patient outcome variables is repeated for each patient id
        # num_time_steps times in the patient_outcome tensor
        # patient_output = [
        # patient1_output,
        # patient1_output,
        # ...
        # patient1_output (num_time_step times)
        # patient2_output,
        # patient2_output
        # ..
        # similar tensor is created for attention mask (corresponsing to each
        # event)

        if (
            os.path.isfile(
                f"{args.finetune_dir}/ft_train.pt"
            )
            and os.path.isfile(
                f"{args.finetune_dir}/ft_test.pt"
            )
            and os.path.isfile(
                f"{args.finetune_dir}/ft_val.pt"
            )
        ):

            print(
                f"Loading pre-defined training, validation and test tensors\
                  from {args.finetune_dir}",
                flush=True,
            )
            training_data = torch.load(
                f"{args.finetune_dir}/ft_train.pt"
            )
            val_data = torch.load(
                f"{args.finetune_dir}/ft_val.pt"
            )
            test_data = torch.load(
                f"{args.finetune_dir}/ft_test.pt"
            )
        else:
            print("Creating input tensors .....", flush=True)
            (
                rep_patient_ids,
                all_patient_events,
                all_patient_outcomes,
                all_patient_event_attention_masks,
                all_patient_demographics,
                all_patient_labvalues,
                masked_posn,
                pos_ids,
                patient_id_with_timestep,
                args,
                num_time_steps,
            ) = prepare_finetuning_input(data, windows, tokenizer, args)
            print("here2", flush=True)

            patient_split_path = generate_patient_split_path(args.ft_base_path, args.run, patient_splits_dir=args.patient_splits_dir)
            patient_split = load_patient_split(patient_split_path)
            # if patient split doesn't exist, create it
            if not patient_split:
                print(f"Patient split doesn't exist already so we are generating it")
                patient_split = generate_patient_split(args.ft_base_path, args.run, rep_patient_ids, val_split=args.val_split, test_split=args.test_split, patient_splits_dir=args.patient_splits_dir)

            training_data, val_data, test_data = split_train_val_test_finetune(
                rep_patient_ids,
                patient_split,
                all_patient_events,
                all_patient_outcomes,
                all_patient_event_attention_masks,
                all_patient_demographics,
                all_patient_labvalues,
                masked_posn,
                pos_ids,
                patient_id_with_timestep,
                num_time_steps,
                args.finetune_dir
            )

        model_runner = SSLFineTuneRunner(
            patient_pretrain_model,
            patient_finetune_model,
            args.num_epochs,
            args.batch_size,
            args.ft_patience,
            args.ft_patience_threshold,
            args.finetune_dir,
            args,
            device,
            weights,
        )

        # run_train would go through each event of a patient , get its embedding
        # from BERT, combine it to get a single patient vector and call fine
        # tuning model
        best_model, best_epoch = model_runner.run_train(training_data, val_data)

        (
            precision,
            recall,
            avg_fscore,
            support,
            f1,
            auroc,
            auprc,
        ), patient_score_dict = model_runner.run_test(best_model, test_data)

        patient_score_dict_path = f"{args.finetune_dir}/patient_score_dict.json"
        print(f"Dumping patient score dict to {patient_score_dict_path}")
        with open(patient_score_dict_path, 'w') as f:
            json.dump(patient_score_dict, f)

        print(
            f"precision: {precision}, recall: {recall}, avg_fscore: {avg_fscore}, "
            f"support: {support}, f1: {f1}, auroc: {auroc}, auprc: {auprc}",
            flush=True,
        )
        save_finetune_config(
            best_epoch,
            num_out_classes,
            num_static_and_temporal_steps,
            precision,
            recall,
            avg_fscore,
            support,
            f1,
            auroc,
            args,
            args.finetune_dir
        )
        logfile = os.path.join(args.finetune_dir, f"finetune_log.csv")
        args_dict = vars(args)
        if not (os.path.exists(logfile)):
            with open(logfile, "w") as f:
                f.write(
                    ",".join(list(args_dict.keys()))
                    + ",precision, recall, avg_fscore, support, f1, auroc, auprc\n"
                )
        args_dict.update(
            {
                "precision": precision,
                "recall": recall,
                "avg_fscore": avg_fscore,
                "support": support,
                "f1": f1,
                "auroc": auroc,
                "auprc": auprc,
            }
        )
        log = pd.read_csv(logfile)
        log = pd.concat([log, pd.DataFrame({k: [v] for k, v in args_dict.items()})])
        log.to_csv(logfile, index=False)
    else:
        print(f"Unhandled model option: {args.model}")
        print("Please use an accepted model option: ['pretrain', 'finetune']")
        print(f"Exiting...")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    print(f"{args}", flush=True)
    main(args)
