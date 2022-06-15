import argparse
import json
import os
import sys

import torch

print(f"sys.path: {sys.path}")
print(f"os.getcwd(): {os.getcwd()}")
sys.path.append(os.getcwd())
print(f"sys.path: {sys.path}")
from src.config import parse_arguments
from src.data_loader import get_lab_codes_for_dataset, load_dataframe
from src.finetune_df import (FineTuneData, get_outcome_weights,
                             prepare_finetuning_input)
from src.pretrain_df import load_pretrain_model
from src.ssl import PatientFineTuneModel
from src.ssl_finetune_runner import SSLFineTuneRunner


def load_finetune_config(args):
    #finetune_config_file = args.finetune_dir + "/finetune_results_outcome_ventilation.json"
    finetune_config_file = args.finetune_dir + "/finetune_results_outcome_mace.json"

    with open(finetune_config_file) as fp:
        last_line = fp.readlines()[-1]
        finetune_config = json.loads(last_line)
        print(finetune_config)
    return finetune_config

def get_test_df(data, windows, test_patient_ids):
    data = data.loc[data['pid_vid'].isin(test_patient_ids)]
    windows = windows.loc[windows['pid_vid'].isin(test_patient_ids)]
    data = data.groupby(['pid_vid'])
    return data, windows


def init(args):
    #get finetune_config
    finetune_config = load_finetune_config(args)
    best_epoch = int(finetune_config["best_epoch"])
    print("Best epoch", best_epoch)
    args.features = finetune_config['features']
    args.outcome_var = finetune_config['outcome_var']
    args.num_time_steps = finetune_config['timesteps']
    num_static_and_temporal_steps = finetune_config['timesteps_including_static']
    args.lookahead = finetune_config['lookahead']
    num_out_classes = 2
    args.hidden_dim = finetune_config['hidden_dim']

    # specify lab codes we are looking for for each dataset in this funciton
    lab_codes = get_lab_codes_for_dataset(args.infile)

    # get data
    data, windows =  load_dataframe(args.data_dir, args.infile, args.model, args.features, lab_codes)
    print(data.columns)
    print(windows.columns)


    # load_pretrained_models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}", flush=True)
    patient_pretrain_model, tokenizer, pretrain_config = load_pretrain_model(args.pretrain_dir)
    args.ft_lr =0.01
    args.ft_scheduler_patience=10
    args.topk=1
    outcome, num_out_classes, weights = get_outcome_weights(windows, args)
    weights = weights.to(device)
    print(f"weights: {weights}", flush=True)
    args.batch_size = args.num_time_steps * args.batch_size
    print(" Init fine tune runner with batch size", args.batch_size)



    patient_finetune_model = PatientFineTuneModel(
        args.hidden_dim,
        num_static_and_temporal_steps,
        args.d_ffn,
        num_out_classes,
        args.dropout,
    )
    patient_finetune_model.load_state_dict(
            torch.load(
                f"{args.finetune_dir}/checkpoints/finetune_model.{best_epoch}.ckpt",
                map_location="cpu",
            )
        )



    model_runner = SSLFineTuneRunner(
            patient_pretrain_model,
            patient_finetune_model,
            10,
            args.batch_size,
            10,
            10,
            args.finetune_dir,
            args,
            device,
            weights,
    )

    return patient_finetune_model, model_runner, tokenizer, data, windows



class PatientInferenceModel():
    def __init__(self, model, model_runner, tokenizer, windows, args):
        self.model = model
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.windows = windows
        self.args = args
        self.output_names = ["risk score"]

    def __call__(self, patient_df):
        scores = dict()

        for patient_iter in patient_df:
            pid = patient_iter[0]
            patient_windows = self.windows.loc[self.windows['pid_vid'] == pid]
            #print(pid, patient_iter[1])
            #print(patient_windows)
            (rep_patient_ids, x, y, attention_mask, static_embeddings,
             labvalues, masked_posn, pos_ids, patient_id_with_timestep,
            args2, num_time_steps) = prepare_finetuning_input(patient_iter[1], patient_windows, self.tokenizer, self.args)
            #test_ind = [i for i in range(len(rep_patient_ids)) if rep_patient_ids[i] in pid]
            test_ind = [i for i in range(len(rep_patient_ids)) if str(rep_patient_ids[i]) in str(pid)]
            test_patient_id_with_timestep = [patient_id_with_timestep[i] for i in test_ind]
            #print(pid, test_patient_id_with_timestep, test_ind)
            tmp = FineTuneData(x[test_ind, :],
                               y[test_ind, :],
                               attention_mask[test_ind, :],
                               static_embeddings[test_ind, :],
                               labvalues[test_ind, :],
                               masked_posn[test_ind, :],
                               pos_ids[test_ind, :],
                               test_patient_id_with_timestep,
                               num_time_steps)
            #print(len(x), len(y), len(rep_patient_ids))
            #run_test expects a finetune data object, so we run per patient
            mydict = self.model_runner.run_test(self.model, tmp, return_score_dict_only = True)

            scores.update(mydict)

        return scores



def run_inference(args, test_patient_ids):
    """ @Input : (run time args , list(patient_ids))
    reads finetune runtime inference config from
    args.finetune_config_file , load a pretrained model
    and generates scores for all patient_ids prescribed in windowed setting
    @output: dict("patientid;timestep": [y_pred, y_true, y_score])
    """

    patient_finetune_model, model_runner, tokenizer, data, windows = init(args)
    mymodel = PatientInferenceModel(patient_finetune_model, model_runner, tokenizer, windows, args)
    test_patient_df, test_windows = get_test_df(data, windows, test_patient_ids)
    print(len(test_patient_df), len(test_windows))
    print(test_windows['pid_vid'].values)
    test_result = mymodel(test_patient_df)
    print(test_result)


if __name__ == "__main__":
    args = parse_arguments()
    print(f"{args}", flush=True)
    pids = [0, 1]
    run_inference(args, pids)
