import argparse


# If nothing is passed to this function, sys.argv will be used for argv
def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="finetune", type=str, help="pretrain, finetune"
    )
    parser.add_argument(
        "--infile",
        default="",
        type=str,
        help="path to top patient data folder contaiing DOB cohorts",
    )
    parser.add_argument(
        "--data_type",
        default="shc",
        type=str,
        help="random (generates random patient profiles)/shc(stanford health data)",
    )
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="SSL hidden dimension size"
    )
    parser.add_argument(
        "--dropout", default=0.3, type=float, help="SSL dropout"
    )
    parser.add_argument("--nheads", default=2, type=int, help="number of heads for SSL")
    parser.add_argument(
        "--nlayers", default=2, type=int, help="number of encoder layers in SSL"
    )
    parser.add_argument(
        "--nmask",
        default=1,
        type=int,
        help="number of masked features per sample during pretraining",
    )
    parser.add_argument(
        "--topk",
        default=3,
        type=int,
        help="consider topk matches for a positive score for masked code prediction during eval",
    )

    parser.add_argument(
        "--pretrain_dir",
        type=str,
        help="directory to save/load pretrained model and config",
    )
    parser.add_argument(
        "--finetune_dir",
        type=str,
        help="directory to save/load finetuned model and config",
    )
    parser.add_argument(
        "--num_time_steps",
        default=2,
        type=int,
        help="number of patient events to consider for \
                        predicting patient outcome",
    )

    parser.add_argument(
        "--lookahead",
        default=2,
        type=int,
        help="prediction timestep after observation, 1 means \
                        predict next timestep after observation, 2 => \
                        predict 2nd timestep after observation ends and so on",
    )
    parser.add_argument(
        "--window_patient_data", default=1, type=int, help="window patient data 0/1"
    )
    parser.add_argument(
        "--aggregation_interval",
        default="24",
        type=str,
        help="6 ,12 or 24 hours to aggregate inpatient stay data",
    )
    parser.add_argument(
        "--features",
        default="dx_rx_pr_mea_demo_nlp_val",
        type=str,
        help="a string containing any combination of [dx, rx, proc, labs, demo",
    )
    parser.add_argument(
        "--use_uneven_time_bins", action="store_true",
        help="use VA CVD uneven time bins info for position encoding"
    )
    parser.add_argument(
        "--outcome_var", default="outcome_los", type=str, help="train for outcome"
    )

    parser.add_argument(
        "--log_interval",
        default=128,
        type=int,
        help="number of batches after which to do validation",
    )
    parser.add_argument(
        "--num_epochs", default=10, type=int, help="number of epochs for training"
    )
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--pt_patience", default=15, type=int, help="patience for pretraining")
    parser.add_argument("--ft_patience", default=15, type=int, help="patience for finetuning")
    parser.add_argument(
        "--pt_patience_threshold",
        default=0.0001,
        type=float,
        help="Minimum relative loss decrease before new best val loss is found for pretraining",
    )
    parser.add_argument(
        "--ft_patience_threshold",
        default=0.0001,
        type=float,
        help="Minimum relative loss decrease before new best val loss is found for finetuning",
    )
    parser.add_argument("--pt_lr", default=0.01, type=float, help="initial learning rate for pretraining")
    parser.add_argument("--ft_lr", default=0.01, type=float, help="initial learning rate for pretraining")
    parser.add_argument(
        "--val_split",
        default=0.2,
        type=float,
        help="fraction of data to use for validation",
    )
    parser.add_argument(
        "--test_split",
        default=0.2,
        type=float,
        help="fraction of data to use for testing",
    )
    parser.add_argument(
        "--d_ffn", default=64, type=int, help="ffn dimension in finetuning"
    )
    parser.add_argument("--sampling", default=None, help="undersampling/oversampling")
    parser.add_argument(
        "--sampling_ratio", default=1, type=float, help="majority to minoriy ratio"
    )
    parser.add_argument("--run", default="9", help="run")
    parser.add_argument("--pt_scheduler_patience", default=5, type=int)
    parser.add_argument("--ft_scheduler_patience", default=5, type=int)
    # VA Arguments
    parser.add_argument("--data_dir", type=str, help="top level directory for data")
    parser.add_argument("--task", type=str, default="mace", help='Task name, follows convention of benchmarks, unused so far')
    parser.add_argument(
        "--ft_base_path", type=str, help="base path for finetuning data"
    )
    parser.add_argument("--ft_layer", default="dense", help="dense | bert | gru")
    parser.add_argument("--bin_age", action='store_true', help='bin age in intervals')
    parser.add_argument("--patient_splits_dir", default="patient_splits")
    parser.add_argument("--create_uniq_dirs", action='store_true', help='Create uniq directories for pretrain and finetune output dependent on features, lookahead, num_timesteps, and run')
    parser.add_argument("--modify_pretrain_dir", action='store_true', help='Modify pretrain dir to load from uniq generated path from --create_uniq_dirs options used in pretraining')
    parser.add_argument('--force_binary_outcome', action='store_true')
    args = parser.parse_args(argv)
    return args
