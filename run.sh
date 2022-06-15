date
printf "\nPretraining\n"
python main_ssl_df.py \
    --model 'pretrain' \
    --num_epochs 2 \
    --pt_lr 0.0005 \
    --batch_size 64 \
    --data_dir 'data/mimic' \
    --infile 'mimic.pkl' \
    --pretrain_dir 'pretrain' \
    --topk 3 \
    --num_time_steps 2 \
    --features 'dx_pr_rx_mea' \
    --hidden_dim 64

date
printf "\nFinetuning\n"
python main_ssl_df.py \
    --model 'finetune' \
    --num_epochs 2 \
    --batch_size 64 \
    --outcome_var 'outcome_readmission' \
    --ft_lr 0.0005 \
    --data_dir 'data/mimic' \
    --infile 'readmission_mimic_lookahead1_numts2.pkl' \
    --ft_base_path 'mimic' \
    --pretrain_dir 'pretrain' \
    --finetune_dir 'finetune' \
    --features 'dx_pr_rx_mea_demo' \
    --topk 1 \
    --lookahead 1 \
    --num_time_steps 2 \
    --hidden_dim 64 \
    --bin_age

printf "\nDONE\n"
date