# Benchmarks

## Environment

- Python 3.8
- Dependencies:
  - scikit-learn 0.24.1
  - TensorFlow 2.6.0
  - xgboost 1.5.0
- CPU Install

```shell
conda create -n transmed_benchmarks python=3.8
conda activate transmed_benchmarks
pip install -r environment/benchmarks/benchmarks.cpu.requirements.txt
```

## Gated Recurrent Unit (GRU)

```shell
python benchmarks/main_gru.py \
    --data_dir 'data/mimic' \
    --infile 'readmission_mimic_lookahead1_numts2.pkl' \
    --task 'readmission' \
    --base_path 'mimic' \
    --num_timesteps 2 \
    --lookahead 1 \
    --features 'dx_pr_rx_demo' \
    --run 9 \
    --results_dir "benchmarks_results/lr_results/mimic/outcome_readmission" \
    --outcome_var "outcome_readmission" \
    --epochs 10
```

## Logistic Regression (LR)

```shell
python benchmarks/main_lr.py \
    --data_dir 'data/mimic' \
    --infile 'readmission_mimic_lookahead1_numts2.pkl' \
    --task 'readmission' \
    --base_path 'mimic' \
    --num_timesteps 2 \
    --lookahead 1 \
    --features 'dx_pr_rx_demo' \
    --run 9 \
    --results_dir "benchmarks_results/lr_results/mimic/outcome_readmission" \
    --outcome_var "outcome_readmission"
```

## Multilayer Perceptron (MLP)

```shell
python benchmarks/main_mlp.py \
    --data_dir 'data/mimic' \
    --infile 'readmission_mimic_lookahead1_numts2.pkl' \
    --task 'readmission' \
    --base_path 'mimic' \
    --num_timesteps 2 \
    --lookahead 1 \
    --features 'dx_pr_rx_demo' \
    --run 9 \
    --results_dir "benchmarks_results/lr_results/mimic/outcome_readmission" \
    --outcome_var "outcome_readmission" \
    --embedding_method "one_hot" \
    --epochs 10
```

## Random Forest (RF)

```shell
python benchmarks/main_rf.py \
    --data_dir 'data/mimic' \
    --infile 'readmission_mimic_lookahead1_numts2.pkl' \
    --task 'readmission' \
    --base_path 'mimic' \
    --num_timesteps 2 \
    --lookahead 1 \
    --features 'dx_pr_rx_demo' \
    --run 9 \
    --results_dir "benchmarks_results/lr_results/mimic/outcome_readmission" \
    --outcome_var "outcome_readmission"
```

## XGBoost (XGB)

```shell
python benchmarks/main_xgb.py \
    --data_dir 'data/mimic' \
    --infile 'readmission_mimic_lookahead1_numts2.pkl' \
    --task 'readmission' \
    --base_path 'mimic' \
    --num_timesteps 2 \
    --lookahead 1 \
    --features 'dx_pr_rx_demo' \
    --run 9 \
    --results_dir "benchmarks_results/lr_results/mimic/outcome_readmission" \
    --outcome_var "outcome_readmission"
```
