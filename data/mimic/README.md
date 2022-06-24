# Example dataset generation (MIMIC-III)

1. [Complete MIMIC-III researcher requirements](https://mimic.mit.edu/docs/gettingstarted/)
   1. Become a credentialed user on PhysioNet. This involves completion of a training course in human subjects research.
   2. Sign the data use agreement (DUA). Adherence to the terms of the DUA is paramount.
   3. Follow the tutorials for direct cloud access (recommended), or download the data locally.
2. Download the [MIMIC-III dataset](https://mimic.mit.edu/docs/iii/) and place the required tables in a directory that will be referenced in steps 3 & 4 with the argument '--data_dir'.
   1. Required Tables
      1. ADMISSIONS.csv
      2. DIAGNOSES_ICD.csv
      3. PATIENTS.csv
      4. CPTEVENTS.csv
      5. DRGCODES.csv
3. Generate pretraining data with [src/convert_mimic.py](../../src/convert_mimic.py). You can use the TransMED environment detailed above for both steps.

    ```shell
    python src/convert_mimic.py \
        --random_seed 0 \
        --data_dir 'data/mimic' \
        --output_dir 'data/mimic' \
        --output_base_path 'mimic'
    ```

4. Generate finetuning data for readmission task with [src/create_mimic_ft_dataset.py](../../src/create_mimic_ft_dataset.py). You could edit this script for any outcome in MIMIC-III you want to target.

    ```shell
    python src/create_mimic_ft_dataset.py \
        --task "readmission" \
        --outcome_var "outcome_readmission" \
        --num_timesteps 2 \
        --data_dir 'data/mimic' \
        --base_path 'mimic' \
        --window_method 'first' \
        --lookahead 1
    ```
