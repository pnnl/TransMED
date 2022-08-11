# Example dataset generation (MIMIC-III)

1. [Complete MIMIC-III researcher requirements](https://mimic.mit.edu/docs/gettingstarted/)
   1. Become a credentialed user on PhysioNet. This involves completion of a training course in human subjects research.
   2. Sign the data use agreement (DUA). Adherence to the terms of the DUA is paramount.
   3. Follow the tutorials for direct cloud access (recommended), or download the data locally.
2. Download the [MIMIC-III dataset](https://mimic.mit.edu/docs/iii/) and place the required tables in a directory that will be referenced in step 3 with the argument '--data_dir'.
   1. Required Tables
      1. ADMISSIONS.csv
      2. CPTEVENTS.csv
      3. DIAGNOSES_ICD.csv
      4. ICUSTAYS.csv
      5. PATIENTS.csv
      6. PRESCRIPTIONS.csv
      7. PROCEDURES_ICD.csv
3. Generate pretraining and finetuning data with [src/create_mimic_dataset.py](../../src/create_mimic_dataset.py). You can use the TransMED environment detailed above.

    ```shell
    python src/create_mimic_dataset.py \
        --data_dir 'data/mimic'
    ```
