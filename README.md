<h1 align="center">
    TransMED
</h1>
<h4 align="center">Transfer Learning From Existing Diseases Via Hierarchical Multi-Modal BERT Models</h4>

Paper dataset details
------

Our study is based on de-identified EHR data of all patients treated at Stanford Hospital, between January 1, 2015 and March 19, 2021. This dataset was provided via [STAnford medicine Research data Repository (STARR)](https://starr.stanford.edu/) and was used under approval by Stanford University Institutional Review Board (IRB) protocol: 50033 (Machine Learning of Electronic Medical Records for Precision Medicine. Patient informed consent was waived by Stanford University Institutional Review Board (IRB) for this protocol. All methods were carried out in accordance with relevant guidelines and regulations. We cannot share this data publicly, so in order to facilitate code use on publicly available datasets we provide two scripts to process and generate a sample input for TransMED based on the MIMIC-III dataset.

Data format details
------

The dataset format expected for the pretraining data is indexed by 'pid_vid' which is a combination of patient id and visit id as well as by 'timestep' which is absolute timestep for the entire patient trajectory. Multiple modalities of data are accepted and can be switched on or off with the 'features' argument. Static features including demographics and risk factors may be specified and they will be used in the finetuning phase. Dynamic features including diagnosis codes (dx), prescriptions (rx), procedure codes (pr) and lab measurement (mea) codes will be tokenized and used in both the pretraining and finetuning stages of the model, dynamic features that have continous values such as lab measurements (val) will be used exclusively in the finetuning phase and must be numeric. Detailed instructions on how to generate a dataset from MIMIC-III are included in [data/mimic/README.md](data/mimic/README.md) but this data format is very flexible and you can format your own data targeting various outcomes. The script [src/create_dataset.py](src/create_dataset.py) can be modified and generally used to format finetuning data.

Pretraining data format
------

- How to load pretraining dataframe

  - ```python
    import pandas as pd
    data = pd.read_pickle('mimic.pkl')
    ```

- data column names (Bold column names are always required. Codes including conditions, procedures, drugs, measurements, and measurement values are only required if that feature is selected. Codes are usually medical codes but arbitrary symbols are handled. Race, ethnicity, sex, and age are currently supported demographic features.)
  - **pid_vid**: int, patient ID or patient ID and visit ID combination
  - hadm_id: int, hospital admission id
  - **timestep**: int, absolute timestep in patient trajectory
  - race: string, description of race
  - ethnicity: string, description of ethnicity
  - sex: string, description of sex
  - age: float, years old
  - marital_status: string, description of marital status (not currently handled)
  - admission_type: string, type of admission (not currently handled)
  - initial_diagnosis: string, preliminary free text diagnosis for the patient on hospital admission (not currently handled)
  - outcome_readmission: bool, True if the patient came back for another admission after this visit
  - conditions: List[obj], ICD diagnoses for patients, most notably ICD-9 diagnoses
  - procedures: List[obj], Procedure codes for patients
  - drugs: List[obj], Drug codes for patients
  - measurements: List[obj], Measurement codes for patients
  - measurement_values: Dict[obj, float], Measurement code keys with measurement values for the visit

Finetuning data format
------

- How to load finetuning data objects

  - ```python
    import pickle as pkl
    data, windows = pkl.load(open('readmission_mimic_lookahead1_numts2.pkl', 'rb'))
    ```

- data column names (Same columns used in pretraining described above, although the data may be a subset of the pretraining dataframe based on the task and outcome. For this example dataset, any patients with less than two visits were dropped for the finetuning data.).

- windows column names (Bold column names are always required. You must include a target outcome variable, for this example dataset, 'outcome_readmission' is our target.)
  - **pid_vid**: int, patient ID or patient ID and visit ID combination
  - **input_timesteps**: List[int], List of input timesteps for this finetuning sample
  - outcome_readmission: bool, True if the patient came back for another admission after this visit

Install instructions
------

- Python 3.8
- Dependencies:
  - PyTorch 1.7.1 w/ CUDA 9.2
- GPU install

```shell
conda create -n transmed python=3.8
conda activate transmed
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r environment/transmed/transmed.gpu.requirements.txt
```

- CPU install

```shell
conda create -n transmed python=3.8
conda activate transmed
pip install -r environment/transmed/transmed.cpu.requirements.txt
```

TransMED Training
------

Our model has two stages, first we run pretraining, and then we run finetuning. Only two training epochs are specified for testing the installation in a quick manner. In our experiements we usually ran pretraining and finetuning about 500 epochs for with a patience of 15 and a patience threshold of 0.0001. You can run the model using the statements below or use the script [run.sh](run.sh).

Pretraining
------

```shell
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
```

Finetuning
------

```shell
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
```

Benchmarks information
------

A patient split must be generated by running TransMED finetuning before being used in the benchmarks. See [benchmarks/README.md](benchmarks/README.md) for installation and training instructions.

Citation
------

Please cite the following paper if you use this code in your work.

```bibtex
@inproceedings{agarwal22NSR,
  title={Preparing For The Next Pandemic via Transfer Learning From Existing Diseases with Hierarchical Multi-Modal BERT: A Study on COVID-19 Outcome Prediction},
  author={Agarwal, Khushbu and Choudhury, Sutanay and Tipirneni, Sindhu and Mukherjee, Pritam and Ham, Colby and Tamang, Suzanne and Baker, Matthew and Tang, Siyi and Kocaman, Veysel and Gevaert, Olivier and Rallo, Robert and Reddy, Chandan},
  booktitle={Nature Scientific Reports 2022},
  year={2022}
}
```

License
------

Released under the Simplified BSD license (see [LICENSE.md](LICENSE.md))
