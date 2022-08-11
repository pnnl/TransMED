"""
Script to create pretraining dataframe from mimic-III tables
"""

import argparse
import pandas as pd
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mimic_common import load_mimic_tables


def gather_patient_static(first_patient_admission, patient_patients):
    patient_static = dict()
    # Demographics
    # pid_vid aka subject_id

    patient_static["subject_id"] = patient_patients.SUBJECT_ID
    # race - No race information
    patient_static["race"] = None
    # ethnicity - use the first admission
    patient_static["ethnicity"] = first_patient_admission.ETHNICITY
    # sex - use the patients table
    patient_static["sex"] = patient_patients.GENDER
    # age - calculate after aggregating time
    patient_static["dob"] = patient_patients.DOB
    # expire_flag and death related variables
    patient_static["expire_flag"] = patient_patients.EXPIRE_FLAG
    patient_static["dod"] = patient_patients.DOD
    patient_static["dod_hosp"] = patient_patients.DOD_HOSP
    patient_static["dod_ssn"] = patient_patients.DOD_SSN
    # insurance - use the first admission
    patient_static["insurance"] = first_patient_admission.INSURANCE
    # language - use the first admission
    patient_static["language"] = first_patient_admission.LANGUAGE
    # religion - user the first admission
    patient_static["religion"] = first_patient_admission.RELIGION

    # cohort - not relevant
    return patient_static


def gather_patient_dynamic(
    patient_static,
    num_admissions,
    timestep,
    patient_admission,
    admissions_df,
    patients_df,
    diagnoses_df,
    procedures_df,
    prescriptions_df,
    icustays_df
):
    # hospital admission id for linking to visit tables
    hadm_id = patient_admission.HADM_ID

    # Get diagnoses codes for this visit
    admission_diagnoses = diagnoses_df[diagnoses_df.HADM_ID == hadm_id]
    admission_diagnoses = admission_diagnoses.sort_values(by="SEQ_NUM", ascending=True)
    admission_diagnoses_codes = admission_diagnoses.ICD9_CODE.tolist()

    # Get procedures codes for this visit
    admission_procedures = procedures_df[procedures_df.HADM_ID == hadm_id]
    admission_procedures = admission_procedures.sort_values(by="SEQ_NUM", ascending=True)
    admission_procedures_codes = admission_procedures.ICD9_CODE.tolist()
    # convert ints to strings
    admission_procedures_codes = [str(c) for c in admission_procedures_codes]

    # Get prescriptions for this visit
    admission_prescriptions = prescriptions_df[prescriptions_df.HADM_ID == hadm_id]
    admission_prescriptions_codes = admission_prescriptions.DRUG.tolist()
    #admission_prescriptions_codes2 = admission_prescriptions.DRUG_TYPE.tolist()
    #admission_prescriptions_codes3 = admission_prescriptions.DRUG_NAME_POE.tolist()
    #admission_prescriptions_codes4 = admission_prescriptions.DRUG_NAME_GENERIC.tolist()
    #admission_prescriptions_codes5 = admission_prescriptions.GSN.tolist()
    #admission_prescriptions_codes6 = admission_prescriptions.NDC.tolist()
    # this variable seemed like a useful generic code representation
    #admission_prescriptions_codes7 = admission_prescriptions.FORMULARY_DRUG_CD.tolist()

    patient_dynamic = dict()
    # dynamic demographics

    try:
        age = patient_admission.ADMITTIME - patient_static["dob"]
        days_in_year = 365.2425
        age_years = age.days / days_in_year
    except pd.errors.OutOfBoundsDatetime:
        # https://mimic.mit.edu/docs/iii/tables/patients/#dob
        # patients older than 89 years old at any time have date shifted to 300 years before first admission
        age_years = 90.0

    patient_dynamic["age"] = age_years

    # print(f"age_years: {age_years}, age: {patient_dynamic['age']}")
    patient_dynamic["marital_status"] = patient_admission.MARITAL_STATUS
    # Other admission information
    patient_dynamic["admission_type"] = patient_admission.ADMISSION_TYPE
    patient_dynamic["initial_diagnosis"] = patient_admission.DIAGNOSIS
    patient_dynamic["hospital_expire_flag"] = bool(patient_admission.HOSPITAL_EXPIRE_FLAG)
    # icu
    admission_icustays = icustays_df[icustays_df.HADM_ID == hadm_id]
    if not admission_icustays.empty:
        icustay_id = admission_icustays.ICUSTAY_ID.values[0]
    else:
        icustay_id = None
    patient_dynamic['icustay_id'] = icustay_id

    # dynamic codes
    patient_dynamic["conditions"] = admission_diagnoses_codes
    patient_dynamic["procedures"] = admission_procedures_codes
    patient_dynamic["drugs"] = admission_prescriptions_codes

    # empty for now
    patient_dynamic["measurements"] = []
    patient_dynamic["measurement_values"] = dict()

    # set up outcomes
    patient_dynamic["outcome_readmission"] = False
    if num_admissions > 1 and (timestep + 1) < num_admissions:
        # If a patient only has on admission they won't be readmitted
        patient_dynamic["outcome_readmission"] = True
    return patient_dynamic


def gather_patient_info(
    seed,
    num_patients,
    admissions_df,
    patients_df,
    diagnoses_df,
    procedures_df,
    prescriptions_df,
    icustays_df
):
    print(f"Gathering patient information...")
    # Set random seed for reproducibility
    random.seed(seed)

    oldssl_columns = [
        "pid_vid",
        "timestep",
        "race",
        "ethnicity",
        "sex",
        "age",
        "cohort",
        "outcome_mace",
        "outcome_ami",
        "outcome_stroke",
        "outcome_chd_death",
        "outcome_date_year",
        "conditions",
        "procedures",
        "drugs",
        "measurements",
        "measurement_values",
    ]

    ssl_columns_v1 = [
        "pid_vid",
        "hadm_id",
        "timestep",
        "race",
        "ethnicity",
        "sex",
        "age",
        "marital_status",
        "admission_type",
        "initial_diagnosis",
        "outcome_readmission",
        "conditions",
        "procedures",
        "drugs",
        "measurements",
        "measurement_values",
    ]

    ssl_columns = [
        "pid_vid",
        "hadm_id",
        "timestep",
        "race",
        "ethnicity",
        "sex",
        "insurance",
        "language",
        "religion",
        "expire_flag",
        "age",
        "marital_status",
        "admission_type",
        "initial_diagnosis",
        "hospital_expire_flag",
        "icustay_id",
        "outcome_readmission",
        "conditions",
        "procedures",
        "drugs",
        "measurements",
        "measurement_values",
    ]


    date_format = "%Y-%m-%d %H:%M:%S"
    admissions_df["ADMITTIME"] = pd.to_datetime(
        admissions_df["ADMITTIME"], format=date_format
    )
    patient_admission_groups = admissions_df.groupby("SUBJECT_ID")

    patients_df["DOB"] = pd.to_datetime(patients_df["DOB"], format=date_format)
    patient_admission_rows = []
    for s_i, (subject_id, group) in enumerate(tqdm(patient_admission_groups), start=1):
        # gather patient static

        patient_patients = patients_df.loc[patients_df.SUBJECT_ID == subject_id].iloc[0]

        patient_admissions = group.sort_values(by="ADMITTIME", ascending=True)
        num_admissions = len(patient_admissions)
        first_patient_admission = patient_admissions.iloc[0]

        # sort visits by admittime
        patient_static = gather_patient_static(
            first_patient_admission, patient_patients
        )
        for timestep, patient_admission in enumerate(
            patient_admissions.itertuples(index=False)
        ):
            hadm_id = patient_admission.HADM_ID

            patient_dynamic = gather_patient_dynamic(
                patient_static,
                num_admissions,
                timestep,
                patient_admission,
                admissions_df,
                patients_df,
                diagnoses_df,
                procedures_df,
                prescriptions_df,
                icustays_df
            )

            patient_admission_row = (
                subject_id,
                hadm_id,
                timestep,
                patient_static["race"],
                patient_static["ethnicity"],
                patient_static["sex"],
                patient_static["insurance"],
                patient_static["language"],
                patient_static["religion"],
                patient_static["expire_flag"],
                patient_dynamic["age"],
                patient_dynamic["marital_status"],
                patient_dynamic["admission_type"],
                patient_dynamic["initial_diagnosis"],
                patient_dynamic["hospital_expire_flag"],
                patient_dynamic["icustay_id"],
                patient_dynamic["outcome_readmission"],
                patient_dynamic["conditions"],
                patient_dynamic["procedures"],
                patient_dynamic["drugs"],
                patient_dynamic["measurements"],
                patient_dynamic["measurement_values"],
            )
            patient_admission_rows.append(patient_admission_row)

        # Only process a certain number of patients
        if num_patients and s_i == num_patients:
            print(f"Breaking after processing {num_patients} patients")
            break

    model_input = pd.DataFrame(patient_admission_rows, columns=ssl_columns)
    return model_input


def save_patient_info(output_dir, output_base_path, patient_info):
    print(f"Number of patients: {patient_info.pid_vid.nunique()}")
    print(f"patient_info.shape: {patient_info.shape}")
    output_base_path = f"{output_dir}/{output_base_path}"
    pkl_path = f"{output_base_path}.pkl"
    print(f"Saving patient info to {pkl_path}")
    patient_info.to_pickle(pkl_path)

    csv_path = f"{output_base_path}.csv"
    print(f"Saving a copy of patient info as csv for debug to {csv_path}")
    patient_info.to_csv(csv_path, index=False)


def create_mimic_pt(seed, data_dir, base_path, num_patients, extension):

    # Load original data tables
    mimic_tables = load_mimic_tables(data_dir, extension)

    print(f"Making sure output directory {data_dir} exists")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    patient_info = gather_patient_info(
        seed,
        num_patients,
        mimic_tables['ADMISSIONS'],
        mimic_tables['PATIENTS'],
        mimic_tables['DIAGNOSES_ICD'],
        mimic_tables['PROCEDURES_ICD'],
        mimic_tables['PRESCRIPTIONS'],
        mimic_tables['ICUSTAYS']
    )


    # deal with null values
    # race - already all Nones
    patient_info.race = patient_info.race.fillna('')
    # language
    patient_info.language = patient_info.language.fillna('')
    # religion
    patient_info.religion = patient_info.religion.fillna('')
    # marital_status
    patient_info.marital_status = patient_info.marital_status.fillna('')
    # initial_diagnosis
    patient_info.initial_diagnosis = patient_info.initial_diagnosis.fillna('')
    # icustay_id
    patient_info.icustay_id = patient_info.icustay_id.replace({np.nan: None})

    save_patient_info(data_dir, base_path, patient_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=3, help="Consistent seed for RNG")
    parser.add_argument("--data_dir", type=str, help="Input and output data directory")
    parser.add_argument(
        "--base_path",
        type=str,
        default="mimic",
        help="Base path for naming output files",
    )
    parser.add_argument(
        "--num_patients", type=int, help="Only process this many patients then break"
    )
    parser.add_argument("--extension", type=str, help='File extension: csv | pkl', default='csv')
    args = parser.parse_args()
    print(f"{args}", flush=True)
    create_mimic_pt(args.random_seed, args.data_dir, args.base_path, args.num_patients, args.extension)
