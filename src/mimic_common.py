import pandas as pd
from datetime import datetime
import numpy as np
import sys
from mimic_mappings import get_mimic_mappings_external


def load_mimic_tables(data_dir, extension=".pkl", table_names=['ADMISSIONS', 'PATIENTS', 'CPTEVENTS', 'DIAGNOSES_ICD', 'PRESCRIPTIONS', 'PROCEDURES_ICD', 'ICUSTAYS']):
    if extension == 'csv':
        pd_read = pd.read_csv
    elif extension == 'pkl':
        pd_read = pd.read_pickle
    else:
        print(f"Unhandled extension: {extension}, exiting...")
        sys.exit(1)

    mimic_tables = dict()


    # Loading tables from mimic-III
    print("Loading tables from local file storage")

    for table_name in table_names:
        table_path = f"{data_dir}/{table_name}.{extension}"
        table_df = pd_read(table_path)
        mimic_tables[table_name] = table_df

    return mimic_tables


def transform_year(date):
    if date < datetime(2120, 1, 1):
        replacement_year = 2018
    elif (date >= datetime(2120, 1, 1)) and (date < datetime(2140, 1, 1)):
        replacement_year = 2019
    elif (date >= datetime(2140, 1, 1)) and (date < datetime(2160, 1, 1)):
        replacement_year = 2020
    elif (date >= datetime(2160, 1, 1)) and (date < datetime(2180, 1, 1)):
        replacement_year = 2021
    else:
        replacement_year = 2022
    try:
        transformed_date = date.replace(year=replacement_year)
    except ValueError:
        # For handling leap years, set to 2020 which is a leap year
        transformed_date = date.replace(year=2020)

    return transformed_date


def date_transform_df(df, date_fields, patient_offsets):
    date_format = "%Y-%m-%d %H:%M:%S"

    # calculate date distance (delta days)
    patient_offsets[["FIRST_VISIT", "FIRST_VISIT_TRANSFORMED"]] = patient_offsets[
        ["FIRST_VISIT", "FIRST_VISIT_TRANSFORMED"]
    ].apply(
        pd.to_datetime, format=date_format, errors="coerce"
    )  # if conversion required; errors='coerce' handles invalid values
    patient_offsets["delta_days"] = (
        patient_offsets["FIRST_VISIT"] - patient_offsets["FIRST_VISIT_TRANSFORMED"]
    ).dt.days
    patient_offsets = patient_offsets[["SUBJECT_ID", "delta_days"]]

    # join mimic-df with patients_offsets so that we match each delta-days to each subject-id
    df = df.merge(patient_offsets, on="SUBJECT_ID", how="inner")

    # offset all date fields by delta days amount
    for date_field in date_fields:
        df[date_field] = convert_date(df[date_field], df["delta_days"])
        df[date_field] = df[date_field].apply(
            lambda x: x.strftime(date_format) if (np.all(pd.notnull(x))) else None
        )
    return df


def convert_date(original_date, delta_days):
    date_format = "%Y-%m-%d %H:%M:%S"
    if original_date is None:
        return None
    else:
        converted_date = pd.to_datetime(
            original_date, format=date_format, errors="coerce"
        ) - pd.to_timedelta(delta_days, unit="d")
        return converted_date


def get_date_columns(mimic_key):
    date_fields = []
    # specific date columns for particular data
    if mimic_key == "ADMISSIONS":
        date_fields = ["ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"]
    elif mimic_key == "CALLOUT":
        date_fields = [
            "CREATETIME",
            "UPDATETIME",
            "ACKNOWLEDGETIME",
            "OUTCOMETIME",
            "FIRSTRESERVATIONTIME",
            "CURRENTRESERVATIONTIME",
        ]
    elif mimic_key == "CHARTEVENTS":
        date_fields = ["CHARTTIME", "STORETIME"]
    elif mimic_key == "CPTEVENTS":
        date_fields = ["CHARTDATE"]
    elif mimic_key == "DATETIMEEVENTS":
        date_fields = ["CHARTTIME", "STORETIME", "VALUE"]
    elif mimic_key == "ICUSTAYS":
        date_fields = ["INTIME", "OUTTIME"]
    elif mimic_key == "INPUTEVENTS_CV":
        date_fields = ["CHARTTIME", "STORETIME"]
    elif mimic_key == "INPUTEVENTS_MV":
        date_fields = ["STARTTIME", "ENDTIME", "STORETIME", "COMMENTS_DATE"]
    elif mimic_key == "LABEVENTS":
        date_fields = ["CHARTTIME"]
    elif mimic_key == "MICROBIOLOGYEVENTS":
        date_fields = ["CHARTDATE", "CHARTTIME"]
    elif mimic_key == "NOTEEVENTS":
        date_fields = ["CHARTDATE", "CHARTTIME", "STORETIME"]
    elif mimic_key == "OUTPUTEVENTS":
        date_fields = ["CHARTTIME", "STORETIME"]
    elif mimic_key == "PATIENTS":
        date_fields = ["DOB", "DOD", "DOD_HOSP", "DOD_SSN"]
    elif mimic_key == "PRESCRIPTIONS":
        date_fields = ["STARTDATE", "ENDDATE"]
    elif mimic_key == "PROCEDUREEVENTS_MV":
        date_fields = ["STARTTIME", "ENDTIME", "STORETIME", "COMMENTS_DATE"]
    elif mimic_key == "SERVICES":
        date_fields = ["TRANSFERTIME"]
    elif mimic_key == "TRANSFERS":
        date_fields = ["INTIME", "OUTTIME"]
    else:
        print(f"Unhandled index name: {mimic_key}, no date_fields set")
    return date_fields


def get_mimic_mappings(index_name, data):
    index_body, actions = get_mimic_mappings_external(index_name, data)
    return index_body, actions


def generate_patient_offsets(admissions_df):
    patient_offsets = []
    date_format = "%Y-%m-%d %H:%M:%S"
    # This is converting items to timestamps
    admissions_df["visit_start"] = pd.to_datetime(
        admissions_df["ADMITTIME"], format=date_format
    )
    admissions_df["visit_end"] = pd.to_datetime(
        admissions_df["DISCHTIME"], format=date_format
    )

    patient_admission_groups = admissions_df.groupby("SUBJECT_ID")
    for subject_id, group in patient_admission_groups:
        patient_first_admittime = group.visit_start.min()
        print()
        patient_first_admittime_transformed = transform_year(patient_first_admittime)
        patient_offset = (
            subject_id,
            patient_first_admittime,
            patient_first_admittime_transformed,
        )
        patient_offsets.append(patient_offset)

    patient_offsets_df = pd.DataFrame(
        patient_offsets,
        columns=["SUBJECT_ID", "FIRST_VISIT", "FIRST_VISIT_TRANSFORMED"],
    )
    return patient_offsets_df
