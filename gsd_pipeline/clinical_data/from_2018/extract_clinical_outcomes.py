import os, hashlib
import pandas as pd
import numpy as np
from unidecode import unidecode
from difflib import get_close_matches, SequenceMatcher

default_outcomes = [
    # immediate outcomes
    'NIHSS 24h',
    'Duration of hospital stay (days)',
    'Death in hospital',
    # immediate complications
    'Decompr. craniectomy',
    'Symptomatic ICH',
    'Recurrent stroke',
    # long term outcomes
    '3M mRS',
    '3M Death'
]


def extract_clinical_outcomes(patient_id_path, patient_info_path, id_sheet = 'Sheet1', info_sheet = 'Export cases registered in.', anonymise=True):
    """
    Example use:
    extract_clinical_outcomes(
    './path',
    './path', id_sheet = 'Sheet1', info_sheet = 'Sheet1',
    anonymise=False)
    """
    print('WARNING: Remove header and blank lines from patient info file first.')

    # Load spreadsheet
    patient_info_xl = pd.ExcelFile(patient_info_path)
    patient_id_xl = pd.ExcelFile(patient_id_path)
    # Load a sheet into a DataFrame
    patient_info_df = patient_info_xl.parse(info_sheet)
    patient_id_df = patient_id_xl.parse(id_sheet)

    patient_info_df['hospital_id'] = patient_info_df['Case ID'].apply(lambda x: str(x)[8:-4])
    patient_info_df['hospital_id'] = patient_info_df['hospital_id'].astype(int)
    patient_id_df['hospital_id'] = patient_id_df['hospital_id'].astype(int)

    output_df = patient_id_df.merge(patient_info_df, how='left', left_on='hospital_id', right_on='hospital_id')

    output_df = output_df[
        ['pid', 'hospital_id', 'first_name','last_name', 'dob', 'Onset time']
        + default_outcomes
    ]


    outfile_name = 'extracted_clinical_outcomes_' + os.path.basename(patient_info_path)

    if anonymise:
        output_df = output_df[['pid'] + default_outcomes]
        outfile_name = 'anon_' + outfile_name

    output_df.to_excel(os.path.join(os.path.dirname(patient_info_path), outfile_name))
    print('Output may contain duplicates, please remove them manually as not all duplicate entries are the same.')

