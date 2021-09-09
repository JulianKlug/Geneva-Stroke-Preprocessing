import os, hashlib
import pandas as pd
import numpy as np
from unidecode import unidecode
from difflib import get_close_matches, SequenceMatcher

default_outcomes = [
    # immediate outcomes
    'NIH_24h',
    'NIH sortie',
    'mRs sortie',
    'duration_hospital',
    'death_hospital',
    # immediate complications
    'decompression',
    'ich_sympt_hospital',
    'stroke_hospital',
    # long term outcomes
    'mRS_90days',
    'death_followup'
]

def extract_clinical_outcomes(patient_id_path, patient_info_path, id_sheet = 'Sheet1', info_sheet = 'Sheet1', anonymise=True, selected_outcomes = default_outcomes):
    """
    """
    # Load spreadsheet
    patient_info_xl = pd.ExcelFile(patient_info_path)
    patient_id_xl = pd.ExcelFile(patient_id_path)
    # Load a sheet into a DataFrame
    patient_info_df = patient_info_xl.parse(info_sheet)
    patient_id_df = patient_id_xl.parse(id_sheet)

    patient_info_df['combined_id'] = patient_info_df['Nom'].apply(lambda x : unidecode(str(x))).str.upper().str.strip() \
                                    + '^' + patient_info_df['Prénom'].apply(lambda x : unidecode(str(x))).str.upper().str.strip() \
                                   + '^' + patient_info_df['birth_date'].astype(str).str.split("-").str.join('')
    patient_info_df['hashed_id'] = ['subj-' + str(hashlib.sha256(str(item).encode('utf-8')).hexdigest()[:8]) for item in patient_info_df['combined_id']]

    match_list = [get_close_matches(item, patient_info_df['combined_id'], 1) for item in patient_id_df['patient_identifier']]
    patient_id_df['combined_id'] = [matches[0] if matches else np.NAN for matches in match_list]
    patient_id_df['combined_id_match'] = [SequenceMatcher(None, row['patient_identifier'], row['combined_id']).ratio()
                                          if not row.isnull().values.any() else np.nan
                                          for index, row in patient_id_df.iterrows()]

    output_df = patient_id_df.merge(patient_info_df, how='left', left_on='combined_id', right_on='combined_id')

    output_df['pid'] = patient_id_df['anonymised_id']

    output_df = output_df[
        ['patient_identifier', 'combined_id', 'combined_id_match', 'Nom', 'Prénom', 'birth_date', 'anonymised_id', 'pid', 'id_hospital_case', 'onset_time']
        + selected_outcomes
    ]

    output_df = output_df.drop_duplicates(subset='combined_id')

    outfile_name = 'extracted_clinical_outcomes_' + os.path.basename(patient_info_path)

    if anonymise:
        output_df = output_df[['pid'] + selected_outcomes]
        outfile_name = 'anon_' + outfile_name

    output_df.to_excel(os.path.join(os.path.dirname(patient_info_path), outfile_name))
#
# extract_clinical_outcomes(
#     '/Users/jk1/temp/clinical_data_prepro/anonymisation_key_pCT_2016_2017.xlsx',
#     '/Users/jk1/temp/clinical_data_prepro/190419_Données 2015-16-17.xlsx', id_sheet = 'Sheet1', info_sheet = 'Sheet1 (2)',
#     anonymise=False)