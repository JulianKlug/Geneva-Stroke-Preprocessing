import os, hashlib, argparse
import pandas as pd
from unidecode import unidecode

def anonymise(data_path, output_dir=None, sheet = 'Sheet1', patient_identifying_columns=['Nom', 'Prénom', 'birth_date', 'id_hospital_case', 'EDS']):
    """
    Anonymise clinical data sheets (returns and saves anon data)
    :param data_path: path to clinical data sheet
    :param output_dir: directory to save anonymised data to
    :param sheet: sheet in excel file containing data
    :param patient_identifying_columns: columns containing identifiable data that should be dropped
    :return: dataframe with anonymised data
    """

    # Load clinical data spreadsheet
    patient_info_df = pd.read_excel(data_path, sheet_name=sheet)

    patient_info_df['combined_id'] = patient_info_df['Nom'].apply(lambda x : unidecode(str(x))).str.upper().str.strip() \
                                    + '^' + patient_info_df['Prénom'].apply(lambda x : unidecode(str(x))).str.upper().str.strip() \
                                   + '^' + patient_info_df['birth_date'].astype(str).str.split("-").str.join('')
    patient_info_df['hashed_id'] = ['subj-' + str(hashlib.sha256(str(item).encode('utf-8')).hexdigest()[:8]) for item in patient_info_df['combined_id']]
    patient_info_df = patient_info_df.drop(columns=['combined_id'])

    # reorganise column order
    reor_columns = [patient_info_df.columns[-1], *patient_info_df.columns[:-1]]
    patient_info_df = patient_info_df[reor_columns]

    patient_info_df = patient_info_df.drop(columns=patient_identifying_columns)

    if output_dir is None:
        output_dir = os.path.dirname(data_path)
    patient_info_df.to_excel(os.path.join(output_dir, f'anon_{os.path.basename(data_path)}'))

    return patient_info_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anonymise clinical data')
    parser.add_argument('data_path')
    parser.add_argument('-o', action="store", dest='output_dir', help='output directory')
    parser.add_argument('-s', action="store", dest='data_sheet', help='sheet in excel file', default='Sheet1')
    args = parser.parse_args()
    anonymise(args.data_path, args.output_dir, args.data_sheet)