import os, pydicom
from shutil import copy

"""
Given a main directory, check if any subjects has a study named "study" that contains inaccurately named images
and copy them into the appropriate directory
"""

def find_matching_modality_folder(given_modality, given_date, given_time, search_dir):
    modalities = [o for o in os.listdir(search_dir)
                    if os.path.isdir(os.path.join(search_dir,o))]
    for modality in modalities:
        # don't put studies back into the same folder
        if modality == 'study':
            continue

        modality_dir = os.path.join(search_dir, modality)
        studies = [o for o in os.listdir(modality_dir)
                        if os.path.isdir(os.path.join(modality_dir,o))]
        first_study_dir = os.path.join(search_dir, modality, studies[0])
        dcms = [f for f in os.listdir(first_study_dir) if f.endswith(".dcm")]
        first_dcm = pydicom.dcmread(os.path.join(first_study_dir, dcms[0]))
        modality = first_dcm.Modality
        date = first_dcm.StudyDate
        time = first_dcm.StudyTime

        print(modality, given_modality, date, given_date, time, given_time)
        if modality == given_modality and date == given_date and time == given_time:
            return modality_dir
    return False

def extract_unknown_studies_folder(dir):
    subject_dir = os.path.dirname(dir)
    series = [o for o in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir,o))]

    for serie in series:
        dcms = [f for f in os.listdir(os.path.join(dir, serie)) if f.endswith(".dcm") and not f.startswith('.')]
        for dcm in dcms:
            print(dcm)
            dcm_data = pydicom.dcmread(os.path.join(dir, serie, dcm), force = True)
            modality = dcm_data.Modality
            date = dcm_data.StudyDate
            time = dcm_data.StudyTime
            serie_name = dcm_data.SeriesDescription
            print(dcm, modality, date, time, serie_name)
            modality_dir = find_matching_modality_folder(modality, date, time, subject_dir)
            if not modality_dir:
                modality_dir = os.path.join(subject_dir, modality + '_' + date)
                os.makedirs(modality_dir)
            new_serie_path = os.path.join(modality_dir, serie_name + '_' + time)
            if not os.path.exists(new_serie_path):
                os.makedirs(new_serie_path)
            new_file_path = os.path.join(new_serie_path, dcm)
            if not os.path.exists(new_file_path):
                copy(os.path.join(dir, serie, dcm), new_serie_path)

def extract_unknown_studies_folders_wrapper(main_dir):
    folders = [o for o in os.listdir(main_dir)
                    if os.path.isdir(os.path.join(main_dir,o))]
    for folder in folders:
        folder_dir = os.path.join(main_dir, folder)
        modalities = [o for o in os.listdir(folder_dir)
                        if os.path.isdir(os.path.join(folder_dir,o)) and o.startswith('study')]
        for modality in modalities:
            extract_unknown_studies_folder(os.path.join(folder_dir, modality))

subject_dir = '/Volumes/stroke_hdd1/stroke_db/2016/temp/test/Aeby Fabienne 19701111 473290/study'
extract_unknown_studies_folder(subject_dir)