import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler


def find_max_shape(data_dir, file_name):
    '''
    Given a directory and a filename, find the biggest dimension along x, y and z
    :param data_dir:
    :param file_name: in which file to look for dimensions
    :return:
    '''
    max_x = 0; max_y = 0; max_z = 0;
    subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, o))]

    for subject in subjects:
        print(subject)
        subject_dir = os.path.join(data_dir, subject)
        modalities = [o for o in os.listdir(subject_dir)
                      if os.path.isdir(os.path.join(subject_dir, o))]

        for modality in modalities:
            modality_dir = os.path.join(subject_dir, modality)
            studies = [o for o in os.listdir(modality_dir)
                       if os.path.isfile(os.path.join(modality_dir, o))]

            for study in studies:
                study_path = os.path.join(modality_dir, study)
                if study.startswith(file_name):
                    img = nib.load(study_path)
                    data = img.get_data()
                    if data.shape[0] > max_x: max_x = data.shape[0]
                    if data.shape[1] > max_y: max_y = data.shape[1]
                    if data.shape[2] > max_z: max_z = data.shape[2]

    return (max_x, max_y, max_z)


def rescale_outliers(imgX, MASKS):
    '''
    Rescale outliers as some images from RAPID seem to be scaled x10
    Outliers are detected if their median exceeds 5 times the global median and are rescaled by dividing through 10
    :param imgX: image data (n, x, y, z, c)
    :return: rescaled_imgX
    '''

    for i in range(imgX.shape[0]):
        for channel in range(imgX.shape[-1]):
            median_channel = np.median(imgX[..., channel][MASKS])
            if np.median(imgX[i, ..., 0][MASKS[i]]) > 5 * median_channel:
                imgX[i, ..., 0] = imgX[i, ..., channel] / 10

    return imgX


def standardise(imgX, clinX):
    original_shape = imgX.shape
    imgX = imgX.reshape(-1, imgX.shape[-1])
    scaler = StandardScaler(copy = False)
    rescaled_imgX = scaler.fit_transform(imgX).reshape(original_shape)
    if clinX is not None:
        rescaled_clinX = scaler.fit_transform(clinX)
    else:
        rescaled_clinX = clinX
    return rescaled_imgX, rescaled_clinX