import os
import numpy as np
from gsd_pipeline import data_loader as dl
from gsd_pipeline.data_loader import save_dataset


def mirror_dataset_images(dataset_path:str):

    data_dir = os.path.dirname(dataset_path)
    file_name = os.path.basename(dataset_path)
    data = dl.load_saved_data(data_dir, file_name)
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params = data

    # flip along x axis
    flipped_ct_inputs = np.flip(ct_inputs, axis=1)
    flipped_ct_lesion_GT = np.flip(ct_lesion_GT, axis=1)
    flipped_masks = np.flip(masks, axis=1)

    flipped_ids = ['flipped_' + id for id in ids]

    augmented_ct_inputs =  np.concatenate((ct_inputs, flipped_ct_inputs), axis=0)
    augmented_ct_lesion_GT =  np.concatenate((ct_lesion_GT, flipped_ct_lesion_GT), axis=0)
    augmented_masks =  np.concatenate((masks, flipped_masks), axis=0)
    augmented_ids = np.concatenate((ids, flipped_ids), axis=0)

    if len(mri_inputs) == 0:
        augmented_mri_inputs = []
        augmented_mri_lesion_GT = []
    else:
        flipped_mri_inputs = np.flip(mri_inputs, axis=1)
        flipped_mri_lesion_GT = np.flip(mri_lesion_GT, axis=1)
        augmented_mri_inputs = np.concatenate((mri_inputs, flipped_mri_inputs), axis=0)
        augmented_mri_lesion_GT = np.concatenate((mri_lesion_GT, flipped_mri_lesion_GT), axis=0)

    augmented_dataset = (clinical_inputs, augmented_ct_inputs, augmented_ct_lesion_GT,
               augmented_mri_inputs, augmented_mri_lesion_GT,
               augmented_masks, augmented_ids, params)

    save_dataset(augmented_dataset, data_dir, 'flipped_' + file_name)