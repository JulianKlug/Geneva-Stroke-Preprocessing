import os
import numpy as np
from gsd_pipeline import data_loader as dl



def add_noise_to_channel(ct_dataset:[str, np.ndarray], channel_index=5, outfile=None):

    if isinstance(ct_dataset, str):
        data_dir = os.path.dirname(ct_dataset)
        file_name = os.path.basename(ct_dataset)
        data = dl.load_saved_data(data_dir, file_name)
        clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params = data
        if masks.size <= 1:
            masks = None
    else:
        ct_inputs = ct_dataset

    noise = np.random.choice([0, 1], ct_inputs[..., channel_index].shape, p=[0.998, 0.002])
    noised_ct_inputs = ct_inputs
    noised_ct_inputs[..., channel_index] += noise
    noised_ct_inputs[..., channel_index][noised_ct_inputs[..., channel_index] == 2] = 1

    if isinstance(ct_dataset, str):
        if outfile is None:
            outfile = os.path.basename(ct_dataset).split('.')[0] + f'_noisy{channel_index}.npz'
        params = params.item()

        dataset = (clinical_inputs, noised_ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params)
        dl.save_dataset(dataset, os.path.dirname(ct_dataset), out_file_name=outfile)

    return ct_inputs

