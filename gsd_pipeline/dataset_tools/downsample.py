from skimage.transform import rescale, resize
import numpy as np
from gsd_pipeline.data_loader import load_saved_data, save_dataset


def downsample_dataset(data_dir, scale_factor: float, filename = 'data_set.npz'):

    (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params) = \
        load_saved_data(data_dir, filename)

    final_ct_inputs = []
    final_ct_lesion_GT = []
    final_brain_masks = []

    for subj_idx, id, in enumerate(ids):
        print('Scaling', subj_idx, id)
        final_ct_inputs.append(rescale(ct_inputs[subj_idx], (scale_factor, scale_factor, scale_factor, 1)))
        final_ct_lesion_GT.append(rescale(ct_lesion_GT[subj_idx], (scale_factor, scale_factor, scale_factor)))
        final_brain_masks.append(rescale(brain_masks[subj_idx], (scale_factor, scale_factor, scale_factor)))

    dataset = (clinical_inputs, np.array(final_ct_inputs), np.array(final_ct_lesion_GT),
               mri_inputs, mri_lesion_GT,
               np.array(final_brain_masks), ids, params)

    save_dataset(dataset, data_dir, f'scale{scale_factor}_' + filename)


def downsample_dataset_to_shape(data_dir, target_shape: tuple, filename = 'data_set.npz', n_c_ct = 4):
    (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params) = \
        load_saved_data(data_dir, filename)

    final_ct_inputs = []
    final_ct_lesion_GT = []
    final_brain_masks = []

    for subj_idx, id, in enumerate(ids):
        print('Scaling', subj_idx, id)
        final_ct_inputs.append(resize(ct_inputs[subj_idx], target_shape + (n_c_ct,)))
        final_ct_lesion_GT.append(resize(ct_lesion_GT[subj_idx], target_shape))
        final_brain_masks.append(resize(brain_masks[subj_idx], target_shape))

    dataset = (clinical_inputs, np.array(final_ct_inputs), np.array(final_ct_lesion_GT),
               mri_inputs, mri_lesion_GT,
               np.array(final_brain_masks), ids, params)

    save_dataset(dataset, data_dir, f'shape{target_shape[0]}x{target_shape[1]}x{target_shape[2]}_' + filename)


