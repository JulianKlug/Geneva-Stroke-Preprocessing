import os
import shutil
import tempfile

import numpy as np
import nibabel as nib
from gsd_pipeline.data_loader import save_dataset, load_saved_data
from gsd_pipeline.utils.utils import pad_to_shape
import nilearn.image as nilimg

def find_minimal_common_shape(data_dir, filename = 'data_set.npz'):
    brain_masks = np.load(os.path.join(data_dir, filename), allow_pickle=True)['brain_masks']

    cropped_brain_masks = []
    crop_offsets = []
    max_x, max_y, max_z = 0, 0, 0
    for mask in brain_masks:
        dummy_affine = np.eye(4)
        mask_img = nib.Nifti1Image(mask.astype(int), affine=dummy_affine)
        # crop to minimal non zero
        cropped_mask, crop_offset = nilimg.image.crop_img(mask_img, return_offset=True)
        cropped_mask_data = cropped_mask.get_fdata()
        cropped_brain_masks.append(cropped_mask_data)
        crop_offsets.append(crop_offset)
        # find maximum shape for padding later
        if cropped_mask_data.shape[0] > max_x: max_x = cropped_mask_data.shape[0]
        if cropped_mask_data.shape[1] > max_y: max_y = cropped_mask_data.shape[1]
        if cropped_mask_data.shape[2] > max_z: max_z = cropped_mask_data.shape[2]

    minimal_common_shape = (max_x, max_y, max_z)
    print('Found minimal common shape', minimal_common_shape)
    return minimal_common_shape, crop_offsets

def group_to_file(minimal_common_shape, data_dir, temp_dir, filename = 'data_set.npz', n_c = 4):
    ids = np.load(os.path.join(data_dir, filename), allow_pickle=True)['ids']
    clinical_inputs = np.load(os.path.join(data_dir, filename), allow_pickle=True)['clinical_inputs']
    params = np.load(os.path.join(data_dir, filename), allow_pickle=True)['params']
    mri_inputs = np.load(os.path.join(data_dir, filename), allow_pickle=True)['mri_inputs']
    mri_lesion_GT = np.load(os.path.join(data_dir, filename), allow_pickle=True)['mri_lesion_GT']

    n_x, n_y, n_z = minimal_common_shape
    final_ct_inputs = np.empty((len(ids), n_x, n_y, n_z, n_c))
    final_ct_lesion_GT = np.empty((len(ids), n_x, n_y, n_z))
    final_brain_masks = np.empty((len(ids), n_x, n_y, n_z), dtype=bool)

    for subj_index, id in enumerate(ids):
        print('Loading', subj_index, id)

        load_padded_ct_input = nib.load(os.path.join(temp_dir, f'{id}_padded_ct_input.nii')).get_fdata()
        load_padded_ct_lesion = nib.load(os.path.join(temp_dir, f'{id}_padded_ct_lesion.nii')).get_fdata()
        load_padded_brain_mask = nib.load(os.path.join(temp_dir, f'{id}_padded_brain_mask.nii')).get_fdata()

        final_ct_inputs[subj_index] = load_padded_ct_input
        final_ct_lesion_GT[subj_index] = load_padded_ct_lesion
        final_brain_masks[subj_index] = load_padded_brain_mask

    dataset = (clinical_inputs, np.array(final_ct_inputs), np.array(final_ct_lesion_GT),
               mri_inputs, mri_lesion_GT,
               np.array(final_brain_masks), ids, params)

    save_dataset(dataset, data_dir, 'comMin_' + filename)

def crop_to_minimal(data_dir, filename = 'data_set.npz', n_c=4):
    """
    Crop all images of dataset to a common minimal shape and saves new dataset
    :param data_dir: dir containing dataset file
    :param filename: dataset filename
    :param n_c: number of channels in CT image
    :return:
    """

    print('Cropping to Minimal Common shape')
    print('WARNING: MRI files are not cropped for now.')

    minimal_common_shape, crop_offsets = find_minimal_common_shape(data_dir, filename)

    temp_dir = tempfile.mkdtemp()
    print('Using temporary dir', temp_dir)

    (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params) = load_saved_data(data_dir, filename)

    for subj_index, id in enumerate(ids):
        print('Processing', subj_index, id)

        # Crop to minimal zero shape found above
        ct_input_img = nib.Nifti1Image(ct_inputs[subj_index], np.eye(4))
        cropped_ct_input_img = nilimg.image._crop_img_to(ct_input_img, crop_offsets[subj_index])
        ct_lesion_img = nib.Nifti1Image(ct_lesion_GT[subj_index], np.eye(4))
        cropped_ct_lesion_img = nilimg.image._crop_img_to(ct_lesion_img, crop_offsets[subj_index])
        mask_img = nib.Nifti1Image(brain_masks[subj_index].astype(int), np.eye(4))
        cropped_mask_img = nilimg.image._crop_img_to(mask_img, crop_offsets[subj_index])

        # pad minimal common shape (found from max shapes of individual subjects)
        padded_ct_input = pad_to_shape(cropped_ct_input_img.get_fdata(), minimal_common_shape)
        padded_ct_lesion = pad_to_shape(cropped_ct_lesion_img.get_fdata(), minimal_common_shape)
        padded_brain_mask = pad_to_shape(cropped_mask_img.get_fdata(), minimal_common_shape)

        padded_ct_input_img = nib.Nifti1Image(padded_ct_input, np.eye(4))
        padded_ct_lesion_img = nib.Nifti1Image(padded_ct_lesion, np.eye(4))
        padded_brain_mask_img = nib.Nifti1Image(padded_brain_mask, np.eye(4))

        nib.save(padded_ct_input_img, os.path.join(temp_dir, f'{id}_padded_ct_input.nii'))
        nib.save(padded_ct_lesion_img, os.path.join(temp_dir, f'{id}_padded_ct_lesion.nii'))
        nib.save(padded_brain_mask_img, os.path.join(temp_dir, f'{id}_padded_brain_mask.nii'))

    print('Processing done. Now loading files.')

    group_to_file(minimal_common_shape, data_dir, temp_dir, filename, n_c)

    shutil.rmtree(temp_dir)
