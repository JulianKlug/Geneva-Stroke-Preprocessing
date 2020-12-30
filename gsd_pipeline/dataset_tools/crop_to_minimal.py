import numpy as np
import nibabel as nib
from gsd_pipeline.data_loader import load_saved_data, save_dataset
from gsd_pipeline.utils.utils import pad_to_shape
import nilearn.image as nilimg

def crop_to_minimal(data_dir, filename = 'data_set.npz'):
    (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params) = load_saved_data(data_dir, filename)

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

    final_ct_inputs = []
    final_ct_lesion_GT = []
    final_brain_masks = []

    for subj_index, ct_input in enumerate(ct_inputs):
        # Crop to minimal zero shape found above
        ct_input_img = nib.Nifti1Image(ct_input, np.eye(4))
        cropped_ct_input_img = nilimg.image._crop_img_to(ct_input_img, crop_offsets[subj_index])
        ct_lesion_img = nib.Nifti1Image(ct_lesion_GT[subj_index], np.eye(4))
        cropped_ct_lesion_img = nilimg.image._crop_img_to(ct_lesion_img, crop_offsets[subj_index])

        # pad minimal common shape (found from max shapes of individual subjects)
        padded_ct_input = pad_to_shape(cropped_ct_input_img.get_fdata(), minimal_common_shape)
        padded_ct_lesion = pad_to_shape(cropped_ct_lesion_img.get_fdata(), minimal_common_shape)
        padded_brain_mask = pad_to_shape(cropped_brain_masks[subj_index], minimal_common_shape)

        final_ct_inputs.append(padded_ct_input)
        final_ct_lesion_GT.append(padded_ct_lesion)
        final_brain_masks.append(padded_brain_mask)

    dataset = (clinical_inputs, np.array(final_ct_inputs), np.array(final_ct_lesion_GT), mri_inputs, mri_lesion_GT,
               np.array(final_brain_masks), ids, params)

    save_dataset(dataset, data_dir, 'comMin_' + filename)


crop_to_minimal('/home/klug/working_data/hd_perfusion_maps', filename='hd_pmaps_all_2016_2017_data_set.npz')

