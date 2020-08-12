import os
import numpy as np
import argparse
from gsd_pipeline import data_loader as dl
from gsprep.utils.smoothing import gaussian_smoothing
from gsprep.tools.perfusion_maps_tools.normalisation import normalise_by_contralateral_median
from gsprep.tools.segmentation.ct_brain_extraction import ct_brain_extraction
import scipy.ndimage.morphology as ndimage
from skimage.morphology import ball, disk


def add_core_map(ct_dataset:[str, np.ndarray], masks = None, cbf_channel = 1, ncct_channel = 4, outfile = None, one_hot_encode_core=False, dilation_dimension=3):

    if int(dilation_dimension) == 2:
        dilation_structure = dilation_structure_2d
    elif int(dilation_dimension) == 3:
        dilation_structure = dilation_structure_3d
    else:
        raise NotImplementedError

    if isinstance(ct_dataset, str):
        data_dir = os.path.dirname(ct_dataset)
        file_name = os.path.basename(ct_dataset)
        data = dl.load_saved_data(data_dir, file_name)
        clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params = data
        if masks.size <= 1:
            masks = None
    else:
        ct_inputs = ct_dataset

    n_subj, n_x, n_y, n_z, n_c = ct_inputs.shape

    # Create CSF mask
    low_bounded_ncct = ct_inputs[..., ncct_channel][ct_inputs[..., ncct_channel] > 0]
    up_and_low_bounded_ncct = low_bounded_ncct[low_bounded_ncct < 100]
    # threshold = 20
    threshold = np.percentile(up_and_low_bounded_ncct, 5)
    csf_mask = gaussian_smoothing(ct_inputs[..., ncct_channel, None], kernel_width=3) < threshold
    enlarged_csf_mask = np.array(
        [ndimage.binary_dilation(csf_mask[idx, ..., 0], structure=dilation_structure(2)) for idx in range(csf_mask.shape[0])])
    inv_csf_mask = -1 * enlarged_csf_mask + 1

    # Create Skull mask
    brain_mask = np.array([ct_brain_extraction(ct_inputs[subj, ..., ncct_channel], fsl_path='/usr/local/fsl/bin')[0]
                           for subj in range(n_subj)])
    not_brain_mask = 1 - brain_mask
    # enlargen slighlty
    enlarged_not_brain_mask = np.array(
        [ndimage.binary_dilation(not_brain_mask[subj], dilation_structure(3)) for subj in range(n_subj)])
    inv_skull_mask = 1 - enlarged_not_brain_mask

    ## Create major vessel mask
    threshold = np.percentile(ct_inputs[..., cbf_channel], 99)
    vessel_mask = ct_inputs[..., cbf_channel] > threshold
    enlarged_vessel_mask = np.array(
        [ndimage.binary_dilation(vessel_mask[idx], structure=dilation_structure(2)) for idx in range(vessel_mask.shape[0])])
    vessel_mask = enlarged_vessel_mask
    inv_vessel_mask = -1 * vessel_mask + 1

    ## Create Core mask
    smooth_rCBF = normalise_by_contralateral_median(gaussian_smoothing(ct_inputs[..., 1, None], kernel_width=2))
    smooth_core_masks = smooth_rCBF < 0.38
    corr_csf_core_masks = smooth_core_masks * inv_csf_mask[..., None]
    corr_vx_core_masks = corr_csf_core_masks * inv_vessel_mask[..., None]
    corr_skull_core_masks = corr_vx_core_masks * inv_skull_mask[..., None]

    if masks is not None:
        # Restrict to defined prior mask
        restr_core = corr_skull_core_masks * masks[..., None]
    else:
        restr_core = corr_skull_core_masks

    if one_hot_encode_core:
        class_0_core = 1 - restr_core
        class_1_core = restr_core
        restr_core = np.concatenate((class_0_core, class_1_core), axis=-1)

    ct_inputs = np.concatenate((ct_inputs, restr_core), axis=-1)

    if isinstance(ct_dataset, str):
        if outfile is None:
            if one_hot_encode_core:
                outfile = os.path.basename(ct_dataset).split('.')[0] + '_with_one_hot_encoded_core.npz'
            else:
                outfile = os.path.basename(ct_dataset).split('.')[0] + '_with_core.npz'
        params = params.item()
        if one_hot_encode_core:
            params['ct_sequences'].append('core_rCBF_0.38_class0')
            params['ct_sequences'].append('core_rCBF_0.38_class1')
        else:
            params['ct_sequences'].append('core_rCBF_0.38')
        dataset = (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params)
        dl.save_dataset(dataset, os.path.dirname(ct_dataset), out_file_name=outfile)

    return ct_inputs, restr_core

def dilation_structure_3d(radius):
    return ball(radius)

def dilation_structure_2d(radius):
    return np.expand_dims(disk(radius), axis=-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add core segmentation based on rCBF to the perfusion map dataset')
    parser.add_argument('data_path')
    parser.add_argument('-o', '--outfile',  help='Name of output file', required=False, default=None)
    parser.add_argument('-f', '--cbf_channel',  help='CBF Channel in dataset', required=False, default=1)
    parser.add_argument('-nc', '--ncct_channel',  help='Non contrast CT Channel in dataset', required=False, default=4)
    parser.add_argument('-oh', '--one_hot_encode', action='store_true', default=False, required=False)
    parser.add_argument('-d', '--dilation_dimension', help='Dimension to perform dilations over (for thin datasets, 2 is preferred)', required=False, default=3)


    args = parser.parse_args()
    add_core_map(args.data_path, outfile=args.outfile, cbf_channel=args.cbf_channel, ncct_channel=args.ncct_channel, one_hot_encode_core=args.one_hot_encode, dilation_dimension=args.dilation_dimension)

