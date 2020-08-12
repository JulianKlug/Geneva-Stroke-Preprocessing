import os
import argparse
import numpy as np
from gsd_pipeline.utils.utils import rescale_outliers
from gsd_pipeline.data_loader import load_saved_data

def standardize_data(data_dir, filename = 'data_set.npz', channels_to_leave_out = [], outlier_scaling=True, min_max_scaling=True):
    (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params) = load_saved_data(data_dir, filename)

    standardize = lambda x: (x - np.mean(x)) / np.std(x)
    scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    if outlier_scaling:
        print('Before outlier scaling', np.mean(ct_inputs[..., 0]), np.std(ct_inputs[..., 0]))
        # correct for outliers that are scaled x10
        ct_inputs = rescale_outliers(ct_inputs, brain_masks)
        print('After outlier scaling', np.mean(ct_inputs[..., 0]), np.std(ct_inputs[..., 0]))


    standardized_ct_inputs = np.empty(ct_inputs.shape)
    for c in range(ct_inputs.shape[-1]):
        if c in channels_to_leave_out:
            standardized_ct_inputs[..., c] = ct_inputs[..., c]
            continue
        if min_max_scaling:
            standardized_ct_inputs[..., c] = scale(standardize(ct_inputs[..., c]))
        else:
            standardized_ct_inputs[..., c] = standardize(ct_inputs[..., c])
        print('CT channel', c, np.mean(standardized_ct_inputs[..., c]), np.std(standardized_ct_inputs[..., c]))

    if len(mri_inputs.shape) != 0:
        standardized_mri_inputs = np.empty(mri_inputs.shape)
        for c in range(mri_inputs.shape[-1]):
            standardized_mri_inputs[..., c] = standardize(mri_inputs[..., c])
            print('MRI channel', c, np.mean(standardized_mri_inputs[..., c]), np.std(standardized_mri_inputs[..., c]))
    else:
        standardized_mri_inputs = mri_inputs

    np.savez_compressed(os.path.join(data_dir, 'standardized_' + filename),
                        params=params,
                        ids=ids,
                        clinical_inputs=clinical_inputs, ct_inputs=standardized_ct_inputs, ct_lesion_GT=ct_lesion_GT,
                        mri_inputs=standardized_mri_inputs, mri_lesion_GT=mri_lesion_GT, brain_masks=brain_masks)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standardize dataset')
    parser.add_argument('data_path')
    parser.add_argument('-c', '--channels_to_leave_out', nargs='+',  help='Indexes of channels to leave out', required=False, type=int, default=[])
    parser.add_argument('-o', '--outlier_scaling', action='store_true', default=False, required=False)

    args = parser.parse_args()
    data_dir = os.path.dirname(args.data_path)
    file_name = os.path.basename(args.data_path)

    standardize_data(data_dir, filename=file_name, channels_to_leave_out=args.channels_to_leave_out, outlier_scaling=args.outlier_scaling)
