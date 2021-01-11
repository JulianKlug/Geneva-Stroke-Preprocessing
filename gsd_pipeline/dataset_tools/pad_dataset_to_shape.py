import os, argparse
from gsd_pipeline.data_loader import save_dataset, load_saved_data
from gsd_pipeline.utils.utils import pad_to_shape
import numpy as np

def pad_dataset_to_shape(dataset_path: str, shape: tuple):
    """
    Pad a dataset to a desired shape (equal padding on both sides)
    :param dataset_path: path do dataset
    :param shape: tuple, desired shape
    :return: saves new padded dataset to same location with "padded_" prefix
    """
    data_dir = os.path.dirname(dataset_path)
    filename = os.path.basename(dataset_path)
    (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params) = \
        load_saved_data(data_dir, filename)

    padded_ct_inputs = np.array([pad_to_shape(ct_input, shape) for ct_input in ct_inputs])
    padded_ct_lesion_GT = np.array([pad_to_shape(lesion_GT, shape) for lesion_GT in ct_lesion_GT])
    padded_brain_masks = np.array([pad_to_shape(brain_mask, shape) for brain_mask in brain_masks])

    if len(mri_inputs) == 0:
        padded_mri_inputs = []
    else:
        padded_mri_inputs = np.array([pad_to_shape(mri_input, shape) for mri_input in mri_inputs])
    if len(mri_lesion_GT) == 0:
        padded_mri_lesion_GT = []
    else:
        padded_mri_lesion_GT = np.array([pad_to_shape(lesion_GT, shape) for lesion_GT in mri_lesion_GT])

    dataset = (clinical_inputs, padded_ct_inputs, padded_ct_lesion_GT,
               padded_mri_inputs, padded_mri_lesion_GT,
               padded_brain_masks, ids, params)

    save_dataset(dataset, data_dir, 'padded_' + filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pad dataset to given shape')
    parser.add_argument('data_path')
    parser.add_argument('-s', '--shape', help='Target shape (eg: 256 256 100)', required=True, nargs="+", type=int, default=None)

    args = parser.parse_args()
    shape = tuple(args.shape)
    pad_dataset_to_shape(args.data_path, shape)
