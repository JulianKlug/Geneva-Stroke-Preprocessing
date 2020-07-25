import os
import argparse
import numpy as np
import nibabel as nib


def split_unique_values(data_path, save_path=None, background_value=0):
    """
    Split a nifti image along all unique values it contains. Useful for decomposing mask containing multiple labels.

    :param data_path: path to nifti image
    :param save_path: Optional, path to directory where output should be saved. Defaults to name of input file.
    :param background_value: Optional, value given to background. Defaults to 0.
    :return: masks: np.array containing all final masks
    """
    img = nib.load(data_path)
    data = img.get_fdata()
    coordinate_space = img.affine
    image_extension = '.nii'

    if save_path is None:
        save_path = data_path.split('.')[0]
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    masks = []
    for value in np.unique(data):
        # fill background with background value
        unique_value_mask = np.full(data.shape, background_value)
        unique_value_mask[data == value] = 1

        # MATLAB can not open NIFTI saved as int, thus float is necessary
        unique_value_mask_img = nib.Nifti1Image(unique_value_mask.astype('float64'), affine=coordinate_space)
        output_img_name = os.path.basename(data_path).split('.')[0] + '_' + str(value) + image_extension
        nib.save(unique_value_mask_img, os.path.join(save_path, output_img_name))
        masks.append(unique_value_mask)

    return np.array(masks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a nifti image along all unique values it contains.')
    parser.add_argument('data_path')
    parser.add_argument('-s', '--save_path',  help='Save to a specific directory', required=False, default=None)
    parser.add_argument('-b', '--background_value',  help='Value given to background', required=False, default=0)

    args = parser.parse_args()
    split_unique_values(args.data_path, args.save_path, args.background_value)

