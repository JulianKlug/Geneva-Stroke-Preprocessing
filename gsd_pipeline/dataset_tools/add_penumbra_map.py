import os
import numpy as np
import argparse
from gsd_pipeline import data_loader as dl


def add_penumbra_map(ct_dataset:[str, np.ndarray], masks = None, tmax_channel = 0, outfile = None, one_hot_encode=False):

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

    penumbra_mask = np.zeros((n_subj, n_x, n_y, n_z))
    penumbra_mask[ct_inputs[..., tmax_channel] > 6] = 1

    if masks is not None:
        # Restrict to defined prior mask
        restr_penumbra_mask = penumbra_mask * masks
    else:
        restr_penumbra_mask = penumbra_mask

    restr_penumbra_mask = np.expand_dims(restr_penumbra_mask, axis=-1)

    if one_hot_encode:
        class_0_core = 1 - restr_penumbra_mask
        class_1_core = restr_penumbra_mask
        restr_penumbra_mask = np.concatenate((class_0_core, class_1_core), axis=-1)

    ct_inputs = np.concatenate((ct_inputs, restr_penumbra_mask), axis=-1)

    if isinstance(ct_dataset, str):
        if outfile is None:
            if one_hot_encode:
                outfile = os.path.basename(ct_dataset).split('.')[0] + '_with_one_hot_encoded_core_penumbra.npz'
            else:
                outfile = os.path.basename(ct_dataset).split('.')[0] + '_with_penumbra.npz'
        params = params.item()
        if one_hot_encode:
            params['ct_sequences'].append('penumbra_Tmax_6_class0')
            params['ct_sequences'].append('penumbra_Tmax_6_class1')
        else:
            params['ct_sequences'].append('penumbra_Tmax_6')
        dataset = (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params)
        dl.save_dataset(dataset, os.path.dirname(ct_dataset), out_file_name=outfile)

    return ct_inputs, restr_penumbra_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add penumbra segmentation based on Tmax > 6s to the perfusion map dataset')
    parser.add_argument('data_path')
    parser.add_argument('-o', '--outfile',  help='Name of output file', required=False, default=None)
    parser.add_argument('-t', '--tmax_channel',  help='Tmax Channel index in dataset', required=False, default=1)
    parser.add_argument('-oh', '--one_hot_encode', action='store_true', default=False, required=False)

    args = parser.parse_args()
    add_penumbra_map(args.data_path, outfile=args.outfile, tmax_channel=args.tmax_channel, one_hot_encode=args.one_hot_encode)

