import argparse
import numpy as np
import pandas as pd
import math, os
from gsd_pipeline import data_loader as dl


def filter_for_clinical_param(dataset_path, clinical_path, clinical_parameter, id_parameter='anonymised_id'):
    data_dir = os.path.dirname(dataset_path)
    file_name = os.path.basename(dataset_path)
    data = dl.load_saved_data(data_dir, file_name)
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params = data

    clinical_df = pd.read_excel(clinical_path)

    indices_to_remove = []
    for idx, subj_id in enumerate(ids):
        if not subj_id in set(clinical_df[id_parameter]):
            print(f'{subj_id} not found in clinical database. Will be removed.')
            indices_to_remove.append(idx)
            continue
        if math.isnan(clinical_df.loc[clinical_df[id_parameter] == subj_id, clinical_parameter].iloc[0]):
            print(f'{subj_id} has no parameter "{clinical_parameter}" in clinical database. Will be removed.')
            indices_to_remove.append(idx)

    ct_inputs = np.delete(ct_inputs, indices_to_remove, axis=0)
    masks = np.delete(masks, indices_to_remove, axis=0)
    ids = np.delete(ids, indices_to_remove, axis=0)

    assert ct_inputs.shape[0] == masks.shape[0]
    assert ct_inputs.shape[0] == ids.shape[0]

    if not ct_lesion_GT.size <= 1:
        ct_lesion_GT = np.delete(ct_lesion_GT, indices_to_remove, axis=0)
        assert ct_inputs.shape[0] == ct_lesion_GT.shape[0]

    if not len(mri_inputs) == 0:
        mri_inputs = np.delete(mri_inputs, indices_to_remove, axis=0)
        mri_lesion_GT = np.delete(mri_lesion_GT, indices_to_remove, axis=0)

    outfile = os.path.basename(dataset_path).split('.')[0] + f'_with_{clinical_parameter}.npz'

    dataset = (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params)
    dl.save_dataset(dataset, os.path.dirname(dataset_path), out_file_name=outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add penumbra segmentation based on Tmax > 6s to the perfusion map dataset')
    parser.add_argument('dataset_path')
    parser.add_argument('clinical_path')
    parser.add_argument('-p', '--parameter',  help='Clinical parameter to be filtered for', required=True, default=None)
    parser.add_argument('-i', '--id',  help='Name of id field in clinical df', required=False, default='anonymised_id')

    args = parser.parse_args()
    filter_for_clinical_param(args.dataset_path, args.clinical_path, args.parameter, id_parameter=args.id)

