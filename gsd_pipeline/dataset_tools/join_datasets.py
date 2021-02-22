import os
import numpy as np
import argparse
from gsd_pipeline import data_loader as dl


def contains_duplicates(X):
    return len(np.unique(X)) != len(X)


def join_datasets(dataset1_path, dataset2_path, outfile = None):
    data_dir1 = os.path.dirname(dataset1_path)
    file_name1 = os.path.basename(dataset1_path)
    data_dir2 = os.path.dirname(dataset2_path)
    file_name2 = os.path.basename(dataset2_path)
    data1 = dl.load_saved_data(data_dir1, file_name1)
    data2 = dl.load_saved_data(data_dir2, file_name2)

    clinical_inputs1, ct_inputs1, ct_lesion_GT1, mri_inputs1, mri_lesion_GT1, masks1, ids1, params1 = data1
    clinical_inputs2, ct_inputs2, ct_lesion_GT2, mri_inputs2, mri_lesion_GT2, masks2, ids2, params2 = data2

    out_ids = np.append(ids1, ids2, axis=0)

    assert not contains_duplicates(out_ids), 'Joined dataset would contain duplicated ids. Aborting.'

    assert ct_inputs2.shape[-1] == ct_inputs1.shape[-1], 'Datasets do not have the same number of channels. Aborting.'
    out_ct_inputs = np.append(ct_inputs1, ct_inputs2, axis=0)

    if ct_lesion_GT1.size <= 1 or ct_lesion_GT2.size <= 1:
        print('Ignoring ct ground truth, as at least one dataset has no entries.')
        out_ct_lesion_GT = np.array([])
    else:
        out_ct_lesion_GT = np.append(ct_lesion_GT1, ct_lesion_GT2, axis=0)

    if clinical_inputs1.size <= 1 or clinical_inputs2.size <= 1:
        print('Ignoring clinical input, as at least one dataset has no entries.')
        out_clinical_inputs = np.array([])
    else:
        out_clinical_inputs = np.append(clinical_inputs1, clinical_inputs2, axis=0)

    if len(mri_inputs1) == 0 or len(mri_inputs2) == 0:
        print('Ignoring mri input, as at least one dataset has no entries.')
        out_mri_inputs = np.array([])
        out_mri_lesion_GT = np.array([])
    else:
        assert mri_inputs1.shape[-1] == mri_inputs2.shape[-1], 'Datasets do not have the same number of channels. Aborting.'

        out_mri_inputs = np.append(mri_inputs1, mri_inputs2, axis=0)
        out_mri_lesion_GT = np.append(mri_lesion_GT1, mri_lesion_GT2, axis=0)

    out_masks = np.append(masks1, masks2, axis=0)

    # params should stay the same
    out_params = params1

    print('Saving new dataset with: ', out_params)
    print('Ids:', out_ids.shape)
    print('Clinical:', out_clinical_inputs.shape)
    print('CT in:', out_ct_inputs.shape)
    print('CT gt:', out_ct_lesion_GT.shape)
    print('MRI in:', out_mri_inputs.shape)
    print('MRI gt:', out_mri_lesion_GT.shape)
    print('masks:', out_masks.shape)

    if outfile is None:
        outfile = 'joined_dataset.npz'

    dataset = (out_clinical_inputs, out_ct_inputs, out_ct_lesion_GT, out_mri_inputs, out_mri_lesion_GT, out_masks, out_ids, out_params)
    dl.save_dataset(dataset, os.path.dirname(dataset1_path), out_file_name=outfile)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Join two datasets')
    parser.add_argument('data1_path')
    parser.add_argument('data2_path')
    parser.add_argument('-o', '--outfile',  help='Name of output file', required=False, default=None)

    args = parser.parse_args()
    join_datasets(args.data1_path, args.data2_path, outfile=args.outfile)

