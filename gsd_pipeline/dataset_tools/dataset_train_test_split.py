import argparse
import os
from sklearn.model_selection import train_test_split
from gsd_pipeline import data_loader as dl
import pandas as pd
import numpy as np


def dataset_train_test_split(dataset_path, test_size=0.33, random_state=42, stratification_data=None, stratify_var=None,
                             shuffle=True):
    data_dir = os.path.dirname(dataset_path)
    file_name = os.path.basename(dataset_path)
    data = dl.load_saved_data(data_dir, file_name)
    clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, masks, ids, params = data
    params = params.item()
    all_indices = list(range(len(ids)))

    if stratification_data is not None:
        stratification_df = pd.read_excel(stratification_data)
        stratification_labels = [
            stratification_df.loc[stratification_df['anonymised_id'] == subj_id, stratify_var].iloc[0]
            for subj_id in ids]
        print(f'Initial positive label distribution: {sum(stratification_labels)/len(stratification_labels)}')
        train_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=random_state,
                                                       shuffle=shuffle, stratify=stratification_labels)
    else:
        train_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=random_state,
                                                       shuffle=shuffle)

    def split_data(data, train_indices, test_indices):
        if len(data.shape) > 0 and data.size > 0:
            train_data = data[train_indices]
            test_data = data[test_indices]
        else:
            train_data = data
            test_data = data
        return train_data, test_data

    train_clinical, test_clinical = split_data(clinical_inputs, train_indices, test_indices)
    train_ct_inputs, test_ct_inputs = split_data(ct_inputs, train_indices, test_indices)
    train_ct_lesion_GT, test_ct_lesion_GT = split_data(ct_lesion_GT, train_indices, test_indices)
    train_mri_inputs, test_mri_inputs = split_data(mri_inputs, train_indices, test_indices)
    train_mri_lesion_GT, test_mri_lesion_GT = split_data(mri_lesion_GT, train_indices, test_indices)
    train_brain_masks, test_brain_masks = split_data(masks, train_indices, test_indices)
    train_ids, test_ids = split_data(ids, train_indices, test_indices)

    if stratification_data is not None:
        print(f'Train positive label distribution: '
              f'{sum(np.array(stratification_labels)[train_indices])/len(np.array(stratification_labels)[train_indices])}')
    train_dataset = (
        train_clinical, train_ct_inputs, train_ct_lesion_GT, train_mri_inputs, train_mri_lesion_GT, train_brain_masks,
        train_ids, params)

    if stratification_data is not None:
        print(f'Test positive label distribution: '
              f'{sum(np.array(stratification_labels)[test_indices])/len(np.array(stratification_labels)[test_indices])}')
    test_dataset = (
        test_clinical, test_ct_inputs, test_ct_lesion_GT, test_mri_inputs, test_mri_lesion_GT, test_brain_masks,
        test_ids,
        params)

    train_outfile = 'train_' + file_name
    test_outfile = 'test_' + file_name

    dl.save_dataset(train_dataset, data_dir, out_file_name=train_outfile)
    dl.save_dataset(test_dataset, data_dir, out_file_name=test_outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into a train and test subset')
    parser.add_argument('dataset_path')
    parser.add_argument('-t', '--test_size', help='Relative size of test set (0-1)', required=False, default=0.33)
    parser.add_argument('-r', '--random_state', help='Random state for splitting', required=False, default=42)
    parser.add_argument('-d', '--stratification_data_path', help='Path to excel file for stratification data',
                        required=False, default=None)
    parser.add_argument('-v', '--stratification_var', help='Variable of stratification data to use for stratification',
                        required=False, default=None)
    parser.add_argument('-ns', '--no_shuffle', action='store_false', default=True, required=False)

    args = parser.parse_args()
    dataset_train_test_split(args.dataset_path, test_size=float(args.test_size), random_state=args.random_state,
                             stratification_data=args.stratification_data_path, stratify_var=args.stratification_var,
                             shuffle=args.no_shuffle)
