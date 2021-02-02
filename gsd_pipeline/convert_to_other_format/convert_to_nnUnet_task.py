import argparse
import os, json
import nibabel as nib
import numpy as np
from tqdm import tqdm
from gsd_pipeline.utils.file_operations import subfiles, save_json, mmkdir


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def convert_to_nnUnet_task(path_to_gsd_dataset: str, path_to_new_dataset_main_dir: str, channel_names: list = []):
    path_to_new_dataset_dir = os.path.join(path_to_new_dataset_main_dir, 'nnUNet_raw_data')
    path_to_new_dataset_task_dir = os.path.join(path_to_new_dataset_dir, 'Task101_GSD')
    new_dataset_prepro_dir = os.path.join(path_to_new_dataset_main_dir, 'nnUNet_prepro_data')
    new_dataset_prepro_dir = os.path.join(path_to_new_dataset_main_dir, 'results')
    mmkdir(path_to_new_dataset_main_dir)
    mmkdir(path_to_new_dataset_dir)
    mmkdir(path_to_new_dataset_task_dir)
    mmkdir(new_dataset_prepro_dir)


    params = np.load(path_to_gsd_dataset, allow_pickle=True)['params'].item()
    ids = np.load(path_to_gsd_dataset, allow_pickle=True)['ids']
    ct_inputs = np.load(path_to_gsd_dataset, allow_pickle=True)['ct_inputs']
    try:
        ct_lesion_GT = np.load(path_to_gsd_dataset, allow_pickle=True)['ct_lesion_GT']
    except:
        ct_lesion_GT = np.load(path_to_gsd_dataset, allow_pickle=True)['lesion_GT']

    # brain_masks = np.load(path_to_gsd_dataset, allow_pickle=True)['brain_masks']

    print('Loading a total of', ct_inputs.shape[0], 'subjects.')
    print('Sequences used:', params)

    target_imagesTrain = os.path.join(path_to_new_dataset_task_dir, "imagesTr")
    # target_imagesTs = join(target_base, "imagesTs")
    # target_labelsTs = join(target_base, "labelsTs")
    target_labelsTrain = os.path.join(path_to_new_dataset_task_dir, "labelsTr")

    mmkdir(path_to_new_dataset_task_dir)
    mmkdir(target_imagesTrain)
    mmkdir(target_labelsTrain)

    with open(os.path.join(path_to_new_dataset_task_dir, 'params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    n_channels = ct_inputs.shape[-1]

    if not channel_names:
        if len(params['ct_sequences']) == n_channels:
            channel_names = params['ct_sequences']
        else:
            raise Exception('Please provide channel names, as those provided in GSD parameters are insufficient')

    for subj_idx, id in enumerate(tqdm(ids)):
        for channel in range(n_channels):
            subj_pct_channel_img = nib.Nifti1Image(ct_inputs[subj_idx, ..., channel], np.eye(4))
            nib.save(subj_pct_channel_img, os.path.join(target_imagesTrain, f'{id}_{channel:04d}.nii.gz'))

        subj_lesion_img = nib.Nifti1Image(ct_lesion_GT[subj_idx], np.eye(4))
        nib.save(subj_lesion_img, os.path.join(target_labelsTrain, f'{id}.nii.gz'))

    train_identifiers = get_identifiers_from_splitted_files(target_imagesTrain)

    # if imagesTs_dir is not None:
    #     test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    # else:
    #     test_identifiers = []

    test_identifiers = []

    # generate dataset json
    json_dict = {}
    json_dict['name'] = 'Geneva Stroke Dataset'
    json_dict['description'] = 'Acute Perfusion CT of stroke patients with final lesion labelling'
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = 'https://doi.org/10.1177%2F0271678X20924549'
    json_dict['licence'] = 'Confidential'
    json_dict['release'] = '1.0 05/06/2020'
    json_dict['modality'] = {str(i): channel_names[i] for i in range(len(channel_names))}
    json_dict['labels'] = {
       "0": "background",
       "1": "lesion"
     }

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = []
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    save_json(json_dict, os.path.join(path_to_new_dataset_task_dir, 'dataset.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert GSD dataset to nnUnet dataset')
    parser.add_argument('gsd_data_path')
    parser.add_argument('new_data_path')

    args = parser.parse_args()
    convert_to_nnUnet_task(args.gsd_data_path, args.new_data_path)
