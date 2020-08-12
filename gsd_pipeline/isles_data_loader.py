import os, math
import numpy as np
import nibabel as nib

# Save data as compressed numpy array
def create_dataset(save_dir, data_dir, filename='isles_data_set', ct_input_sequences = [], use_perfusion_maps=True, use_nc_ct=True, use_4d_pct=False):
    """
    Load data from ISLES dataset
        - Image data (from preprocessed Nifti)

    Args:
    :param     save_dir : directory to save data_to
    :param     data_dir : directory containing images
    :param     filename (optional) : output filename
    :param     ct_input_sequences (optional, array) : array with names of ct sequences
    :param     use_nc_ct (optional, default True): use non contrast CT as additional input
    :param     use_perfusion_maps (optional, default True): use perfusion maps as input
    :param     use_4d_pct (optional, default False): use 4D perfusion CT as input


    Returns:
    :return    save data as a date_set file storing
                (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids, params)
    """

    input_sequences = {
        'tmax': False,
        'cbf': False,
        'mtt': False,
        'cbv': False,
        'ncct': False,
        'pct': False
    }

    sequence_names = {
        'tmax': 'SMIR.Brain.XX.O.CT_Tmax',
        'cbf': 'SMIR.Brain.XX.O.CT_CBF',
        'mtt': 'SMIR.Brain.XX.O.CT_MTT',
        'cbv': 'SMIR.Brain.XX.O.CT_CBV',
        'ncct': 'SMIR.Brain.XX.O.CT',
        'pct': 'SMIR.Brain.XX.O.CT_4DPWI',
        'label': 'SMIR.Brain.XX.O.OT'
    }

    for sequence in ct_input_sequences:
        sequence = sequence.lower()
        if sequence in input_sequences:
            input_sequences[sequence] = True
        else:
            print(f'{sequence} not defined for ISLES dataset')

    if use_perfusion_maps:
        input_sequences['tmax'] = True
        input_sequences['cbf'] = True
        input_sequences['mtt'] = True
        input_sequences['cbv'] = True

    if use_nc_ct:
        input_sequences['ncct'] = True

    if use_4d_pct:
        input_sequences['pct'] = True

    if all(value is False for value in input_sequences.values()):
        raise IOError('No input sequence defined.')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ids, ct_inputs, ct_lesion_GT = load_data(data_dir, input_sequences, sequence_names)

    print('Saving a total of', ct_inputs.shape[0], 'subjects.')
    np.savez_compressed(os.path.join(save_dir, filename),
                        params = {'ct_sequences': [sequence for sequence, used in input_sequences.items() if used], 'ct_label_sequences': sequence_names['label'],
                                  'mri_sequences': None, 'mri_label_sequences': None},
                        ids = ids, included_subjects = ids,
                        clinical_inputs = None, ct_inputs = ct_inputs, ct_lesion_GT = ct_lesion_GT,
                        mri_inputs = None, mri_lesion_GT = None, brain_masks = None)

def load_data(data_dir, ct_sequences: dict, sequence_names: dict):
    """
    Given a data dir, a directory of ct_sequences to use and their respective file names return all input and label as np.arrays
    :param data_dir: main dir containing all subject directories
    :param ct_sequences: dictionary of sequence_names and a boolean defining if this sequence should be part of the final dataset
    :param sequence_names: dictionary of sequence_names and their corresponding filenames
    :return: ids, input_data, labels
    """

    subjects = [o for o in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, o))]

    ids = []
    input_data = []
    labels = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        subj_input_sequences = []

        for sequence, add_sequence_to_input in ct_sequences.items():
            if add_sequence_to_input:
                sequence_data = load_subj_sequence(subj_dir, sequence_names[sequence])
                subj_input_sequences.append(sequence_data)

        # if all requested sequences were found
        if len(subj_input_sequences) > 1:
            ids.append(subj)
            input_data.append(np.array(subj_input_sequences))
            subj_label = load_subj_sequence(subj_dir, sequence_names['label'])
            labels.append(np.array(subj_label))

    # harmonise shapes by padding to max shape
    max_z_slices = np.max([subj_data.shape[-1] for subj_data in input_data])
    for subj_idx in range(len(input_data)):
        delta_z = max_z_slices - input_data[subj_idx].shape[-1]
        # pad half on bottom and half on top (when delta not even, 1 more slice on top)
        z_padding = (int(delta_z/2), math.ceil(delta_z/2))
        input_data[subj_idx] = np.pad(input_data[subj_idx], ((0, 0), (0, 0), (0, 0), z_padding), constant_values=0)
        labels[subj_idx] = np.pad(labels[subj_idx], ((0, 0), (0, 0), z_padding), constant_values=0)

    # set channels last
    input_data = np.moveaxis(np.array(input_data), 1, -1)
    labels = np.expand_dims(np.array(labels), axis=-1)
    ids = np.array(ids)

    return ids, input_data, labels

def load_subj_sequence(subj_dir, sequence_name):
    sequence_directories = [o for o in os.listdir(subj_dir)
                            if os.path.isdir(os.path.join(subj_dir, o))]
    sequence_dirname = [o for o in sequence_directories
                        if '.'.join(o.split('.')[:-1]) == sequence_name][0]
    sequence_dir = os.path.join(subj_dir, sequence_dirname)
    sequence_file = [o for o in os.listdir(sequence_dir)
                     if os.path.splitext(o)[1] == '.nii' and '.'.join(os.path.splitext(o)[0].split('.')[:-1]) == sequence_name][0]
    sequence_data = nib.load(os.path.join(sequence_dir, sequence_file)).get_fdata()
    return sequence_data