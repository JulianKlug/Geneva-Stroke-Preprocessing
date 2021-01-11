from gsd_pipeline.data_loader import load_saved_data
import numpy as np
import os

def binarize_lesions(data_dir, filename='data_set.npz', threshold=0.1):
        (clinical_inputs, ct_inputs, ct_lesion_GT, mri_inputs, mri_lesion_GT, brain_masks, ids,
         params) = load_saved_data(data_dir, filename)

        brain_masks[brain_masks < threshold] = 0
        brain_masks[brain_masks >= threshold] = 1
        brain_masks = brain_masks.astype(int)

        np.savez_compressed(os.path.join(data_dir, 'bin_' + filename),
                            params=params,
                            ids=ids,
                            clinical_inputs=clinical_inputs, ct_inputs=ct_inputs, ct_lesion_GT=ct_lesion_GT,
                            mri_inputs=mri_inputs, mri_lesion_GT=mri_lesion_GT, brain_masks=brain_masks)

