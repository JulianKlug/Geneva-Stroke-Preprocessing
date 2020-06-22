import os
import matplotlib.pyplot as plt
import numpy as np


def display(img_data, mask=None, block=True, title=None, cmap='gray', mask_cmap='Reds', mask_alpha=0.2):
    def show_slices(slices, masks=None):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap=cmap, origin="lower")
            if masks is not None:
                axes[i].imshow(masks[i].T, cmap=mask_cmap, origin="lower", alpha=mask_alpha)

    if len(img_data.shape) > 3 : img_data = np.squeeze(img_data)
    if mask is not None:
        if len(mask.shape) > 3: mask = np.squeeze(mask)
        assert mask.shape == img_data.shape, 'Input and mask shapes do not match'
    n_i, n_j, n_k = img_data.shape
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2
    print('Image center: ', center_i, center_j, center_k)
    center_vox_value = img_data[center_i, center_j, center_k]
    print('Image center value: ', center_vox_value)

    slice_0 = img_data[center_i, :, :]
    slice_1 = img_data[:, center_j, :]
    slice_2 = img_data[:, :, center_k]

    if mask is not None:
        mask_0 = mask[center_i, :, :]
        mask_1 = mask[:, center_j, :]
        mask_2 = mask[:, :, center_k]
        show_slices([slice_0, slice_1, slice_2], masks=[mask_0, mask_1, mask_2])
    else:
        show_slices([slice_0, slice_1, slice_2])

    plt.suptitle("Center slices for image")
    if title:
        plt.suptitle(title)
    plt.show(block = block)
    return plt
