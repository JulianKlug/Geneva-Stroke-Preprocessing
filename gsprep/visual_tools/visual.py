import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


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


def display_4D(img_data, block=False, title=None, cmap='gray', fps=30):
    if len(img_data.shape) > 4: img_data = np.squeeze(img_data)
    if len(img_data.shape) != 4: raise Exception('Image data must be of shape (x,y,z,t).')
    n_x, n_y, n_z, n_t = img_data.shape
    center_x = (n_x - 1) // 2
    center_y = (n_y - 1) // 2
    center_z = (n_z - 1) // 2
    x_slices = img_data[center_x, :, :, :]
    y_slices = img_data[:, center_y, :, :]
    z_slices = img_data[:, :, center_z, :]
    first_time_points = [x_slices[..., 0], y_slices[..., 0], z_slices[..., 0]]

    fig, axes = plt.subplots(1, 3)
    for i, axe in enumerate(axes):
        axe.imshow(first_time_points[i].T, cmap=cmap, origin="lower")


    def animate(i):
        if i % fps == 0:
            print('.', end='')

        axes[0].imshow(x_slices[..., i].T, cmap=cmap, origin="lower")
        axes[1].imshow(y_slices[..., i].T, cmap=cmap, origin="lower")
        axes[2].imshow(z_slices[..., i].T, cmap=cmap, origin="lower")

        return [axes]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=n_t,
        interval=1000 / fps,  # in ms
    )
    plt.show(block=block)

    return plt