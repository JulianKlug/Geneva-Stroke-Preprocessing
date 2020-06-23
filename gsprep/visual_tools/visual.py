from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from ipywidgets import fixed
import ipywidgets as widgets
import IPython


def display(
        img_data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        block=False,
        title=None,
        idx_x: Optional[int] = None,
        idx_y: Optional[int] = None,
        idx_z: Optional[int] = None,
        verbose: Optional[bool] = True,
        return_figure: Optional[bool] = False,
        cmap='gray',
        mask_cmap='Reds',
        mask_alpha=0.2
        ):
    fig = plt.figure(figsize=(20, 5))

    def show_slices(slices, masks=None):
        """ Function to display row of image slices """
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        gs.tight_layout(fig)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        axes = ax1, ax2, ax3
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap=cmap, origin="lower")
            if masks is not None:
                axes[i].imshow(masks[i].T, cmap=mask_cmap, origin="lower", alpha=mask_alpha)

    if len(img_data.shape) > 3 : img_data = np.squeeze(img_data)
    if mask is not None:
        if len(mask.shape) > 3: mask = np.squeeze(mask)
        assert mask.shape == img_data.shape, 'Input and mask shapes do not match'

    n_x, n_y, n_z = img_data.shape
    idx_x = (n_x - 1) // 2 if idx_x is None else idx_x
    idx_y = (n_y - 1) // 2 if idx_y is None else idx_y
    idx_z = (n_z - 1) // 2 if idx_z is None else idx_z

    if verbose:
        print('Image center: ', idx_x, idx_y, idx_z)
        center_vox_value = img_data[idx_x, idx_y, idx_z]
        print('Image center value: ', center_vox_value)

    slice_0 = img_data[idx_x, :, :]
    slice_1 = img_data[:, idx_y, :]
    slice_2 = img_data[:, :, idx_z]

    if mask is not None:
        mask_0 = mask[idx_x, :, :]
        mask_1 = mask[:, idx_y, :]
        mask_2 = mask[:, :, idx_z]
        show_slices([slice_0, slice_1, slice_2], masks=[mask_0, mask_1, mask_2])
    else:
        show_slices([slice_0, slice_1, slice_2])

    plt.suptitle("Center slices for image")
    if title:
        plt.suptitle(title)
    plt.show(block = block)
    if return_figure:
        return fig


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

def idisplay(array: np.ndarray, **kwargs) -> None:
    # inspired by https://github.com/fepegar/miccai-educational-challenge-2019/blob/master/visualization.py
    def get_widget(size, description):
        widget = widgets.IntSlider(
            min=0,
            max=size-1,
            step=1,
            value=size//2,
            continuous_update=False,
            description=description,
        )
        return widget
    shape = array.shape[:3]
    names = 'Sagittal', 'Coronal', 'Axial'
    widget_sag, widget_cor, widget_axi = [
        get_widget(s, n) for (s, n) in zip(shape, names)]
    ui = widgets.HBox([widget_sag, widget_cor, widget_axi])
    args_dict = {
        'img_data': fixed(array),
        'idx_x': widget_sag,
        'idx_y': widget_cor,
        'idx_z': widget_axi,
        'verbose': fixed(False),
        'return_figure': fixed(True)
    }
    kwargs = {key: fixed(value) for (key, value) in kwargs.items()}
    args_dict.update(kwargs)
    out = widgets.interactive_output(display, args_dict)
    IPython.display.display(ui, out)