"""
    Sample implementations of visualization functions. 
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm


def draw_cell_area(
    imgs=None, channel=None, seg_masks=None, savedir=None, configs=None, display=False
):
    """
    Draw the cell areas on top of the original images.
    """

    assert len(imgs.shape) == 4
    assert len(seg_masks.shape) == 4

    # pick the channel to visualize
    _imgs = imgs[channel]
    _seg_masks = seg_masks[channel]

    field_id = "area"
    field_savedir = os.path.join(savedir, field_id)
    if not os.path.exists(field_savedir):
        os.makedirs(field_savedir)

    frame_ids = np.linspace(0, len(_seg_masks) - 1, len(_seg_masks)).astype(np.uint8)

    # find the min and max areas for the colormap
    # by binning the areas of all cells in all frames
    all_areas = []
    for frame_idx in frame_ids:
        _mask = _seg_masks[frame_idx]
        for _region in regionprops(_mask):
            if _region.label == 0:  # background
                continue
            all_areas.append(_region.area)

    _, area_bin_edges = np.histogram(all_areas, bins=10)
    min_area = area_bin_edges[1]
    max_area = area_bin_edges[-2]

    for frame_idx in tqdm(frame_ids, ascii=True, desc="Drawing cell areas..."):
        _cell_masks = _seg_masks[frame_idx]

        _areas_mask = np.zeros_like(_cell_masks)

        fig, ax = plt.subplots(figsize=configs["figsize"])

        ax.imshow(_imgs[frame_idx], cmap=configs["img_cmap"])

        for cell_region in tqdm(regionprops(_cell_masks)):
            if cell_region.label == 0:
                continue

            _area = cell_region.area

            _areas_mask[_cell_masks == cell_region.label] = _area

        # ax.imshow(_areas_mask, alpha=0.5, cmap=colormap, vmin=min_area, vmax=max_area)
        ax.imshow(
            _areas_mask,
            alpha=configs["field_alpha"],
            cmap=configs["field_cmap"],
            vmin=min_area,
            vmax=max_area,
        )
        ax.axis("off")
        fig.tight_layout()

        filepath = os.path.join(
            field_savedir, field_id + "_" + str(frame_idx).zfill(4) + ".png"
        )
        fig.savefig(
            filepath,
            dpi=configs["dpi"],
            bbox_inches=configs["bbox_inches"],
            pad_inches=configs["pad_inches"],
            transparent=configs["transparent"],
        )
        if not display:
            plt.close()
            del fig


def draw_cell_orientation(
    imgs=None, channel=None, seg_masks=None, savedir=None, configs=None, display=False
):
    """
    Draw the cell orientations on top of the original images.
    """

    assert len(imgs.shape) == 4
    assert len(seg_masks.shape) == 4

    # pick the channel to visualize
    _imgs = imgs[channel]
    _seg_masks = seg_masks[channel]

    field_id = "orientation"
    field_savedir = os.path.join(savedir, field_id)
    if not os.path.exists(field_savedir):
        os.makedirs(field_savedir)

    frame_ids = np.linspace(0, len(_seg_masks) - 1, len(_seg_masks)).astype(np.uint8)

    for frame_idx in tqdm(
        frame_ids,
        ascii=True,
        desc="Drawing cell orientations...",
    ):
        _cell_masks = _seg_masks[frame_idx]

        fig, ax = plt.subplots(figsize=configs["figsize"])

        ax.imshow(_imgs[frame_idx], cmap=configs["img_cmap"])

        for cell_region in tqdm(regionprops(_cell_masks)):
            if cell_region.label == 0:  # background
                continue

            orientation = cell_region.orientation
            centroid = cell_region.centroid
            major_axis_length = cell_region.major_axis_length

            line_length = major_axis_length / 4.0

            x0 = centroid[1] - (line_length) * np.sin(orientation)
            y0 = centroid[0] - (line_length) * np.cos(orientation)

            x1 = centroid[1] + (line_length) * np.sin(orientation)
            y1 = centroid[0] + (line_length) * np.cos(orientation)

            ax.plot(
                [x0, x1],
                [y0, y1],
                linewidth=configs["line_width"],
                color=configs["orientation_color"],
            )

        ax.axis("off")
        fig.tight_layout()

        filepath = os.path.join(
            field_savedir, field_id + "_" + str(frame_idx).zfill(4) + ".png"
        )
        fig.savefig(
            filepath,
            dpi=configs["dpi"],
            bbox_inches=configs["bbox_inches"],
            pad_inches=configs["pad_inches"],
            transparent=configs["transparent"],
        )
        if not display:
            plt.close()
            del fig


class TissuePlotter:
    fields = ["area", "orientation"]
    funcs = {"area": draw_cell_area, "orientation": draw_cell_orientation}

    plotter_configs = {
        "img_cmap": "gray",
        "field_cmap": "magma",
        "field_alpha": 0.5,
        "orientation_color": "g",
        "field_vmin": 0,
        "field_vmax": 1000,
        "line_width": 2.0,
        "figsize": (5, 5),
        "dpi": 300,
        "bbox_inches": "tight",
        "pad_inches": 0,
        "transparent": True,
    }

    def __init__(self, imgs, seg_masks):
        """
        imgs: [C, T, H, W]
        seg_masks: [1, T, H, W]
        """

        assert len(imgs.shape) == 4
        assert len(seg_masks.shape) == 4

        self.imgs = imgs
        self.seg_masks = seg_masks

    def plot(self, field=None, channel=None, savedir=None, display=False):
        if field in self.fields:
            assert channel in [0, 1], "Channel must be 0 or 1"  # TODO: generalize

            assert savedir is not None, "Must provide a savedir"
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            self.funcs[field](
                imgs=self.imgs,
                channel=channel,
                seg_masks=self.seg_masks,
                savedir=savedir,
                configs=self.plotter_configs,
                display=display,
            )
