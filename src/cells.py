import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from tqdm import tqdm

from .configs import configs


def mask_of_cell_cytoplasms(cell_masks, footprint):
    """
    Erode all cell masks to cover only the cytoplasms in the current timepoint;
    for now this uses the same footprint for all cells
    """
    cyto_masks = np.zeros_like(cell_masks)
    for cell_region in regionprops(cell_masks):
        if cell_region.label == 0:
            continue
        _this_cell_mask = cell_masks == cell_region.label
        _this_cyto_mask = binary_erosion(_this_cell_mask, footprint)
        cyto_masks = cyto_masks + _this_cyto_mask * cell_region.label
    return cyto_masks


def measure_channel(img, cell_cyto_mask):
    """
    Measure the total, mean, and standard deviation of the pixel intensities
    in the cytoplasmic region of a single channel
    """
    intensity_mask = img.copy()
    intensity_mask[np.invert(cell_cyto_mask)] = 0
    total_intensity = intensity_mask.ravel().sum()
    mean_intensity = np.mean(np.ma.masked_equal(intensity_mask, 0))
    std_intensity = np.std(np.ma.masked_equal(intensity_mask, 0))
    return total_intensity, mean_intensity, std_intensity


def tabulate_cells(seg_masks, cyto_masks, img, frame_idx):
    """
    Tabulate cell morphology and cytoplasmic protein intensities for all cells _in the current timepoint_
    """

    cells_table = pd.DataFrame()

    _cell_masks = seg_masks

    for cell_region in tqdm(
        regionprops(_cell_masks),
        ascii=True,
        desc="Generating cells table for frame {}...".format(frame_idx),
    ):
        cell_cyto_mask = cyto_masks == cell_region.label

        row = {}

        # Assuming (C, H, W) for img
        for channel_idx in range(img.shape[0]):
            img_channel = img[channel_idx]
            total_intensity, mean_intensity, std_intensity = measure_channel(
                img_channel, cell_cyto_mask
            )
            row["Channel{}_total_intensity".format(channel_idx)] = total_intensity
            row["Channel{}_mean_intensity".format(channel_idx)] = mean_intensity
            row["Channel{}_std_intensity".format(channel_idx)] = std_intensity

        cyto_pts = np.argwhere(cell_cyto_mask)
        formatted_cell_pts = np.array([np.array(pair) for pair in cyto_pts]).astype(
            np.int64
        )

        row["cell_id"] = cell_region.label
        row["frame_id"] = frame_idx
        row["area"] = cell_region.area
        row["perimeter"] = cell_region.perimeter
        row["major_axis_length"] = cell_region.major_axis_length
        row["minor_axis_length"] = cell_region.minor_axis_length
        row["orientation"] = cell_region.orientation
        row["solidity"] = cell_region.solidity
        row["centroid"] = cell_region.centroid
        row["area_minus_convex_hull_area"] = cell_region.area - cell_region.area_convex
        row["cyto_probe_points"] = formatted_cell_pts

        row_df = pd.DataFrame([row])

        cells_table = pd.concat([cells_table, row_df], ignore_index=True)

    return cells_table


class CytoAnalyzer:
    """
    Tabulate cell morphology and cytoplasmic protein intensities for all timepoints in the movie
    """

    def __init__(self, imgs, seg_masks):
        """
        seg_masks: a cellpose-like non-overlapping labeling of cell interiors of shape (1, T, H, W)
        imgs: an array of intensity images of shape (C, T, H, W)
        """
        self.seg_masks = seg_masks
        self.imgs = imgs
        self.cyto_masks = {}
        self.cells_table = {}

        print(
            "CytoAnalyzer started with imgs: {} and seg masks: {}".format(
                self.imgs.shape, self.seg_masks.shape
            )
        )

    def run(self):
        """
        Tabulate cell morphology and cytoplasmic protein intensities
        for all cells in the current timepoint
        """

        erode_footprint = configs["cyto_mask_erode_footprint"](
            configs["cyto_mask_erode_steps"]
        )

        n_t = self.seg_masks.shape[1]

        for frame_idx in tqdm(
            range(n_t), ascii=True, desc="Generating cytoplasmic masks..."
        ):
            self.cyto_masks[frame_idx] = mask_of_cell_cytoplasms(
                self.seg_masks[0, frame_idx, :, :], erode_footprint
            )

            self.cells_table[frame_idx] = tabulate_cells(
                self.seg_masks[0, frame_idx, :, :],
                self.cyto_masks[frame_idx],
                self.imgs[:, frame_idx, :, :],
                frame_idx,
            )

        return self.cells_table
