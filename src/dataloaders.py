"""
    Helpful dataloaders.
"""

import glob
import os

import numpy as np
from skimage.io import imread


def load_data(
    channel_1_dirpath=None,
    channel_2_dirpath=None,
    seg_channel=0,
    # load_seg_outlines=False,
    img_glob="*.tif",
    seg_glob="*.npy",
    img_reader_plugin="tifffile",
    merge_channels=True,
):
    """
    Load the images for up to 2 channels and Cellpose segmentation masks for one of the channels.
    """
    channel_1_imgs = None
    channel_2_imgs = None
    interior_seg_masks = None
    # outline_seg_masks = None

    channels = []

    channel_1_img_filepaths = glob.glob(os.path.join(channel_1_dirpath, img_glob))
    channel_1_img_filepaths.sort()
    channel_1_imgs = [
        imread(path, plugin=img_reader_plugin) for path in channel_1_img_filepaths
    ]

    channels.append(channel_1_dirpath)

    if channel_2_dirpath:
        channel_2_img_filepaths = glob.glob(os.path.join(channel_2_dirpath, img_glob))
        channel_2_img_filepaths.sort()
        channels.append(channel_2_dirpath)
        channel_2_imgs = [
            imread(path, plugin=img_reader_plugin) for path in channel_2_img_filepaths
        ]

    seg_filepaths = glob.glob(os.path.join(channels[seg_channel], seg_glob))
    seg_filepaths.sort()
    interior_seg_masks = [
        np.load(path, allow_pickle=True).item()["masks"] for path in seg_filepaths
    ]
    # don't really need the outlines for now
    # if load_seg_outlines:
    #     outline_seg_masks = [
    #         np.load(path, allow_pickle=True).item()["outlines"] for path in seg_filepaths
    #     ]
    # else:
    #     outline_seg_masks = None

    if merge_channels:
        data = {
            "img": np.stack([channel_1_imgs, channel_2_imgs], axis=0),
            "seg": np.expand_dims(
                np.stack(interior_seg_masks, axis=0), axis=0
            ),  # add dummy channel dimension
        }
    else:
        data = {
            "img": {
                0: np.stack(channel_1_imgs, axis=0),
                1: np.stack(channel_2_imgs, axis=0),
            },
            "seg": np.stack(interior_seg_masks, axis=0),
        }

    return data
