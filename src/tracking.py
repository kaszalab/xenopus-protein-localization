import os

import btrack
import numpy as np
from btrack import datasets
from tqdm import tqdm


def update_seg_masks_with_tracks(seg_masks, tracks):
    """
    You can also use the same function that ships with btrack now.
    """

    frame_ids = np.linspace(0, len(seg_masks) - 1, len(seg_masks)).astype(np.uint8)

    tracked_seg_masks = {}

    for frame_idx in frame_ids:
        tracked_seg_masks[frame_idx] = np.zeros_like(seg_masks[frame_idx])

    for tracklet in tqdm(tracks):
        frame_ids = tracklet.t
        cx = [int(x) for x in tracklet.x]
        cy = [int(y) for y in tracklet.y]
        label = tracklet.ID

        for idx, frame_idx in enumerate(frame_ids):
            old_label = seg_masks[frame_idx][cy[idx], cx[idx]]

            if old_label == 0:
                continue  # skip the background

            cell_mask = seg_masks[frame_idx] == old_label
            tracked_seg_masks[frame_idx][cell_mask] = label

    return tracked_seg_masks


def run_btrack(
    seg_interior_masks, FEATURES, search_radius, savedir, record_id="ZO1_test"
):
    """
    Track cells over time in a segmented movie
    """
    TRACKING_CONFIG_FILE = datasets.cell_config()

    segmentation = np.stack(seg_interior_masks, axis=0)

    objects = btrack.utils.segmentation_to_objects(
        segmentation,
        properties=tuple(FEATURES),
    )

    tracking_window = (
        (0, seg_interior_masks[0].shape[1]),
        (0, seg_interior_masks[0].shape[0]),
    )

    # savepath
    tracker_savepath = os.path.join(savedir, "cell_tracks_{}.h5".format(record_id))

    with btrack.BayesianTracker() as tracker:
        # configure the tracker using a config file
        tracker.configure_from_file(TRACKING_CONFIG_FILE)
        tracker.verbose = True
        tracker.max_search_radius = search_radius
        tracker.features = FEATURES

        # append the objects to be tracked
        tracker.append(objects)

        # set the tracking volume
        tracker.volume = tracking_window

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # store the tracks
        tracks = tracker.tracks

        # store the configuration
        # cfg = tracker.configuration

        # export the track data
        tracker.export(tracker_savepath, obj_type="obj_type_1")

    """
        Create tracked segmentation masks
    """
    tracked_seg_masks = update_seg_masks_with_tracks(seg_interior_masks, tracks)
    return tracked_seg_masks, tracks
