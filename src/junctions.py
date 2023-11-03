import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, disk, label
from tqdm import tqdm

from .configs import configs
from .tissue_network import TissueNetwork

"""
    "edge": bi-cellular junction
    "node": multi-cellular junction
"""


def measure_channel(img, junction_mask):
    """
    Measure the total, mean, and std intensities of a given channel
    within a given mask
    """
    intensity_mask = img.copy()
    intensity_mask[np.invert(junction_mask)] = 0
    total_intensity = intensity_mask.ravel().sum()
    mean_intensity = np.mean(np.ma.masked_equal(intensity_mask, 0))
    std_intensity = np.std(np.ma.masked_equal(intensity_mask, 0))
    return total_intensity, mean_intensity, std_intensity


def compute_edge_length(edge_pts):
    """
    Sum of Euclidean distances between pairs of adjacent edge pixels.
    (this is known to overestimate the length slightly)
    """
    total_length = 0
    for idx in range(len(edge_pts) - 1):
        total_length = total_length + euclidean(edge_pts[idx], edge_pts[idx + 1])

    return total_length


def compute_edge_tortuosity(edge_pts):
    """
    Estimated edge length divided by the length of the straight line that connects
    the two ends.
    """
    p_start = edge_pts[0]
    p_end = edge_pts[-1]
    _euclid_length = euclidean(p_start, p_end)
    _total_length = compute_edge_length(edge_pts)
    return _total_length / _euclid_length


def mask_of_edge_nodes(edge_pts, num_dilations, shape, footprint=disk):
    """
    Create a (dilated) mask of the multicellular junctions (aka nodes)
    that are linked by the given edge.

    edge_pts: sorted pixel coordinates of the edge points; the first and last points represents the nodes
    num_dilations: number of binary dilations applied to the nodes to create the junction masks
    shape: shape of the output image
    footprint: the footprint used to dilate the nodes

    """

    footprint = footprint(num_dilations, dtype=bool)

    p_start = [int(x) for x in edge_pts[0]]
    p_end = [int(x) for x in edge_pts[-1]]

    nodes_mask = np.zeros(shape, dtype=bool)
    nodes_mask[p_start[0], p_start[1]] = True
    nodes_mask[p_end[0], p_end[1]] = True

    return binary_dilation(nodes_mask, footprint=footprint)


def tabulate_edges(
    graph=None,
    edge_labels=None,
    img=None,
    frame_idx=None,
    node_dilations=configs["node_dilations"],
    edge_dilations=configs["edge_dilations"],
):

    assert graph != None

    assert edge_labels != None

    img_shape = img.shape[1:]  # skip the channel axis
    n_channels = img.shape[0]

    edges_table = pd.DataFrame()
    edge_idx = 0

    for _idx, (s, e) in tqdm(
        enumerate(graph.edges()),
        ascii=True,
        desc="generating junctions tables for frame {}...".format(frame_idx),
    ):
        """
        Read the edge points and compute length
        """

        ps = graph[s][e]["pts"]

        start_centroid = [int(c) for c in graph.nodes()[s]["o"]]
        end_centroid = [int(c) for c in graph.nodes()[e]["o"]]

        edge_length = compute_edge_length(ps)
        edge_tortuosity = compute_edge_tortuosity(ps)

        """
            Create the edge mask and exclude the multicellular junctions (nodes) to make an edge 'probe'
        """

        nodes_mask = mask_of_edge_nodes(
            edge_pts=ps, num_dilations=node_dilations, shape=img_shape
        )
        edge_mask = np.zeros(img_shape, dtype=bool)
        for pt in ps:
            edge_mask[pt[0], pt[1]] = True

        probe_mask = np.bitwise_and(edge_mask, np.invert(nodes_mask))

        for _ in range(edge_dilations):
            probe_mask = binary_dilation(probe_mask)

        probe_mask = np.bitwise_and(probe_mask, np.invert(nodes_mask))
        probe_pts = np.argwhere(probe_mask)

        """
            Compute edge orientation
        """
        edge_regions = regionprops(label(edge_mask))

        # if len(edge_regions) != 2: print('invalid edge with multiple connected components found!')
        # assert(len(edge_regions)==2) # edge and background

        for region in edge_regions:
            if region.label == 0:
                continue
            else:
                edge_orientation = (
                    region.orientation
                )  # keep consistent with cell orientation

        row = {}

        """
            Compute the edge intensity statistics for each channel
        """
        # Assuming (C, H, W) for img
        for channel_idx in range(n_channels):
            img_channel = img[channel_idx]
            total_intensity, mean_intensity, std_intensity = measure_channel(
                img_channel, probe_mask
            )
            row["Channel{}_total_intensity".format(channel_idx)] = total_intensity
            row["Channel{}_mean_intensity".format(channel_idx)] = mean_intensity
            row["Channel{}_std_intensity".format(channel_idx)] = std_intensity

        formatted_pts = np.array([np.array(pair) for pair in ps])
        formatted_probe_pts = np.array([np.array(pair) for pair in probe_pts]).astype(
            np.int64
        )

        row["id"] = edge_idx
        row["tracking_id"] = None
        row["cell_ids"] = tuple(edge_labels[(s, e)])
        row["frame_id"] = frame_idx
        row["length_euclidean_pixels"] = edge_length
        row["length_euclidean_microns"] = None
        row["tortuosity"] = edge_tortuosity
        row["orientation"] = edge_orientation
        row["all_points"] = formatted_pts
        row["points"] = formatted_probe_pts
        row["num_points"] = len(formatted_probe_pts)

        row_df = pd.DataFrame([row])

        edges_table = pd.concat([edges_table, row_df], ignore_index=True)

        edge_idx = edge_idx + 1

    return edges_table


def tabulate_nodes(
    graph=None,
    node_labels=None,
    img=None,
    frame_idx=None,
    node_dilations=configs["node_dilations"],
):
    """
    Now do the same for multicellular junctions (nodes);
    note that we have two nodes for each edge that we would
    have to isolate first:
    The other issue is that having two nodes per edge will
    results in duplicates entries, which we would
    have to remove at the end
    """

    assert graph != None

    assert node_labels != None

    img_shape = img.shape[1:]  # skip the channel axis
    n_channels = img.shape[0]

    nodes_table = pd.DataFrame()
    edge_idx = 0
    node_idx = 0

    for _idx, (s, e) in tqdm(
        enumerate(graph.edges()),
        ascii=True,
        desc="generating junctions tables for frame {}...".format(frame_idx),
    ):
        """
        Read the edge points and compute length
        """

        ps = graph[s][e]["pts"]

        start_centroid = [int(c) for c in graph.nodes()[s]["o"]]
        end_centroid = [int(c) for c in graph.nodes()[e]["o"]]

        edge_length = compute_edge_length(ps)
        edge_tortuosity = compute_edge_tortuosity(ps)

        row = {}

        """
            Create the edge mask and exclude the multicellular junctions (nodes) to make an edge 'probe'
        """

        nodes_mask = mask_of_edge_nodes(
            edge_pts=ps, num_dilations=node_dilations, shape=img_shape
        )

        isolated_node_masks = []
        labeled_nodes_masks = label(nodes_mask)
        for region in regionprops(labeled_nodes_masks):
            if region.label == 0:
                continue
            isolated_node_masks.append(labeled_nodes_masks == region.label)

        for probe_mask in isolated_node_masks:
            node_centroid = np.array(
                np.mean(np.argwhere(probe_mask), axis=0), dtype=int
            )

            # to find the node cell labels, we need to determine if the current
            # node is the start or end node of the edge
            if start_centroid in np.argwhere(probe_mask):
                _node_cell_labels = node_labels[s]
            else:
                _node_cell_labels = node_labels[e]

            # Assuming (C, H, W) shape for imgs
            for channel_idx in range(n_channels):
                img_channel = img[channel_idx]
                total_intensity, mean_intensity, std_intensity = measure_channel(
                    img_channel, probe_mask
                )
                row["Channel{}_total_intensity".format(channel_idx)] = total_intensity
                row["Channel{}_mean_intensity".format(channel_idx)] = mean_intensity
                row["Channel{}_std_intensity".format(channel_idx)] = std_intensity

            node_pts = np.argwhere(probe_mask)
            formatted_node_pts = np.array([np.array(pair) for pair in node_pts]).astype(
                np.int64
            )

            row["tracking_id"] = None
            row["edge_id"] = edge_idx
            row["cell_ids"] = tuple(_node_cell_labels)
            row["frame_id"] = frame_idx
            row["num_pixels"] = len(node_pts)
            row["centroid_x"] = node_centroid[0]
            row["centroid_y"] = node_centroid[1]
            row["points"] = formatted_node_pts
            row["num_points"] = len(formatted_node_pts)

            row_df = pd.DataFrame([row])

            nodes_table = pd.concat([nodes_table, row_df], ignore_index=True)

            edge_idx = edge_idx + 1

        # now we drope the duplicate nodes
        nodes_table = nodes_table.drop_duplicates(subset="cell_ids")

    return nodes_table


class JunctionsAnalyzer:
    """ """

    def __init__(self, imgs, seg_masks):
        assert len(imgs.shape) == 4
        assert len(seg_masks.shape) == 4

        self.imgs = imgs
        self.seg_masks = seg_masks

        self.edge_tables = {}
        self.node_tables = {}

        self.networks = {}

    def generate_networks(self):
        """
        Generate a cell outline network for each frame
        """
        frame_ids = np.linspace(
            0, self.seg_masks.shape[1] - 1, self.seg_masks.shape[1]
        ).astype(np.uint8)

        for frame_idx in tqdm(
            frame_ids,
            ascii=True,
            desc="generating networks...",
        ):
            _seg_mask = self.seg_masks[0, frame_idx]
            self.networks[frame_idx] = TissueNetwork(_seg_mask).build()

    def tabulate_edges(self):
        frame_ids = np.linspace(
            0, self.seg_masks.shape[1] - 1, self.seg_masks.shape[1]
        ).astype(np.uint8)

        for frame_idx in tqdm(
            frame_ids,
            ascii=True,
            desc="generating bi-cellular junctions tables...",
        ):
            _img = self.imgs[:, frame_idx, :, :]

            self.edge_tables[frame_idx] = tabulate_edges(
                graph=self.networks[frame_idx]["graph"],
                edge_labels=self.networks[frame_idx]["edge_labels"],
                img=_img,
                frame_idx=frame_idx,
            )

        return self.edge_tables

    def tabulate_nodes(self):
        frame_ids = np.linspace(
            0, self.seg_masks.shape[1] - 1, self.seg_masks.shape[1]
        ).astype(np.uint8)

        for frame_idx in tqdm(
            frame_ids,
            ascii=True,
            desc="generating multi-cellular junctions tables...",
        ):
            _img = self.imgs[:, frame_idx, :, :]

            self.node_tables[frame_idx] = tabulate_nodes(
                graph=self.networks[frame_idx]["graph"],
                node_labels=self.networks[frame_idx]["node_labels"],
                img=_img,
                frame_idx=frame_idx,
            )

        return self.node_tables