"""
    A set of methods to extract tissue networks from segmented ROIs
"""

import numpy as np
import sknw
from skimage.morphology import skeletonize, thin
from tqdm import tqdm


class TissueNetwork:
    """
    Convert cell segmentation masks to tissue network
    """

    def __init__(self, cell_interior_mask):
        """
        cell_interior_mask: a cellpose-like non-overlapping labeling of cell interiors
        """
        self.cell_interior_mask = cell_interior_mask
        self.network = {
            "skeleton": None,
            "graph": None,
            "edge_labels": None,
            "node_labels": None,
        }

    def skeletonize(self):
        self.network["skeleton"] = skeleton_from_labels(self.cell_interior_mask)

    def build_graph(self):
        self.network["graph"] = graph_from_skeleton(self.network["skeleton"])

    def assign_edge_labels(self):
        self.network["edge_labels"] = assign_edge_labels(
            self.cell_interior_mask, self.network["graph"]
        )

    def assign_node_labels(self):
        self.network["node_labels"] = assign_node_labels(
            self.network["graph"], self.network["edge_labels"]
        )

    def build(self):
        self.skeletonize()
        self.build_graph()
        self.assign_edge_labels()
        self.assign_node_labels()
        return self.network


def get_2_most_frequent_labels(labels, counts):
    """
    Having a list of labels and associated counts, sort the labels by frequency
    and pick the two most-frequent ones.
    """
    # negated array gives descending order
    descending_order_indices = np.argsort(-np.array(counts))
    return labels[descending_order_indices][0:2]


def _label_pixels_by_neighbor_number(cell_masks):
    """
    Assign an integer label to each pixel equal to the number of unique
    cell labels in its 3x3 neighborhood.
    """
    # keys are (row, col) tuples
    pixel_labels = {}

    _neighbor_number_mask = np.zeros_like(cell_masks)

    pad = 1  # don't change

    for i in tqdm(range(1, cell_masks.shape[0] - 1)):
        for j in range(1, cell_masks.shape[1] - 1):
            # searching in a square of side length 1+2*pad centered on the pixel
            pixel_labels[(i, j)] = np.unique(
                cell_masks[i - pad : i + pad, j - pad : j + pad]
            )
            # label the pixel by the number of unique neighbors
            _neighbor_number_mask[i, j] = len(pixel_labels[(i, j)])

    return _neighbor_number_mask


def _thin_and_skeletonize(mask):
    """
    Thin and skeletonize a mask of cell outlines
    """
    _thin_mask = thin(mask)
    return skeletonize(_thin_mask > 0)


def graph_from_skeleton(skel, prune=False):
    """
    Use sknw to find the nodes and edges of the tissue graph
    """
    graph = sknw.build_sknw(skel)

    if prune:
        min_node_degree = 3
        remove = [node for node, degree in graph.degree() if degree < min_node_degree]
        graph.remove_nodes_from(remove)

    return graph


def skeleton_from_labels(cell_masks):
    """
    Boundary pixels are pixels that have more than one unique neighbor
    """
    _neighbor_number_mask = _label_pixels_by_neighbor_number(cell_masks)
    return _thin_and_skeletonize(_neighbor_number_mask > 1)


def assign_edge_labels(cell_interior_mask, graph, pad=4):
    """
    Assign to each edge the cell labels that share it
    """
    edge_labels = {}

    for s, e in tqdm(graph.edges()):
        edge_points = graph[s][e]["pts"]

        # pick a point near the mid-point
        pivot = edge_points[int(len(edge_points) / 2)]

        cx = pivot[0]
        cy = pivot[1]

        overlap = cell_interior_mask[cx - pad : cx + pad, cy - pad : cy + pad].ravel()

        edge_labels[(s, e)] = overlap

        # we have all the overlapping labels now;
        # we only need the 2-most-frequent labels
        # (most likely to be the cells associated with this edge)
        labels, counts = np.unique(edge_labels[(s, e)], return_counts=True)

        # the 2 most likely labels are
        edge_labels[(s, e)] = get_2_most_frequent_labels(labels, counts)

    return edge_labels


def assign_node_labels(graph, edge_labels, background=0):
    """
    Assign to each node the labels of the cells that meet at that node
    """
    node_labels = {}

    node_ids = list(graph.nodes())
    for node_id in node_ids:
        node_labels[node_id] = []

    for edge in edge_labels:
        _node_ids = list(edge)  # the 2 nodes that this edge connects
        _labels = edge_labels[edge]  # the 2 cells that this edge connects
        for node_id in _node_ids:
            for l in _labels:
                node_labels[node_id].append(l)

    for node_id in node_labels.keys():
        # keep the unique values
        node_labels[node_id] = list(np.unique(node_labels[node_id]))

        # drop the zeros (background)
        if background in node_labels[node_id]:
            node_labels[node_id].remove(0)

    return node_labels
