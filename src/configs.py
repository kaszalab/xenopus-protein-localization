from skimage.morphology import disk, square

configs = {
    "img_glob": "*.tif",
    "seg_glob": "*.npy",
    "default_output_directory_name": "protein_localization_analysis",
    "tracking_features": [
        "centroid",
        "area",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "solidity",
        "moments_central",
    ],
    "tracking_search_radius": 60,
    "node_dilations": 5,
    "edge_dilations": 4,
    "cyto_mask_erode_steps": 10,
    # "cyto_mask_erode_footprint": disk,
    "cyto_mask_erode_footprint": square,
    "dx_microns": None,
    "dy_microns": None,
    "dt_seconds": None,
    "n_processes": 1,
}
