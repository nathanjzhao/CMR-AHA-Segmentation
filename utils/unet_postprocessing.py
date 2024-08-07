import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tempfile
from PIL import Image


def get_centroids(mask, num_classes):
    centroids = {}

    # Move the mask to the CPU if it's on the GPU
    if mask.is_cuda:
        mask = mask.cpu()

    for value in range(1, num_classes):
        coords = np.where(mask == value)  # Get the coordinates where mask == value
        if (
            coords[0].size > 0 and coords[1].size > 0
        ):  # Check if the class exists in the mask
            centroid = (
                np.mean(coords[1]),
                np.mean(coords[0]),
            )  # Calculate the centroid
            centroids[value] = centroid

    return centroids


def generate_keypoint_image(mask_true, mask_pred, image, num_classes):
    # Calculate the centroids of the true labels and the predicted labels
    centroids_true = get_centroids(mask_true, num_classes)
    centroids_pred = get_centroids(mask_pred, num_classes)

    # Create a new figure
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image[0].float().cpu().numpy(), cmap="gray")

    # print(centroids_true)
    # Create a colormap
    cmap = plt.cm.get_cmap("hsv", num_classes)

    # Plot the centroids of the true mask
    for value, centroid in centroids_true.items():
        ax.scatter(*centroid, color=cmap(value), label=f"GT {value}", marker="+")

    # Plot the centroids of the predicted mask
    for value, centroid in centroids_pred.items():
        ax.scatter(*centroid, color=cmap(value), label=f"Prediced {value}", marker="x")

    # Create a list of Line2D objects for the legend
    legend_elements = [
        mlines.Line2D(
            [],
            [],
            color=cmap(1),
            marker=".",
            linestyle="None",
            markersize=10,
            label="Anterior",
        ),
        mlines.Line2D(
            [],
            [],
            color=cmap(2),
            marker=".",
            linestyle="None",
            markersize=10,
            label="Inferior",
        ),
    ]

    # Add a legend
    ax.legend(handles=legend_elements)

    # Save the figure to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name)
        # Return the path to the file
        return f.name, centroids_true, centroids_pred


def compile_masks(mask_true, num_classes):
    # Create a tensor of class numbers
    class_numbers = torch.arange(num_classes).view(-1, 1, 1).to(mask_true.device)

    # Multiply each class mask by its class number
    weighted_masks = mask_true * class_numbers

    # Take the maximum along the class dimension to create the single class mask
    compiled_mask = weighted_masks.max(dim=0)[0]

    return compiled_mask


def create_overlap_figure(red_mask, blue_mask, image_np):
    # Ensure both masks have the same shape
    if red_mask.shape != blue_mask.shape:
        raise ValueError("Both masks must have the same dimensions")

    # Create a new RGB array
    overlaps = np.zeros((*red_mask.shape, 3))

    # Move red_mask and blue_mask to CPU and convert to NumPy arrays
    red_mask_np = red_mask.cpu().numpy()
    blue_mask_np = blue_mask.cpu().numpy()

    # Set red channel
    overlaps[:, :, 0] = red_mask_np

    # Set blue channel
    overlaps[:, :, 2] = blue_mask_np

    # Calculate overlap (purple)
    overlap = np.logical_and(red_mask_np, blue_mask_np)
    overlaps[:, :, 0] = np.maximum(
        overlaps[:, :, 0], overlap
    )  # Red component of purple
    overlaps[:, :, 2] = np.maximum(
        overlaps[:, :, 2], overlap
    )  # Blue component of purple

    # Overlay the masks on the image
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
    if len(image_np.shape) == 2:  # if image is grayscale
        image_np = np.stack((image_np,) * 3, axis=-1)

    overlaps_scaled = overlaps * 255
    overlay_image_np = np.where(
        overlaps_scaled.sum(axis=-1, keepdims=True) != 0,
        overlaps_scaled,
        image_np * 255,
    )
    combined_image = Image.fromarray(overlay_image_np.astype(np.uint8))

    return combined_image


def calculate_mse(mask_true, mask_pred, num_classes):
    # Calculate the centroids of the true labels and the predicted labels
    centroids_true = get_centroids(mask_true, num_classes)
    centroids_pred = get_centroids(mask_pred, num_classes)

    # Calculate the squared distances for each class
    squared_distances = []
    for value, centroid_true in centroids_true.items():
        # If value is in centroids_pred, use the corresponding centroid
        # Otherwise, assume the centroid is at (0, 0)
        centroid_pred = centroids_pred.get(value, (0, 0))

        squared_distance = (centroid_true[0] - centroid_pred[0]) ** 2 + (
            centroid_true[1] - centroid_pred[1]
        ) ** 2
        squared_distances.append(squared_distance)

    # Calculate the mean squared error
    mse = np.mean(squared_distances)

    return mse


def calculate_mse_from_multiple_masks(true_masks, pred_masks, num_classes):
    # Go through different GT/Pred pairings
    total_mse = 0
    for true_mask, pred_mask in zip(true_masks, pred_masks):
        mse = calculate_mse(true_mask, pred_mask, num_classes)
        total_mse += mse

    return total_mse / len(true_masks)


def Myocardial_Mask_Contour_Extractor(myocardial_mask):
    """
    ########## Definition Inputs ##################################################################################################################
    # myocardial_mask       : Left or right ventricular binary mask (3D - [rows, columns, slices])
    # slice_number          : Index of slice to evaluate.
    ########## Definition Outputs #################################################################################################################
    # epicardial_contour    : Contour of the epicardium (outtermost layer) for current slice.
    # endocardial_contour   : Contour of the endocardium (innermost layer) for current slice.
    ########## Notes ##############################################################################################################################
    # This fucntion has been archived. Please use the more accurate definition named 'Myocardial_Mask_Contour_Extraction'.
    """
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2021
    ########## Import Modules #####################################################################################################################
    import numpy as np  # Import numpy module
    from skimage import measure  # Import measure from skimage module
    import cv2  # Import open CV module
    import warnings  # Import warning module

    ########## Print Warning ######################################################################################################################
    warnings.warn(
        "This fucntion has been archived. It is reccomended to use the more accurate definition named myocardial_mask_contour_extraction."
    )
    ########## Initalize data #####################################################################################################################
    rows = myocardial_mask.shape[0]  # Number of rows
    columns = myocardial_mask.shape[1]  # Number of columns
    binary = (
        (myocardial_mask * 255).to(torch.uint8).cpu().numpy()
    )  # Make image compadible with OpenCV
    epicardial_contour = np.zeros(([rows, columns]))  # Initialize epicardium
    endocardial_contour = np.zeros(([rows, columns]))  # Initialize endocardium
    ########## Detect edges from input myocardial mask ############################################################################################
    if np.mean(binary) != 0:  # If slice contains myocardium ...
        edges = cv2.Canny(binary, 0, 2)  # Edge detection with Canny filter
        all_labels = measure.label(edges)  # Label edges detected
        if 2 in all_labels:  # If 2 labels were detected ...
            # print('Slice contains endocardium and epicardium.' )            # Print information statment about input data
            epicardial_contour = all_labels == 1  # Identify epicardium contour
            endocardial_contour = all_labels == 2  # Identify endocardium contour
        # else:                                                                       # Otherwise only 1 label was detected and slice is ignored
        # print('Slice contains only epicardium.')                       # Print information statment about input data
    # else:                                                                       # Otherwise myocardium was not detected and slice is ignored
    # print('Slice contains no myocardium.')                        # Print information statment about input data
    return [epicardial_contour, endocardial_contour]


def hausdorff_distance(a_points, b_points):
    """Calculate the Hausdorff distance between nonzero elements of given images.
    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.
    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of nonzero pixels in
        ``image0`` and ``image1``, using the Euclidian distance.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=bool)
    >>> image_b = np.zeros(shape, dtype=bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_distance(image_a, image_b)
    3.0
    """
    from scipy.spatial import cKDTree

    #     a_points = np.transpose(np.nonzero(image0))
    #     b_points = np.transpose(np.nonzero(image1))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(
        max(cKDTree(a_points).query(b_points, k=1)[0]),
        max(cKDTree(b_points).query(a_points, k=1)[0]),
    )


def make_contours_then_hausdorff(mask_true, mask_pred):

    [UNet_Epi, UNet_Endo] = Myocardial_Mask_Contour_Extractor(mask_pred)
    [GT_Epi, GT_Endo] = Myocardial_Mask_Contour_Extractor(mask_true)

    epi_dist = hausdorff_distance(GT_Epi, UNet_Epi)
    endo_dist = hausdorff_distance(GT_Endo, UNet_Endo)
    return (epi_dist + endo_dist) / 2


if __name__ == "__main__":
    from unet_preprocessing import convert_labels_to_single_mask

    # print(get_centroids(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 3))
    # print(get_centroids(torch.tensor([[1, 0, 0], [2, 1, 2], [0, 2, 0]]), 3))

    labels1 = np.array(
        [
            [0.2145, 0.3985, 0.3631, 0.2708],
            [0.2652, 0.3616, 0.4100, 0.4322],
            [0.3655, 0.3238, 0.4658, 0.4262],
            [0.4430, 0.3567, 0.5033, 0.4109],
            [0.2990, 0.2610, 0.4164, 0.1872],
            [0.2892, 0.4398, 0.4035, 0.4572],
            [0.2365, 0.2938, 0.3641, 0.2179],
            [0.3263, 0.3345, 0.4751, 0.4063],
        ]
    )
    labels2 = np.array(
        [
            [0.3086, 0.4273, 0.3566, 0.2583],
            [0.1791, 0.3174, 0.3476, 0.2191],
            [0.0740, 0.5473, 0.2008, 0.2954],
            [0.3407, 0.2814, 0.4510, 0.3495],
            [0.0707, 0.2963, 0.2549, 0.1827],
            [0.3018, 0.3084, 0.4688, 0.1981],
            [0.1651, 0.4412, 0.3179, 0.3123],
            [0.2184, 0.3583, 0.3590, 0.2573],
        ]
    )
    # Assume `labels` is your labels

    mask1 = convert_labels_to_single_mask(labels1, 256, 256, 3)  # sigma
    mask2 = convert_labels_to_single_mask(labels2, 256, 256, 3)

    generate_keypoint_image(mask1[0, 0], mask2[0, 0], torch.rand(1, 1, 256, 256), 3)
    # print(mask1.shape)
    # # Calculate the number of classes
    # num_classes = len(np.unique(mask1))

    # # Calculate the MSE between the mask and itself
    # mse = calculate_mse(mask1, mask2, num_classes)

    # print(f"The MSE between the mask1 and mask2 is {mse}.")
