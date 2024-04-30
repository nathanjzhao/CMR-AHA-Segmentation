import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tempfile
import wandb

def get_centroids(mask, num_classes):
    centroids = {}

    # Move the mask to the CPU if it's on the GPU
    if mask.is_cuda:
        mask = mask.cpu()

    for value in range(1, num_classes + 1):
        coords = np.where(mask == value)  # Get the coordinates where mask == value
        if coords[0].size > 0 and coords[1].size > 0:  # Check if the class exists in the mask
            centroid = (np.mean(coords[1]), np.mean(coords[0]))  # Calculate the centroid
            centroids[value] = centroid

    return centroids

def generate_keypoint_image(mask_true, mask_pred, image, num_classes):
    # Calculate the centroids of the true labels and the predicted labels
    centroids_true = get_centroids(mask_true, num_classes)
    centroids_pred = get_centroids(mask_pred, num_classes)

    # Create a new figure
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image[0].float().cpu().numpy(), cmap='gray')

    # Create a colormap
    cmap = plt.cm.get_cmap('hsv', num_classes)
    
    # Plot the centroids of the true mask
    for value, centroid in centroids_true.items():
        ax.scatter(*centroid, color=cmap(value), label=f'GT {value}', marker='+')
    
    # Plot the centroids of the predicted mask
    for value, centroid in centroids_pred.items():
        ax.scatter(*centroid, color=cmap(value), label=f'Prediced {value}', marker='x')

    # Create a list of Line2D objects for the legend
    legend_elements = [mlines.Line2D([], [], color=cmap(1), marker='.', linestyle='None', markersize=10, label='Anterior'),
                    mlines.Line2D([], [], color=cmap(2), marker='.', linestyle='None', markersize=10, label='Inferior')]

    # Add a legend
    ax.legend(handles=legend_elements)

    # Save the figure to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name)
        # Return the path to the file
        return f.name
    
def compile_masks(mask_true, num_classes):
    # Create a tensor of class numbers
    class_numbers = torch.arange(num_classes).view(-1, 1, 1).to(mask_true.device)

    # Multiply each class mask by its class number
    weighted_masks = mask_true * class_numbers

    # Take the maximum along the class dimension to create the single class mask
    compiled_mask = weighted_masks.max(dim=0)[0]

    return compiled_mask


def calculate_rmse(mask_true, mask_pred, num_classes):
    # Calculate the centroids of the true labels and the predicted labels
    centroids_true = get_centroids(mask_true, num_classes)
    centroids_pred = get_centroids(mask_pred, num_classes)

    # Calculate the squared distances for each class
    squared_distances = []
    for value, centroid_true in centroids_true.items():
        # If value is in centroids_pred, use the corresponding centroid
        # Otherwise, assume the centroid is at (0, 0)
        centroid_pred = centroids_pred.get(value, (0, 0))
        
        squared_distance = (centroid_true[0] - centroid_pred[0])**2 + (centroid_true[1] - centroid_pred[1])**2
        squared_distances.append(np.sqrt(squared_distance))

    # Calculate the mean squared error
    mse = np.mean(squared_distances)

    return mse

def calculate_rmse_from_multiple_masks(true_masks, pred_masks, num_classes):
    # Calculate the mean squared error between the true masks and the predicted masks
    total_mse = 0
    for true_mask, pred_mask in zip(true_masks, pred_masks):
        mse = calculate_rmse(true_mask, pred_mask, num_classes)
        total_mse += mse

    return total_mse / len(true_masks)

if __name__ == "__main__":
    from unet_preprocessing import convert_labels_to_single_mask

    # print(get_centroids(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 3))
    # print(get_centroids(torch.tensor([[1, 0, 0], [2, 1, 2], [0, 2, 0]]), 3))

    labels1 = np.array([[0.2145, 0.3985, 0.3631, 0.2708],
        [0.2652, 0.3616, 0.4100, 0.4322],
        [0.3655, 0.3238, 0.4658, 0.4262],
        [0.4430, 0.3567, 0.5033, 0.4109],
        [0.2990, 0.2610, 0.4164, 0.1872],
        [0.2892, 0.4398, 0.4035, 0.4572],
        [0.2365, 0.2938, 0.3641, 0.2179],
        [0.3263, 0.3345, 0.4751, 0.4063]])
    labels2 = np.array([[0.3086, 0.4273, 0.3566, 0.2583],
        [0.1791, 0.3174, 0.3476, 0.2191],
        [0.0740, 0.5473, 0.2008, 0.2954],
        [0.3407, 0.2814, 0.4510, 0.3495],
        [0.0707, 0.2963, 0.2549, 0.1827],
        [0.3018, 0.3084, 0.4688, 0.1981],
        [0.1651, 0.4412, 0.3179, 0.3123],
        [0.2184, 0.3583, 0.3590, 0.2573]])
    # Assume `labels` is your labels

    mask1 = convert_labels_to_single_mask(labels1, 256, 256, 3) # sigma
    mask2 = convert_labels_to_single_mask(labels2, 256, 256, 3)

    generate_keypoint_image(mask1[0, 0], mask2[0, 0], torch.rand(1, 1, 256, 256), 3)
    # print(mask1.shape)
    # # Calculate the number of classes
    # num_classes = len(np.unique(mask1))

    # # Calculate the MSE between the mask and itself
    # rmse = calculate_rmse(mask1, mask2, num_classes)

    # print(f"The MSE between the mask1 and mask2 is {rmse}.")

