import numpy as np
import torch


def gaussian_2d(x, y, sigma, height, width):
    """Generate a 2D Gaussian distribution."""
    X, Y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    D2 = (X - x) ** 2 + (Y - y) ** 2
    return np.exp(-D2 / (2.0 * sigma**2))


def keypoint_to_gaussian(keypoints, height, width, sigma=1):
    """Convert keypoints to Gaussian distribution mask."""
    mask = np.zeros((height, width))

    for kp in keypoints:
        y, x = kp  # Keypoint coordinates
        # Ensure the keypoint is within the image bounds
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        gaussian = gaussian_2d(x, y, sigma, height, width)
        mask = np.maximum(mask, gaussian)  # Combine distributions

    # Normalize mask
    mask /= np.max(mask)
    return mask


def keypoint_to_radius(keypoints, height, width, i, radius=1):
    """Convert keypoints to radius mask."""
    mask = np.zeros((height, width))

    for kp in keypoints:
        y, x = kp  # Keypoint coordinates

        # Ensure the keypoint is within the image bounds
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))

        # Generate a grid of distances to the keypoint
        X, Y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        D2 = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

        # Set a circle of radius around the keypoint to i
        mask = np.where(D2 <= radius, i, mask)

    return mask


def convert_labels_to_gaussian_masks(labels, height, width, sigma=1):
    """Convert batch of labels to gaussian masks."""
    ret = []
    labels *= 256
    for label in labels:
        masks = []
        for i in range(len(label) // 2):
            one_point = label[i * 2 : i * 2 + 2]
            mask = keypoint_to_gaussian([one_point], height, width, sigma)
            masks.append(mask)
        ret.append(masks)

    return torch.tensor(ret)


def convert_labels_to_radial_masks(labels, height, width, radius=1):
    """Convert batch of labels to radial masks."""
    ret = []
    labels *= 200
    for label in labels:
        masks = []
        for i in range(len(label) // 2):
            one_point = label[i * 2 : i * 2 + 2]
            mask = keypoint_to_radius([one_point], height, width, 1, radius)
            masks.append(mask)
        ret.append(masks)

    return torch.tensor(ret)


# NOTE: THIS IS USED
# creates mask where all keypoints are combined, but without probability. mask of 1, 2, 3
def convert_labels_to_single_mask(labels, height, width, radius=1):
    """Convert batch of labels to single masks, based on radius around each point."""
    ret = []
    labels *= height

    assert (
        height == width
    ), "Current implementation multiplies labels by same amount (height). This is only valid for square images."

    for label in labels:
        mask = np.zeros((height, width))
        for i in range(len(label) // 2):
            one_point = label[i * 2 : i * 2 + 2]

            # Take the maximum between keypoint_to_radius and mask
            mask = np.maximum(
                mask, keypoint_to_radius([one_point], height, width, i + 1, radius)
            )

        ret.append(mask)

    return torch.tensor(np.array(ret))


if __name__ == "__main__":
    labels = np.array([[0.2, 0.2, 0.5, 0.3, 0.7, 0.7]])
    height = 256
    width = 256
    radius = 2

    masks = convert_labels_to_single_mask(labels, height, width, radius)

    import matplotlib.pyplot as plt

    # Save the masks as images
    for i, mask in enumerate(masks):
        plt.imshow(mask, cmap="gray")
        plt.savefig(f"mask_{i}.png")
