import logging
import torch
import numpy as np
import torch.nn.functional as F
from unet.unet_model import UNet
from utils.dataset import DataSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.unet_preprocessing import (
    convert_labels_to_single_mask,
)
from utils.unet_postprocessing import (
    calculate_mse,
    calculate_mse_from_multiple_masks,
    compile_masks,
    generate_keypoint_image,
    make_contours_then_hausdorff,
    create_overlap_figure,
)
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.AHA_segmenting import create_AHA_segmentation

from config import *
import mlflow
import mlflow.pytorch


@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    amp,
    mask_sigma,
    percentiles=[1, 25, 50, 75, 99],
    alpha=0.1,
    downstream=False,
):
    net.eval()

    num_val_batches = len(dataloader)
    scores = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for images, LV, labels, MD, E1 in dataloader:
            input_tensor = images[:, None, :, :]  # to match input channels
            point_masks = convert_labels_to_single_mask(
                labels, input_tensor.shape[2], input_tensor.shape[3], mask_sigma
            )  # radius

            # including the heart mask in the ground truth output mask
            non_overlap_mask = LV == 0
            mask_true = (
                point_masks * non_overlap_mask + (net.n_classes - 1) * ~non_overlap_mask
            )  # combined mask

            # Concatenate MD if used
            if use_MD:
                MD = MD[:, None, :, :]
                input_tensor = torch.cat([input_tensor, MD], dim=1)

            # Concatenate E1 if used
            if use_E1:
                E1 = E1.permute(0, 3, 1, 2)
                input_tensor = torch.cat([input_tensor, E1], dim=1)

            # move images and labels to correct device and type
            input_tensor = input_tensor.to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(input_tensor)

            if net.n_classes == 1:
                assert (
                    mask_true.min() >= 0 and mask_true.max() <= 1
                ), "True mask indices should be in [0, 1]"
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                mse_score += calculate_mse(mask_true, mask_pred, net.n_classes)
            else:
                assert (
                    mask_true.min() >= 0 and mask_true.max() < net.n_classes
                ), "True mask indices should be in [0, n_classes]"
                # convert to one-hot format
                mask_true = (
                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                )
                mask_pred = (
                    F.one_hot(mask_pred.argmax(dim=1), net.n_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )

                # compute the Dice score, ignoring background
                dice_score = multiclass_dice_coeff(
                    mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False
                )

                anterior_dice_score = multiclass_dice_coeff(
                    mask_pred[:, 1], mask_true[:, 1], reduce_batch_first=False
                )
                inferior_dice_score = multiclass_dice_coeff(
                    mask_pred[:, 2], mask_true[:, 2], reduce_batch_first=False
                )
                LV_dice_score = multiclass_dice_coeff(
                    mask_pred[:, 3], mask_true[:, 3], reduce_batch_first=False
                )

                weighted_dice_score = (
                    alpha * anterior_dice_score
                    + alpha * inferior_dice_score
                    + (1 - 2 * alpha) * LV_dice_score
                )

                # only one batch during evaluation, so [0] references image itself
                mse_score = calculate_mse_from_multiple_masks(
                    mask_true[0][1:-1], mask_pred[0][1:-1], net.n_classes - 1
                )  # torch.Size([1, 3, 256, 256])
                hausdorff_score = make_contours_then_hausdorff(
                    mask_true[0][-1], mask_pred[0][-1]
                )

                scores.append(
                    {
                        "weighted_dice_score": weighted_dice_score,
                        "dice_score": dice_score,
                        "mse_score": mse_score,
                        "hausdorff_score": hausdorff_score,
                        "anterior_dice_score": anterior_dice_score,
                        "inferior_dice_score": inferior_dice_score,
                        "LV_dice_score": LV_dice_score,
                        "mask_true": mask_true,
                        "mask_pred": mask_pred,
                        "image": images,
                    }
                )

    # Calculate percentile scores
    dice_scores = [score["dice_score"].cpu() for score in scores]
    percentile_dice_scores = [
        np.percentile(dice_scores, percentile) for percentile in percentiles
    ]

    # Initialize a dictionary to store the masks and images
    for i, percentile_dice_score in enumerate(percentile_dice_scores):
        # Find the entry with the dice score closest to the percentile score
        closest_entry = min(
            scores, key=lambda score: abs(score["dice_score"] - percentile_dice_score)
        )

        # Get the mask_true, mask_pred, and image from the closest entry
        mask_true = closest_entry["mask_true"]
        mask_pred = closest_entry["mask_pred"]
        image = closest_entry["image"]

        # Convert masks back into form of single mask (w/ numbers for each class)
        mask_true_compiled = compile_masks(mask_true[0], net.n_classes)
        mask_pred_compiled = compile_masks(mask_pred[0], net.n_classes)

        keypoint_file_path, GT_keypoints, pred_keypoints = generate_keypoint_image(
            mask_true_compiled, mask_pred_compiled, image, net.n_classes - 1
        )

        # Generate overlaps images
        image_np = image[0].float().cpu().numpy()
        anterior_overlaps = create_overlap_figure(
            mask_true[0, 1], mask_pred[0, 1], image_np
        )
        inferior_overlaps = create_overlap_figure(
            mask_true[0, 2], mask_pred[0, 2], image_np
        )
        LV_overlaps = create_overlap_figure(mask_true[0, 3], mask_pred[0, 3], image_np)

        # NOTE: have to get MD/FA from this + should include slice location
        # GT_AHA_segmentation = create_AHA_segmentation(
        #     GT_keypoints, mask_true[0, 3].float().cpu()
        # )
        # pred_AHA_segmentation = create_AHA_segmentation(
        #     pred_keypoints, mask_pred[0, 3].float().cpu()
        # )

        # Save figures to files
        mask_true_path = f"artifacts/mask_true_{percentiles[i]}.png"
        mask_pred_path = f"artifacts/mask_pred_{percentiles[i]}.png"
        image_path = f"artifacts/image_{percentiles[i]}.png"
        keypoint_path = f"artifacts/keypoints_{percentiles[i]}.png"
        anterior_overlaps_path = f"artifacts/anterior_overlaps_{percentiles[i]}.png"
        inferior_overlaps_path = f"artifacts/inferior_overlaps_{percentiles[i]}.png"
        LV_overlaps_path = f"artifacts/LV_overlaps_{percentiles[i]}.png"

        # Save the images (you might need to adjust this based on your image format)
        plt.imsave(mask_true_path, mask_true_compiled.cpu().numpy())
        plt.imsave(mask_pred_path, mask_pred_compiled.cpu().numpy())
        plt.imsave(image_path, image[0].float().cpu().numpy())
        plt.imsave(anterior_overlaps_path, anterior_overlaps)
        plt.imsave(inferior_overlaps_path, inferior_overlaps)
        plt.imsave(LV_overlaps_path, LV_overlaps)

        # Log the images as artifacts
        mlflow.log_artifact(mask_true_path)
        mlflow.log_artifact(mask_pred_path)
        mlflow.log_artifact(image_path)
        mlflow.log_artifact(keypoint_file_path)
        mlflow.log_artifact(anterior_overlaps_path)
        mlflow.log_artifact(inferior_overlaps_path)
        mlflow.log_artifact(LV_overlaps_path)

        # if GT_AHA_segmentation is not None and pred_AHA_segmentation is not None:
        #     GT_AHA_path = f"GT_AHA_{percentiles[i]}.png"
        #     pred_AHA_path = f"pred_AHA_{percentiles[i]}.png"
        #     plt.imsave(GT_AHA_path, GT_AHA_segmentation)
        #     plt.imsave(pred_AHA_path, pred_AHA_segmentation)
        #     mlflow.log_artifact(GT_AHA_path)
        #     mlflow.log_artifact(pred_AHA_path)

        # Log metrics
        mlflow.log_metric(f"dice_score_{percentiles[i]}", closest_entry["dice_score"])
        mlflow.log_metric(f"mse_score_{percentiles[i]}", closest_entry["mse_score"])
        mlflow.log_metric(
            f"hausdorff_score_{percentiles[i]}", closest_entry["hausdorff_score"]
        )
        mlflow.log_metric(
            f"anterior_dice_score_{percentiles[i]}",
            closest_entry["anterior_dice_score"],
        )
        mlflow.log_metric(
            f"inferior_dice_score_{percentiles[i]}",
            closest_entry["inferior_dice_score"],
        )
        mlflow.log_metric(
            f"LV_dice_score_{percentiles[i]}", closest_entry["LV_dice_score"]
        )

    N = max(num_val_batches, 1)
    for key in [
        "weighted_dice_score",
        "dice_score",
        "mse_score",
        "hausdorff_score",
        "anterior_dice_score",
        "inferior_dice_score",
        "LV_dice_score",
    ]:
        mlflow.log_metric(f"overall_{key}", sum([score[key] for score in scores]) / N)

    net.train()
    return sum([score["dice_score"] for score in scores]) / N


if __name__ == "__main__":

    no_midpoint = True
    bilinear = False

    n_channels = 1 + (1 if use_MD else 0) + (1 if use_E1 else 0)
    net = UNet(
        n_channels=n_channels, n_classes=3 if no_midpoint else 4, bilinear=bilinear
    )

    ########
    MODEL_PATH = "checkpoints/200checkpoint_epoch100.pth"
    TEST_DATA_PATH = "data/original_standard_labels/test"
    LARGEST_SIZE = 200
    ########

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {MODEL_PATH}")
    logging.info(f"Using device {device}")

    net.to(device=device)
    state_dict = torch.load(MODEL_PATH, map_location=device)

    mask_values = state_dict.pop("mask_values", [0, 1, 2])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    test_dataset = DataSet(
        TEST_DATA_PATH, no_midpoint=True, filter_level=0, largest_size=LARGEST_SIZE
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    val_score = evaluate(net, test_dataloader, device, False, 5, downstream=False)
