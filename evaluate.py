import logging
import os
from dotenv import load_dotenv
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from unet.unet_model import UNet
from utils.dataset import DataSet
from torch.utils.data import DataLoader
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
    wandb_logs = {}

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
        GT_AHA_segmentation = create_AHA_segmentation(
            GT_keypoints, mask_true[0, 3].float().cpu()
        )
        pred_AHA_segmentation = create_AHA_segmentation(
            pred_keypoints, mask_pred[0, 3].float().cpu()
        )

        # Store the masks and image in the results dictionary under the percentile key
        wandb_logs[f"validation {percentiles[i]}th percentile"] = {
            "masks_panel": [
                wandb.Image(mask_true_compiled),
                wandb.Image(mask_pred_compiled),
                wandb.Image(
                    image[0].float().cpu(),
                    caption=f"Overall Dice: {closest_entry['dice_score']:.2f}\nMSE: {closest_entry['mse_score']:.2f}\nHausdorff: {closest_entry['hausdorff_score']:.2f}",
                ),
                wandb.Image(
                    anterior_overlaps,
                    caption=f"Dice: {closest_entry['anterior_dice_score']:.2f}",
                ),
                wandb.Image(
                    inferior_overlaps,
                    caption=f"Dice: {closest_entry['inferior_dice_score']:.2f}",
                ),
                wandb.Image(
                    LV_overlaps, caption=f"Dice: {closest_entry['LV_dice_score']:.2f}"
                ),
            ],
            "mask_predictions_panel": [
                wandb.Image(mask_pred[0, 1].float().cpu()),
                wandb.Image(mask_pred[0, 2].float().cpu()),
                wandb.Image(mask_pred[0, 3].float().cpu()),
            ],
            # 'mask_true': wandb.Image(mask_true_compiled),
            # 'mask_pred': wandb.Image(mask_pred_compiled),
            # 'image': wandb.Image(image[0, 0].float().cpu()),
            # "scores": {
            #     "dice_score": closest_entry["dice_score"],
            #     "mse_score": closest_entry["mse_score"],
            #     "hausdorff_score": closest_entry["hausdorff_score"],
            #     "anterior_dice_score": closest_entry["anterior_dice_score"],
            #     "inferior_dice_score": closest_entry["inferior_dice_score"],
            #     "LV_dice_score": closest_entry["LV_dice_score"],
            # }
        }

        if GT_AHA_segmentation is not None and pred_AHA_segmentation is not None:
            print("not None!!!")
            wandb_logs[f"validation {percentiles[i]}th percentile"][
                "AHA_segmentations"
            ] = [wandb.Image(GT_AHA_segmentation), wandb.Image(pred_AHA_segmentation)]

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
        wandb_logs[key] = sum([score[key] for score in scores]) / N

    net.train()
    return wandb_logs["dice_score"], wandb_logs


if __name__ == "__main__":

    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")

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

    wandb.login(key=wandb_key)
    experiment = wandb.init(
        project="U-Net-evaluate", resume="allow", anonymous="allow", magic=True
    )
    val_score, wandb_logs = evaluate(
        net, test_dataloader, device, False, 5, downstream=False
    )

    wandb.log({"val score": val_score, **wandb_logs})
