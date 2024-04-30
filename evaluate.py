import logging
import os
from dotenv import load_dotenv
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from unet.unet_model import UNet
from utils.dataset import DataSet
from torch.utils.data import DataLoader
from utils.unet_preprocessing import convert_labels_to_radial_masks, convert_labels_to_single_mask
from utils.unet_postprocessing import calculate_rmse, compile_masks, generate_keypoint_image
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, mask_sigma, percentiles=[1, 25, 50, 75, 99]):
    net.eval()

    num_val_batches = len(dataloader)
    dice_score_total = 0
    rmse_score_total = 0
    scores = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for images, _, labels in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images = images[:, None, :, :]
            mask_true = convert_labels_to_single_mask(labels, images.shape[2], images.shape[3], mask_sigma) # radius

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(images)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                rmse_score += calculate_rmse(mask_true, mask_pred, net.n_classes)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # compute the Dice score, ignoring background
                dice_score = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                dice_score_total += dice_score

                # only one batch, so [0] references image itself
                rmse_score = calculate_rmse(mask_true[0][1:], mask_pred[0][1:], net.n_classes) # torch.Size([1, 3, 256, 256])
                rmse_score_total += rmse_score

                scores.append({"dice_score" : dice_score, "mse_score" : rmse_score, "mask_true" : mask_true, "mask_pred" : mask_pred, "image" : images})
   
    # Calculate percentile scores
    dice_scores = [score['dice_score'].cpu() for score in scores]
    percentile_dice_scores = [np.percentile(dice_scores, percentile) for percentile in percentiles]

    # Initialize a dictionary to store the masks and images
    percentile_images = {}

    for i, percentile_dice_score in enumerate(percentile_dice_scores):
        # Find the entry with the dice score closest to the percentile score
        closest_entry = min(scores, key=lambda score: abs(score['dice_score'] - percentile_dice_score))

        # Get the mask_true, mask_pred, and image from the closest entry
        mask_true = closest_entry['mask_true']
        mask_pred = closest_entry['mask_pred']
        image = closest_entry['image']

        # Convert masks back into form of single mask (w/ numbers for each class)
        mask_true_compiled = compile_masks(mask_true[0], net.n_classes)
        mask_pred_compiled = compile_masks(mask_pred[0], net.n_classes)

        keypoint_file_path = generate_keypoint_image(mask_true_compiled, mask_pred_compiled, image[0], net.n_classes)

        # Store the masks and image in the results dictionary under the percentile key
        percentile_images[f'validation {percentiles[i]}th percentile'] = {
            'mask_true': wandb.Image(mask_true_compiled),
            'mask_pred': wandb.Image(mask_pred_compiled),
            'mask_pred_anterior': wandb.Image(mask_pred[0, 1].float().cpu()),
            'mask_pred_inferior': wandb.Image(mask_pred[0, 2].float().cpu()),
            'image': wandb.Image(image[0, 0].float().cpu()),
            'keypoints': wandb.Image(keypoint_file_path)
        }

    net.train()
    return dice_score / max(num_val_batches, 1), rmse_score / max(num_val_batches, 1), percentile_images

if __name__ == "__main__":

    load_dotenv()
    wandb_key = os.getenv('WANDB_API_KEY')

    no_midpoint = True
    bilinear = False

    net = UNet(n_channels=1, n_classes=3 if no_midpoint else 4, bilinear=bilinear)

    ########
    MODEL_PATH = 'checkpoints/200checkpoint_epoch100.pth'
    TEST_DATA_PATH = 'data/original_standard_labels/test'
    LARGEST_SIZE = 200
    ########

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {MODEL_PATH}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(MODEL_PATH, map_location=device)

    mask_values = state_dict.pop('mask_values', [0, 1, 2])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    test_dataset = DataSet(TEST_DATA_PATH, no_midpoint=True, filter_level=0, largest_size=LARGEST_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    wandb.login(key=wandb_key)
    experiment = wandb.init(project='U-Net-evaluate', resume='allow', anonymous='allow', magic=True)
    dice, rmse, images = evaluate(net, test_dataloader, device, False, 5)

    wandb.log({'dice score': dice, 'rmse score': rmse, **images})
    