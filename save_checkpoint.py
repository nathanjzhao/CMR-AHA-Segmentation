#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

import matplotlib.pyplot as plt
import os
import logging

from tqdm import tqdm
from unet import UNet
import wandb
from dotenv import load_dotenv

from utils.parse_arguments import parse_arguments
from utils.dataset import DataSet, SLICES
from utils.unet_preprocessing import convert_labels_to_single_mask
from utils.dice_score import dice_loss, tversky_loss
from utils.unet_postprocessing import generate_combined_mask, get_centroids
from evaluate import evaluate

from utils.models.RegressionCNN import *
from utils.models.Resnet import *

load_dotenv()
wandb_key = os.getenv('WANDB_API_KEY')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device {device}')


# In[2]:

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "validation score"},
    "parameters": {
        "lr" : {
            "values" : [0.008616298656403724]
        },
        "scheduler_patience": {
            "values" : [68]
        },
        "weight_decay": {
            "values" : [0.00002558737431840939]
        },
        "momentum": {
            "values" : [0.9459303619054844]
        },

        # HYPERPARAMETERS
        "sigma": {
            "values" : [6]
        },
        "rotation": {
            "values" : [20]
        },
        "translation": {
            "values" : [0.1296383518758189]
        },
        "scale": {
            "values" : [1.1727696678098958]
        },
        "contrast": {
            "values" : [1.9741978206717496]
        },
        "CE": {
            "values" : [0.706915516461399]
        },
        "tversky_beta": {
            "values" : [0.9219095710726336]
        },
    }
}


data_path = './data/original_standard_labels/train'
val_data_path = './data/original_standard_labels/val'

checkpoint_path = Path('./checkpoints/')
save_checkpoint = True

random_seed = 42
results_path = './model_tests'
no_midpoint = True
test_name_prefix = ""
filter_level = 0
record_spread = False

batch_size = 8
num_epochs = 100
# num_features = 8
# relu = False
# dropout = 0.5
# early_stopping_patience = 20

scheduler_patience = 20
lr = 1e-6
weight_decay = 1e-8
momentum = 0.999
bilinear = False
sigma = 5

flipping = False

amp = False


def train_model():

  experiment = wandb.init(project='U-Net', resume='allow', anonymous='allow', magic=True)
  
  lr = wandb.config.lr
  scheduler_patience = wandb.config.scheduler_patience
  weight_decay = wandb.config.weight_decay
  momentum = wandb.config.momentum

  rotation = wandb.config.rotation
  translation = wandb.config.translation
  scale = wandb.config.scale
  contrast = wandb.config.contrast

  CE = wandb.config.CE
  tversky_beta = wandb.config.tversky_beta
  sigma = wandb.config.sigma

  torch.manual_seed(random_seed)

  # TEST_NAME = f"lr_{lr}_wd_{weight_decay}_m_{momentum}_b_{bilinear}_mid_{no_midpoint}_s_{sigma}"
  # print(TEST_NAME)
      
  train_dataset = DataSet(data_path, degrees=rotation, translate=translation, scale=scale, contrast=contrast, flipping=flipping, no_midpoint=no_midpoint, filter_level=filter_level, largest_size=200)
  val_dataset = DataSet(val_data_path, no_midpoint=no_midpoint, filter_level=filter_level, largest_size=200)
  n_train, n_val = len(train_dataset), len(val_dataset)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


  # extra class for background
  unet = UNet(n_channels=1, n_classes=3 if no_midpoint else 4, bilinear=bilinear)
  unet = unet.to(device)

  criterion = nn.CrossEntropyLoss(ignore_index=0)
  optimizer = optim.RMSprop(unet.parameters(),
              lr=lr, weight_decay=weight_decay, momentum=momentum, foreach=True)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=scheduler_patience)  # goal: maximize Dice score
  grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
  criterion = nn.CrossEntropyLoss() if unet.n_classes > 1 else nn.BCEWithLogitsLoss()
  global_step = 0

  logging.info(f'''Starting training:
      Epochs:          {num_epochs}
      Batch size:      {batch_size}
      Training size:   {n_train}
      Validation size: {n_val}
      Checkpoints:     {save_checkpoint}
      Device:          {device.type}
      Mixed Precision: {amp}

      
      Learning rate:   {lr}
      Weight decay:    {weight_decay}
      Momentum:        {momentum}
      Scheduler patience: {scheduler_patience}

      CE: {CE}
      Tversky beta: {tversky_beta}
      Filter level: {filter_level}

      Rotation: {rotation}
      Translation: {translation}
      Scale: {scale}
      Contrast: {contrast}
  ''')


  for epoch in range(1, num_epochs + 1):
          unet.train()
          epoch_loss = 0
          with tqdm(total=n_train, desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
              for images, _, labels in train_dataloader:
                  images = images[:, None, :, :] # to match input channels
                  true_masks = convert_labels_to_single_mask(labels, images.shape[2], images.shape[3], sigma) # radius

                  assert images.shape[1] == unet.n_channels,                     f'Network has been defined with {unet.n_channels} input channels, '                     f'but loaded images have {images.shape[1]} channels. Please check that '                     'the images are loaded correctly.'
                  
                  images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                  true_masks = true_masks.to(device=device, dtype=torch.long)

                  with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                      masks_pred = unet(images)
                      if unet.n_classes == 1:
                          loss = criterion(masks_pred.squeeze(1), true_masks.float())
                          loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                      else:
                          CE_loss = criterion(masks_pred, true_masks)
                          
                          wandb.log({'cross entropy loss': CE_loss})
                          loss = CE * CE_loss

                          # Only tversky/dice on non-background classes
                          if 0 < tversky_beta < 1:
                              tversky = tversky_loss(
                                  F.softmax(masks_pred, dim=1)[:, 1:].float(),
                                  F.one_hot(true_masks, unet.n_classes).permute(0, 3, 1, 2)[:, 1:].float(),
                                  multiclass=True,
                                  alpha=(1-tversky_beta),
                                  beta=tversky_beta
                              )
                              loss += (1 - CE) * tversky
                              wandb.log({'tversky loss': tversky})
                          else:
                              dice = dice_loss(
                                  F.softmax(masks_pred, dim=1)[:, 1:].float(),
                                  F.one_hot(true_masks, unet.n_classes).permute(0, 3, 1, 2)[:, 1:].float(),
                                  multiclass=True
                              )

                              loss += (1 - CE) * dice
                              wandb.log({'dice loss': dice})

                  optimizer.zero_grad(set_to_none=True)
                  grad_scaler.scale(loss).backward()
                  grad_scaler.unscale_(optimizer)
                  torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                  grad_scaler.step(optimizer)
                  grad_scaler.update()

                  pbar.update(images.shape[0])
                  global_step += 1
                  epoch_loss += loss.item()

                  wandb.log({
                      'train loss': loss.item(),
                      'step': global_step,
                      'epoch': epoch
                  })
                  
                  pbar.set_postfix(**{'loss (batch)': loss.item()})

                  # Evaluation round
                  division_step = (n_train // (5 * batch_size)) # change variable for frequency
                  if division_step > 0:
                      if global_step % division_step == 0:
                            histograms = {}
                            for tag, value in unet.named_parameters():
                                tag = tag.replace('/', '.')
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                            
                            val_score, rmse_score, percentile_images = evaluate(unet, val_dataloader, device, amp, sigma)
                            scheduler.step(val_score)

                            logging.info('Validation Dice score: {}'.format(val_score))
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation score': val_score,
                                    'rmse score': rmse_score,
                                    'images': wandb.Image(images[0].cpu()),
                                    'train masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms,
                                    **percentile_images
                                })

                                generate_combined_mask(true_masks[0], masks_pred.argmax(dim=1)[0], images[0], unet.n_classes, experiment)
                            except Exception as e:
                                print(f"An error occurred: {e}")

          
          if save_checkpoint:
              Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
              state_dict = unet.state_dict()
              torch.save(state_dict, str(checkpoint_path / '200checkpoint_epoch{}.pth'.format(epoch)))
              logging.info(f'Checkpoint {epoch} saved!')

wandb.login(key=wandb_key)
sweep_id = wandb.sweep(sweep_config, project="U-Net-get-checkpoint4.29")
wandb.agent(sweep_id, train_model, count=1)
