#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

import os
import logging

from tqdm import tqdm
from unet import UNet
import wandb
from dotenv import load_dotenv

from utils.dataset import DataSet
from utils.unet_preprocessing import convert_labels_to_single_mask
from utils.dice_score import dice_loss, tversky_loss
from evaluate import evaluate

from utils.models.RegressionCNN import *
from utils.models.Resnet import *
from config import *

import optuna
import mlflow
import mlflow.pytorch

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")


def train_model(trial):

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    scheduler_patience = trial.suggest_int('scheduler_patience', 5, 50)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    momentum = trial.suggest_uniform('momentum', 0.9, 0.999)
    rotation = trial.suggest_int('rotation', 0, 30)
    translation = trial.suggest_uniform('translation', 0.1, 1.0)
    scale = trial.suggest_uniform('scale', 1.0, 1.5)
    contrast = trial.suggest_uniform('contrast', 1.0, 2.0)
    CE = trial.suggest_uniform('CE', 0.5, 1.0)
    tversky_beta = trial.suggest_uniform('tversky_beta', 0.5, 0.9)
    

    with mlflow.start_run(nested=True):
        # Log hyperparameters
        mlflow.log_params(trial.params)

        torch.manual_seed(random_seed)

        train_dataset = DataSet(
            data_path,
            degrees=rotation,
            translate=translation,
            scale=scale,
            contrast=contrast,
            flipping=flipping,
            no_midpoint=no_midpoint,
            filter_level=filter_level,
            largest_size=200,
            use_mask=True,
        )
        val_dataset = DataSet(
            val_data_path,
            no_midpoint=no_midpoint,
            filter_level=filter_level,
            largest_size=200,
            use_mask=True,
        )
        n_train, n_val = len(train_dataset), len(val_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        best_val_score = 0
        best_model_path = Path(checkpoint_path) / "best_model.pth"

        # extra class for background
        n_channels = 1 + (1 if use_MD else 0) + (3 if use_E1 else 0)
        unet = UNet(
            n_channels=n_channels, n_classes=4 if no_midpoint else 5, bilinear=bilinear
        )  # NOTE: more classes w/ heart mask included
        unet = unet.to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.RMSprop(
            unet.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=True,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=scheduler_patience
        )  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss() if unet.n_classes > 1 else nn.BCEWithLogitsLoss()
        global_step = 0

        logging.info(
            f"""Starting training:
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

            Use MD: {use_MD}
            Use E1: {use_E1}
        """
        )

        for epoch in range(1, num_epochs + 1):
            unet.train()
            epoch_loss = 0
            with tqdm(
                total=n_train, desc=f"Epoch {epoch}/{num_epochs}", unit="img"
            ) as pbar:
                for images, heart_mask, labels, MD, E1 in train_dataloader:
                    input_tensor = images[:, None, :, :]  # Add channel dimension

                    # convering keypoints into a single mask
                    point_masks = convert_labels_to_single_mask(
                        labels, input_tensor.shape[2], input_tensor.shape[3], sigma
                    )  # radius

                    # including the heart mask in the ground truth output mask
                    non_overlap_mask = heart_mask == 0
                    mask_true = (
                        point_masks * non_overlap_mask
                        + (unet.n_classes - 1) * ~non_overlap_mask
                    )  # combined mask

                    # Concatenate MD if used
                    if use_MD:
                        MD = MD[:, None, :, :]
                        input_tensor = torch.cat([input_tensor, MD], dim=1)

                    # Concatenate E1 if used
                    if use_E1:
                        E1 = E1.permute(0, 3, 1, 2)
                        input_tensor = torch.cat([input_tensor, E1], dim=1)

                    assert input_tensor.shape[1] == unet.n_channels, (
                        f"Network has been defined with {unet.n_channels} input channels, "
                        f"but loaded images have {images.shape[1]} channels. Please check that "
                        "the images are loaded correctly."
                    )

                    input_tensor = input_tensor.to(
                        device=device,
                        dtype=torch.float32,
                        memory_format=torch.channels_last,
                    )
                    mask_true = mask_true.to(device=device, dtype=torch.long)

                    with torch.autocast(
                        device.type if device.type != "mps" else "cpu", enabled=amp
                    ):
                        mask_pred = unet(input_tensor)
                        if unet.n_classes == 1:
                            loss = criterion(mask_pred.squeeze(1), mask_true.float())
                            loss += dice_loss(
                                F.sigmoid(mask_pred.squeeze(1)),
                                mask_true.float(),
                                multiclass=False,
                            )
                        else:
                            CE_loss = criterion(mask_pred, mask_true)

                            mlflow.log_metric("train cross entropy loss", CE_loss, step=global_step)
                            loss = CE * CE_loss

                            # Only tversky/dice on non-background classes
                            if 0 < tversky_beta < 1:
                                tversky = tversky_loss(
                                    F.softmax(mask_pred, dim=1)[:, 1:].float(),
                                    F.one_hot(mask_true, unet.n_classes)
                                    .permute(0, 3, 1, 2)[:, 1:]
                                    .float(),
                                    multiclass=True,
                                    alpha=(1 - tversky_beta),
                                    beta=tversky_beta,
                                )
                                loss += (1 - CE) * tversky
                                mlflow.log_metric("train tversky loss", tversky, step=global_step)
                            else:
                                dice = dice_loss(
                                    F.softmax(mask_pred, dim=1)[:, 1:].float(),
                                    F.one_hot(mask_true, unet.n_classes)
                                    .permute(0, 3, 1, 2)[:, 1:]
                                    .float(),
                                    multiclass=True,
                                )

                                loss += (1 - CE) * dice
                                mlflow.log_metric("train dice loss", dice, step=global_step)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()

                    mlflow.log_metric("train loss", loss.item(), step=global_step)

                    pbar.set_postfix(**{"train loss (batch)": loss.item()})

                    # Evaluation round
                    division_step = n_train // (
                        5 * batch_size
                    )  # change variable for frequency
                    if division_step > 0:
                        if global_step % division_step == 0:
                            histograms = {}
                            # for tag, value in unet.named_parameters():
                            #     tag = tag.replace("/", ".")
                            #     if not (torch.isinf(value) | torch.isnan(value)).any():
                            #         histograms["Weights/" + tag] = wandb.Histogram(
                            #             value.data.cpu()
                            #         )
                            #     if not (
                            #         torch.isinf(value.grad) | torch.isnan(value.grad)
                            #     ).any():
                            #         histograms["Gradients/" + tag] = wandb.Histogram(
                            #             value.grad.data.cpu()
                            #         )

                            val_score, eval_wandb_logs = evaluate(
                                unet, val_dataloader, device, amp, sigma
                            )
                            scheduler.step(val_score)

                            mlflow.log_metric("val_score", val_score, step=epoch)
                            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)
                            logging.info("Validation Dice score: {}".format(val_score))

                            if val_score > best_val_score:
                                best_val_score = val_score
                                torch.save(unet.state_dict(), str(best_model_path))
                                logging.info(f"New best model saved with validation score: {val_score}")
                                mlflow.log_artifact(str(best_model_path))

                            mlflow.log_metric("best_val_score", best_val_score)

            if save_checkpoint and epoch % 10 == 0:
                Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
                state_dict = unet.state_dict()
                torch.save(
                    state_dict,
                    str(checkpoint_path / "checkpoint_epoch{}.pth".format(epoch)),
                )
                logging.info(f"Checkpoint {epoch} saved!")

        return best_val_score




def main():
    mlflow.set_experiment("U-Net-Optimization-3")

    study = optuna.create_study(direction='maximize')
    study.optimize(train_model, n_trials=30)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Log the best trial information
    with mlflow.start_run():
        mlflow.log_params(trial.params)
        mlflow.log_metric("best_val_score", trial.value)
        
        best_model_path = Path(checkpoint_path) / f"best_model_trial_{trial.number}.pth"
        mlflow.log_artifact(str(best_model_path))
        mlflow.pytorch.log_model(torch.load(best_model_path), "best_model")

    print(f"Best model saved to {best_model_path}")

if __name__ == "__main__":
    main()
    # project_name = "U-Net-8.11"
    # wandb.login(key=wandb_key)
    # sweep_id = wandb.sweep(sweep_config, project=project_name)
    # wandb.agent(sweep_id, train_model, count=30)

    # best_run = wandb.Api().sweep(sweep_id).best_run()
    # best_model_path = Path(checkpoint_path) / f"best_model_{best_run.name}.pth"
    # best_run.file(f"best_model-{project_name}.pth").download(
    #     root=str(Path(checkpoint_path)), replace=True
    # )
    # os.rename(Path(checkpoint_path) / f"best_model-{project_name}.pth", best_model_path)
    # print(f"Best model saved to {best_model_path}")
