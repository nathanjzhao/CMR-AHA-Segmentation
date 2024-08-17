import os
import hdf5storage as h5
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


SLICES = {
    "Base": 0,
    "Mid": 1,
    "Apex": 2,
}


class DataSet(Dataset):
    """
    Dataset processing with various affine data augmentation possibilities.
    Transforms for keypoint programmed from scratch using affine matrices
    """

    def __init__(
        self,
        data_folder,
        degrees=None,
        translate=None,
        scale=None,
        contrast=1,
        flipping=False,
        no_midpoint=False,
        filter_level=0,
        largest_size=256,
        use_mask=False,
    ):

        assert os.path.exists(data_folder), "Folder not present"

        self.data_folder = os.path.abspath(data_folder)

        # Filter the data files
        if filter_level == 0:
            self.data_files = os.listdir(data_folder)
        else:
            self.data_files = []
            temp_data_files = os.listdir(data_folder)
            for file in temp_data_files:
                data = h5.loadmat(os.path.join(self.data_folder, file))
                if filter_level == 1 and data["Notes"] != "Low Qual.":
                    self.data_files.append(file)
                elif filter_level == 2 and data["Notes"] not in [
                    "Low Qual.",
                    "Questionable Qual.",
                ]:
                    self.data_files.append(file)

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.contrast = contrast
        self.flipping = flipping
        self.largest_size = largest_size

        self.no_midpoint = no_midpoint
        self.use_mask = use_mask

    def __len__(self):
        return len(self.data_files)

    def affine_transform(
        self, image, label, mask=None, degrees=None, translate=None, scale=None
    ):
        degrees = 0 if degrees is None else degrees
        translate = 0 if translate is None else translate
        scale = 1.0 if scale is None else scale

        r, t, sc, sh = transforms.RandomAffine.get_params(
            degrees=(-degrees, degrees),
            translate=(translate, translate),
            scale_ranges=(1 / scale, scale),
            shears=(0, 0),
            img_size=image.size,
        )
        theta = np.radians(r)

        tx, ty = t
        cx, cy = image.size[0] / 2, image.size[1] / 2

        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        tx_center = cx - sc * (cx * cos_theta - cy * sin_theta)
        ty_center = cy - sc * (cx * sin_theta + cy * cos_theta)

        R = np.array(
            [[sc * cos_theta, sc * -sin_theta], [sc * sin_theta, sc * cos_theta]]
        )

        T = np.array([tx_center, ty_center])

        A = np.zeros((3, 3))
        A[:2, :2] = R
        A[:2, 2] = T
        A[2, 2] = 1

        A = torch.Tensor(A)

        num_points = 2 if self.no_midpoint else 3

        homogeneous_label = torch.cat((label, torch.ones(num_points, 1)), dim=1)
        transformed_label = torch.mm(A, homogeneous_label.t()).t()
        transformed_label = transformed_label[:, :2]
        transformed_label += torch.tensor([[ty, tx] for _ in range(num_points)]) # because keypoints dims swapped

        transformed_image = F.affine(
            image,
            angle=-r,
            translate=(tx, ty),
            scale=sc,
            shear=sh,
            fill=0,
            center=(cx, cy),
        )

        if mask is not None:
            transformed_mask = F.affine(
                mask,
                angle=-r,
                translate=(tx, ty),
                scale=sc,
                shear=sh,
                fill=0,
                center=(cx, cy),
            )
            return [transformed_image, transformed_mask, transformed_label]
        else:
            return [transformed_image, transformed_label]

    def __getitem__(self, index):
        data = h5.loadmat(os.path.join(self.data_folder, self.data_files[index]))

        pil_image = Image.fromarray(data["NiFTi"].astype(np.uint8))

        if self.use_mask:
            pil_mask = Image.fromarray(data["Mask"].astype(np.uint8))
        else:
            pil_mask = None

        # Pad image and mask (if used) until they are 256x256
        width, height = pil_image.size
        right_padding = max(0, self.largest_size - width)
        bottom_padding = max(0, self.largest_size - height)

        border = (0, 0, right_padding, bottom_padding)  # (left, top, right, bottom)
        pil_image = ImageOps.expand(pil_image, border)
        if pil_mask:
            pil_mask = ImageOps.expand(pil_mask, border)

        # Apply random contrast to image (not to mask)
        contrast_factor = random.uniform(1 / self.contrast, self.contrast)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)

        if self.no_midpoint:
            label = torch.Tensor(
                np.array([data["Anterior_RVIP"][0], data["Inferior_RVIP"][0]])
            )
        else:
            label = torch.Tensor(
                np.array(
                    [
                        data["Anterior_RVIP"][0],
                        data["Inferior_RVIP"][0],
                        data["Mid_Septum"][0],
                    ]
                )
            )

        # Randomly decide whether to flip the image, mask (if used), and points
        if self.flipping and random.choice([True, False]):
            pil_image = F.hflip(pil_image)
            if pil_mask:
                pil_mask = F.hflip(pil_mask)

            for i in range(len(label)):
                label[i][1] = pil_image.size[0] - label[i][1]

        if pil_mask:
            NifTi, Mask, label = self.affine_transform(
                pil_image, label, pil_mask, self.degrees, self.translate, self.scale
            )
        else:
            NifTi, label = self.affine_transform(
                pil_image, label, None, self.degrees, self.translate, self.scale
            )

        label = (label / pil_image.size[0]).flatten().to(torch.float32)
        NifTi = transforms.ToTensor()(NifTi)[0]

        additional_data = []
        if "MD" in data and "E1" in data:
            additional_data.extend([data["MD"], data["E1"]])

        if pil_mask:
            Mask = transforms.ToTensor()(Mask)[0]
            return (NifTi, Mask, label, *additional_data)
        else:
            return (NifTi, label, *additional_data)
