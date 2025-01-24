# Copyright (c) Xi Weng.
# Licensed under the Apache License, Version 2.0 (the "License");

import random
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.
        Args:
            img (Image): an image in the PIL.Image format.
        Returns:
            Image: solarized image.
        """
        return ImageOps.solarize(img)

class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

class ImageNetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        equalization_prob: float = 0.0,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        crop_size: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        """Class that applies Imagenet transformations.
        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            equalization_prob (float, optional): probability of applying equalization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.2.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
        """

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomApply([Equalization()], p=equalization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

class MuiltiCropDataset(datasets.ImageFolder):
    """Support two or numtiple views"""
    def __init__(
        self,
        data_path,
        args,
        return_index=False,
    ):
        super(MuiltiCropDataset, self).__init__(data_path)

        if args.size_dataset >= 0:
            self.samples = self.samples[:args.size_dataset]
        self.return_index = return_index

        weak_transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.crops_size[0],
                        scale=(args.crops_min_scale[0], args.crops_max_scale[0])),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        
        trans = [weak_transform]

        for i in range(len(args.crops_nmb)):
            trans.extend([ImageNetTransform(crop_size = args.crops_size[i], 
                                            min_scale = args.crops_min_scale[i], max_scale = args.crops_max_scale[i],
                                            solarization_prob = args.solarization_prob[i]) ] * args.crops_nmb[i])

        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, crops
        return crops
