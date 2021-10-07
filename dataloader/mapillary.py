# -*- coding: utf-8 -*-
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import copy
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms

from .Cityscapes import Cityscapes

ImageFile.LOAD_TRUNCATED_IMAGES = True

class mapillary(Cityscapes):
    def __init__(self,
                 list_path='./data_list/mapillary',
                 split='train',
                 crop_size=(1024, 512),
                 train=True,
                 numpy_transform=False
                 ):

        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True

        image_list_filepath = os.path.join(self.list_path, self.split + "_imgs.txt")
        label_list_filepath = os.path.join(self.list_path, self.split + "_labels.txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        self.labels = [id.strip() for id in open(label_list_filepath)]

        ignore_label = -1
        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                              6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                              13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18}

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.images), self.split))

    def __getitem__(self, item):

        image_path = self.images[item]
        image = Image.open(image_path).convert("RGB")

        gt_image_path = self.labels[item]
        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image
