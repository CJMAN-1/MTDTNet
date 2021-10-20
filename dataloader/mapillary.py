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

class Mapillary(Cityscapes):
    def __init__(self,
                 list_path='./data_list/mapillary',
                 split='train',
                 crop_size=(1024, 512),
                 train=True,
                 numpy_transform=False,
                 super_class=False
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

        # ignore_label = -1
        if super_class:
            self.id_to_trainid = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 7: 0, 9: 0, 10: 0, 
                                  11: 0, 12: 0, 37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 
                                  45: 1, 47: 1, 51: 1, 52: 2, 54: 7, 55: 2, 56: 2, 
                                  57: 2, 58: 2, 65: 7, 70: 2, 74: 7, 75: 7, 80: 7, 
                                  82: 7, 87: 2, 90: 7, 91: 7, 92: 2, 93: 7, 94: 7, 
                                  95: 2, 97: 2, 103: 7, 104: 3, 105: 4, 106: 0, 
                                  111: 6, 112: 6, 113: 6, 114: 6, 115: 6}
        else:
            # TODO : generate 19 class matching dict
            self.id_to_trainid = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 7: 0, 9: 0, 10: 0, 
                                  11: 0, 12: 0, 37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 
                                  45: 1, 47: 1, 51: 1, 52: 2, 54: 7, 55: 2, 56: 2, 
                                  57: 2, 58: 2, 65: 7, 70: 2, 74: 7, 75: 7, 80: 7, 
                                  82: 7, 87: 2, 90: 7, 91: 7, 92: 2, 93: 7, 94: 7, 
                                  95: 2, 97: 2, 103: 7, 104: 3, 105: 4, 106: 0, 
                                  111: 6, 112: 6, 113: 6, 114: 6, 115: 6}
        

        print("{} num images in mapillary {} set have been loaded.".format(len(self.images), self.split))

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
