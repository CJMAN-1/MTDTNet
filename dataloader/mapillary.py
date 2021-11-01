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
        # print(image_list_filepath)

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        self.labels = [id.strip() for id in open(label_list_filepath)]

        if not super_class:
            self.id_to_trainid = { 13 : 0, 24 : 0, 41 : 0,
                                   2 : 1, 15 : 1,
                                   17 : 2,
                                   6 : 3,
                                   3 : 4,
                                   45 : 5, 47 : 5,
                                   48 : 6, 
                                   50 : 7, 
                                   30 : 8,
                                   29 : 9,
                                   27 : 10,
                                   19 : 11,
                                   20 : 12, 21 : 12, 22 : 12,
                                   55 : 13,
                                   61 : 14,
                                   54 : 15,
                                   58 : 16,
                                   57 : 17,
                                   52 : 18,
                                 }
        else:
            self.id_to_trainid = {
                # [165, 42, 42] : ,
                # [0, 192, 0] : ,
                2 : 1,# curb
                3 : 1,# fence
                4 : 1,
                5 : 1,
                6 : 1,# wall
                7 : 0,
                8 : 0,
                9 : 0,
                10 : 0,
                11 : 0,
                12 : 0,
                13 : 0, # road
                14 : 0,
                15 : 0, # sidewalk
                16 : 1,
                17 : 1,   # building
                18 : 1,
                19 : 5, # person
                20 : 5,   # rider
                21 : 5,
                22 : 5,
                23 : 0,
                24 : 0,  # road
                # [64, 170, 64],
                # [128, 64, 64],
                27 : 4,# sky
                # [255, 255, 255],
                29 : 0,# terrain
                30 : 3, # vegatation
                # [0, 170, 30],
                # [255, 255, 128],
                # [250, 0, 30],
                # [0, 0, 0],
                # [220, 220, 220],
                # [170, 170, 170],
                # [222, 40, 40],
                # [100, 170, 30],
                # [40, 40, 40],
                # [33, 33, 33],
                # [170, 170, 170] : 0,# road
                # [0, 0, 142],
                43 : 2,
                44 : 2,
                45 : 2,# pole
                46 : 2,
                47 : 2,    # pole
                48 : 2, # traffic light
                49 : 2,
                50 : 2,  # traffic sign (front only)
                # [180, 165, 180],
                52 : 6, # bicycle
                53 : 6,
                54 : 6,  # bus
                55 : 6,   # car
                56 : 6,
                57 : 6,   # motorcycle
                58 : 6,  # train
                59 : 6,
                60 : 6,
                61 : 6,    # truck
                62 : 6,
                # [32, 32, 32],
                # [0, 0, 0],
                # [0, 0, 0],
            }

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
