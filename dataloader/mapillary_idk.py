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
                                  45: 1, 47: 1, 51: 1, 52: 2, 54: 6, 55: 2, 56: 2, 
                                  57: 2, 58: 2, 65: 6, 70: 2, 74: 6, 75: 6, 80: 6, 
                                  82: 6, 87: 2, 90: 6, 91: 6, 92: 2, 93: 6, 94: 6, 
                                  95: 2, 97: 2, 103: 6, 104: 3, 105: 4, 106: 0, 
                                  111: 5, 112: 5, 113: 5, 114: 5, 115: 5}
        else:
            color_matching = {
                [196, 196, 196] : 1,# sidewalk
                [190, 153, 153] : 4,# fence
                [102, 102, 156] : 3,# wall
                [128, 64, 128] : 0, # road
                [244, 35, 232] : 1, # sidewalk
                [70, 70, 70] : 2,   # building
                [220, 20, 60] : 11, # person
                [255, 0, 0] : 12,   # rider
                [255, 0, 0] : 12,   # rider
                [255, 0, 0] : 12,   # rider
                [255, 255, 255] : 0,# road
                [70, 130, 180] : 10,# sky
                [152, 251, 152] : 9,# terrain
                [107, 142, 35] : 8, # vegatation
                [170, 170, 170] : 0,# road
                [153, 153, 153] : 5,# pole
                [0, 0, 142] : 5,    # pole
                [250, 170, 30] : 6, # traffic light
                [220, 220, 0] : 7,  # traffic sign (front only)
                [119, 11, 32] : 18, # bicycle
                [0, 60, 100] : 15,  # bus
                [0, 0, 142] : 13,   # car
                [0, 0, 230] : 17,   # motorcycle
                [0, 80, 100] : 16,  # train
                [0, 0, 70] : 14,    # truck
            }

            # flat, construction, object, nature, sky, human, vehicle
            super_color = {
                # [165, 42, 42] : ,
                # [0, 192, 0] : ,
                # [196, 196, 196] : # curb
                [190, 153, 153] : 1,# fence
                [180, 165, 180] : 1,
                [102, 102, 156] : 1,
                [102, 102, 156] : 1,# wall
                [128, 64, 255] : 0,
                [140, 140, 200] : 0,
                [170, 170, 170] : 0,
                [250, 170, 160] : 0,
                [96, 96, 96] : 0,
                [230, 150, 140] : 0,
                [128, 64, 128] : 0, # road
                [110, 110, 110] : 0,
                [244, 35, 232] : 0, # sidewalk
                [150, 100, 100] : 1,
                [70, 70, 70] : 1,   # building
                [150, 120, 90] : 1,
                [220, 20, 60] : 5, # person
                [255, 0, 0] : 5,   # rider
                [255, 0, 0] : 5,
                [255, 0, 0] : 5,
                [200, 128, 128] : 0,
                [255, 255, 255] : 0,  # road
                # [64, 170, 64],
                # [128, 64, 64],
                [70, 130, 180] : 4,# sky
                # [255, 255, 255],
                [152, 251, 152] : 0,# terrain
                [107, 142, 35] : 3, # vegatation
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
                [170, 170, 170] : 2,
                [210, 170, 100] : 2,
                [153, 153, 153] : 2,# pole
                [128, 128, 128] : 2,
                [0, 0, 142] : 2,    # pole
                [250, 170, 30] : 2, # traffic light
                [192, 192, 192] : 2,
                [220, 220, 0] : 2,  # traffic sign (front only)
                # [180, 165, 180],
                [119, 11, 32] : 6, # bicycle
                [0, 0, 142] : 6,
                [0, 60, 100] : 6,  # bus
                [0, 0, 142] : 6,   # car
                [0, 0, 90] : 6,
                [0, 0, 230] : 6,   # motorcycle
                [0, 80, 100] : 6,  # train
                [128, 64, 64] : 6,
                [0, 0, 110] : 6,
                [0, 0, 70] : 6,    # truck
                [0, 0, 192] : 6,
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
