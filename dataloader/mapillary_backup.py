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
        self.id_to_trainid = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: ignore_label,
                              6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                              13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18}

        self.id_to_trainid = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 7: 0, 9: 0, 10: 0, 11: 0, 12: 0, 37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 45: 1, 47: 1, 51: 1, 52: 2, 54: 7, 55: 2, 56: 2, 57: 2, 58: 2, 65: 7, 70: 2, 74: 7, 75: 7, 80: 7, 82: 7, 87: 2, 90: 7, 91: 7, 92: 2, 93: 7, 94: 7, 95: 2, 97: 2, 103: 7, 104: 3, 105: 4, 106: 0, 111: 6, 112: 6, 113: 6, 114: 6, 115: 6}



        {'construction--flat--road': 0, 
        'construction--flat--sidewalk': 0, 
        'construction--flat--parking': 0, 
        'construction--flat--crosswalk-plain': 0, 
        'construction--flat--curb-cut': 0, 
        'construction--flat--traffic-island': ignore_label, 
        'construction--flat--road-shoulder': ignore_label, 
        'construction--flat--bike-lane': 0, 
        'construction--flat--driveway': ignore_label, 
        'construction--flat--parking-aisle': 0, 
        'construction--flat--rail-track': 0, 
        'construction--flat--service-lane': 0, 
        'construction--flat--pedestrian-area': 0, 
        # 'marking--discrete--crosswalk-zebra': ignore_label, 
        # 'marking--continuous--solid': ignore_label, 
        # 'marking--discrete--other-marking': ignore_label, 
        # 'marking--discrete--text': ignore_label, 
        # 'marking--discrete--stop-line': ignore_label, 
        # 'marking--discrete--arrow--straight': 18, 
        # 'marking--continuous--dashed': 19, 
        # 'marking--discrete--hatched--diagonal': 20, 
        # 'marking--discrete--ambiguous': 21, 
        # 'marking--discrete--hatched--chevron': 22, 
        # 'marking--discrete--arrow--right': 23, 
        # 'marking--discrete--arrow--split-right-or-straight': 24, 
        # 'marking--discrete--arrow--other': 25, 
        # 'marking--discrete--arrow--left': 26, 
        # 'marking--discrete--symbol--bicycle': 27, 
        # 'marking--continuous--zigzag': 28, 
        # 'marking--discrete--give-way-single': 29, 
        # 'marking--discrete--give-way-row': 30, 
        # 'marking--discrete--arrow--split-left-or-straight': 31, 
        # 'marking--discrete--symbol--other': 32, 
        # 'marking-only--continuous--dashed': 33, 
        # 'marking-only--discrete--other-marking': 34, 
        # 'marking-only--discrete--text': 35, 
        # 'marking-only--discrete--crosswalk-zebra': 36, 
        'construction--structure--building': 1, 
        'construction--barrier--wall': 1, 
        'construction--barrier--fence': 1, 
        'construction--structure--bridge': 1, 
        'construction--barrier--curb': 1, 
        'construction--barrier--temporary': ignore_label, 
        'construction--barrier--separator': ignore_label, 
        'construction--barrier--road-median': ignore_label, 
        'construction--barrier--guard-rail': 1, 
        'construction--barrier--road-side': ignore_label, 
        'construction--structure--tunnel': 1, 
        'construction--barrier--concrete-block': ignore_label, 
        'construction--barrier--ambiguous': ignore_label, 
        'construction--structure--garage': ignore_label, 
        'construction--barrier--other-barrier': 1, 
        'object--support--pole': 2, 
        # 'object--vehicle--vehicle-group': ignore_label, 
        'object--vehicle--car': 7, 
        'object--traffic-sign--back': 2, 
        'object--traffic-sign--front': 2, 
        'object--street-light': 2, 
        'object--support--utility-pole': 2, 
        # 'object--sign--store': ignore_label, 
        # 'object--sign--advertisement': ignore_label, 
        # 'object--traffic-sign--information-parking': ignore_label, 
        # 'object--trash-can': ignore_label, 
        # 'object--traffic-light--pedestrians': ignore_label, 
        # 'object--sign--back': ignore_label, 
        'object--vehicle--other-vehicle': 7, 
        # 'object--bike-rack': ignore_label, 
        # 'object--manhole': ignore_label, 
        # 'object--catch-basin': ignore_label, 
        # 'object--water-valve': ignore_label, 
        'object--traffic-light--general-upright': 2, #  ???
        # 'object--traffic-sign--direction-front': ignore_label, 
        # 'object--traffic-sign--ambiguous': ignore_label, 
        # 'object--cctv-camera': 73, 
        'object--vehicle--bicycle': 7, 
        'object--vehicle--truck': 7, 
        # 'object--sign--other': 76, 
        # 'object--bench': 77, 
        # 'object--traffic-sign--direction-back': 78, 
        # 'object--sign--ambiguous': 79, 
        'object--vehicle--bus': 7, 
        # 'object--traffic-light--other': 81, 
        'object--vehicle--boat': 7, 
        # 'object--traffic-cone': ignore_label, 
        # 'object--banner': ignore_label, 
        # 'object--junction-box': ignore_label, 
        # 'object--sign--information': ignore_label, 
        'object--support--traffic-sign-frame': 2, 
        # 'object--support--pole-group': 88, 
        # 'object--traffic-light--cyclists': 89, 
        'object--vehicle--wheeled-slow': 7, 
        'object--vehicle--motorcycle': 7, 
        'object--traffic-light--general-single': 2, #   ???
        'object--vehicle--trailer': 7, 
        'object--vehicle--on-rails': 7, 
        'object--traffic-light--general-horizontal': 2, 
        # 'object--fire-hydrant': 96, 
        'object--pothole': 2, 
        # 'object--traffic-sign--temporary-back': 98, 
        # 'object--parking-meter': 99, 
        # 'object--traffic-sign--temporary-front': 100, 
        # 'object--phone-booth': ignore_label, 
        # 'object--mailbox': ignore_label, 
        'object--vehicle--caravan': 7, 
        'nature--vegetation': 3, 
        'nature--sky': 4, 
        'nature--terrain': 0, 
        # 'nature--water': 107, 
        # 'nature--mountain': 108, 
        # 'nature--snow': 109, 
        # 'nature--sand': 110, 
        'human--person--individual': 6, 
        'human--rider--motorcyclist': 6, 
        'human--person--person-group': 6, 
        'human--rider--other-rider': 6, 
        'human--rider--bicyclist': 6, 
        }

        # 'animal--bird': 116, 
        # 'animal--ground-animal': 117, 

        # 'void--ground': 118, 
        # 'void--static': 119, 
        # 'void--car-mount': 120, 
        # 'void--dynamic': 121, 
        # 'void--ego-vehicle': 122}

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
