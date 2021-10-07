# -*- coding: utf-8 -*-
import copy
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
NUM_CLASSES = 19

# colour map
label_colours = [
    # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]]  # the color of ignored label(-1)
label_colours = list(map(tuple, label_colours))


class IDD(data.Dataset):
    def __init__(self,
                 list_path='./data_list/IDD',
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
            raise Warning("split must be train/val")

        self.images = [id.strip() for id in open(image_list_filepath)]
        self.labels = [id.strip() for id in open(label_list_filepath)]


        ignore_label = -1
        self.id_to_trainid = {0: 10,
                              1: 0, #road
                              2: ignore_label,
                              3: 1, #sidewalk
                              4: 2, 
                              5: 3, 
                              6: 8,    #vegatation?
                              7: 4,    #fence 
                              8: ignore_label, 
                              9: ignore_label, 
                              10: 5, 
                              11: 12, 
                              12: 14, 
                              13: ignore_label, 
                              14: ignore_label, 
                              15: 17, 
                              16: 11, 
                              17: 15,
                              18: 13,
                              19: 18, 
                              20: ignore_label, 
                              21: 9, 
                              22: ignore_label, 
                              23: 7, 
                              24: ignore_label, 
                              25: 6, 
                              26: ignore_label, 
                              27: ignore_label, 
                              28: ignore_label, 
                              29: ignore_label, 
                              30: ignore_label, 
                              31: ignore_label, 
                              32: ignore_label, 
                              33: ignore_label, 
                              34: 16, 
                              35: ignore_label, 
                              36: ignore_label, 
                              37: ignore_label, 
                              38: ignore_label, 
                              39: ignore_label}

        # self.id_to_trainid = {0: 0, #sky?
        #                       1: ignore_label, 
        #                       2: ignore_label,
        #                       3: 1, 
        #                       4: ignore_label, 
        #                       5: ignore_label, 
        #                       6: 11,
        #                       7: ignore_label, 
        #                       8: 12, 
        #                       9: 17, 
        #                       10: 18, 
        #                       11: ignore_label, 
        #                       12: 13, 
        #                       13: 14, 
        #                       14: 15, 
        #                       15: ignore_label, 
        #                       16: ignore_label, 
        #                       17: 16,
        #                       18: ignore_label,
        #                       19: ignore_label, 
        #                       20: 3, 
        #                       21: 4, 
        #                       22: ignore_label, 
        #                       23: ignore_label, 
        #                       24: 7, 
        #                       25: 6, 
        #                       26: 5, 
        #                       27: ignore_label, 
        #                       28: ignore_label, 
        #                       29: 2, 
        #                       30: ignore_label, 
        #                       31: ignore_label, 
        #                       32: 8, 
        #                       33: 10, 
        #                       34: ignore_label, 
        #                       35: ignore_label, 
        #                       36: ignore_label, 
        #                       37: ignore_label, 
        #                       38: ignore_label, 
        #                       39: ignore_label}


        print("{} num images in IDD {} set have been loaded.".format(len(self.images), self.split))
        if self.numpy_transform:
            print("use numpy_transform, instead of tensor transform!")

    def id2trainId(self, label, reverse=False, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path).convert("RGB")

        gt_image_path = self.labels[item]
        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or self.split == "val") and self.train:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image

    def _train_sync_transform(self, img, mask):
        '''
        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''

        if self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            if mask: mask = mask.resize(self.crop_size, Image.NEAREST)

        # final transform
        if mask:
            img, mask = self._img_transform(img), self._mask_transform(mask)
            return img, mask
        else:
            img = self._img_transform(img)
            return img

    def _val_sync_transform(self, img, mask):
        if self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            mask = mask.resize(self.crop_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, image):
        if self.numpy_transform:
            image = np.asarray(image, np.float32)
            image = image[:, :, ::-1]  # change to BGR
            image -= IMG_MEAN
            image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
            new_image = torch.from_numpy(image)
        else:
            image_transforms = ttransforms.Compose([
                ttransforms.ToTensor(),
                # ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
                ttransforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
            new_image = image_transforms(image)
        return new_image

    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target

    def __len__(self):
        return len(self.images)


def decode_labels(mask, num_images=1, num_classes=NUM_CLASSES):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[int(k_), int(j_)] = label_colours[int(k)]
        outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)


name_classes = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'trafflight',
    'traffsign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'unlabeled'
]


def inspect_decode_labels(pred, num_images=1, num_classes=NUM_CLASSES,
                          inspect_split=[0.9, 0.8, 0.7, 0.5, 0.0], inspect_ratio=[1.0, 0.8, 0.6, 0.3]):
    """Decode batch of segmentation masks accroding to the prediction probability.

    Args:
      pred: result of inference.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
      inspect_split: probability between different split has different brightness.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.data.cpu().numpy()
    n, c, h, w = pred.shape
    pred = pred.transpose([0, 2, 3, 1])
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for j_, j in enumerate(pred[i, :, :, :]):
            for k_, k in enumerate(j):
                assert k.shape[0] == num_classes
                k_value = np.max(np.softmax(k))
                k_class = np.argmax(k)
                for it, iv in enumerate(inspect_split):
                    if k_value > iv: break
                if iv > 0:
                    pixels[k_, j_] = tuple(map(lambda x: int(inspect_ratio[it] * x), label_colours[k_class]))
        outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)