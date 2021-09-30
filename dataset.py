import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

########################################
import os 
import cv2
import sys
import json

import numpy as np
import skimage.io as io

from skimage.draw import polygon
from skimage import img_as_float
########################################

## Import Data Loaders ##
from dataloader import *

def get_all_labels(root):
    labels = []
    final_labels = {}
    
    img_folder, segmap_folder = os.listdir(root)
    
#     sub_path_train_img = os.path.join(root, img_folder, './train')
    sub_path_train_seg = os.path.join(root, segmap_folder, './train')

#     sub_path_val_img = os.path.join(root, img_folder, './val')
    sub_path_val_seg = os.path.join(root, segmap_folder, './val')
    
    # extract training labels
    for folder in os.listdir(sub_path_train_seg):
#         curr_img_folder = os.path.join(sub_path_train_img, folder)
        curr_seg_folder = os.path.join(sub_path_train_seg, folder)
        for file in os.listdir(curr_seg_folder):
            if 'json' not in file:
                continue
            f = open(os.path.join(curr_seg_folder, file), 'r')
            data = json.loads(f.read())
            for obj in data['objects']:
                if obj['label'] not in labels:
                    labels.append(obj['label'])

    # extract validation labels
    for folder in os.listdir(sub_path_val_seg):
#         curr_img_folder = os.path.join(sub_path_val_img, folder)
        curr_seg_folder = os.path.join(sub_path_val_seg, folder)
        for file in os.listdir(curr_seg_folder):
            if 'json' not in file:
                continue
            f = open(os.path.join(curr_seg_folder, file), 'r')
            data = json.loads(f.read())
            for obj in data['objects']:
                if obj['label'] not in labels:
                    labels.append(obj['label'])
                    
    for i in range(len(labels)):
        final_labels[labels[i]] = i
    return final_labels


def create_segmentation_maps(root, labels):
    img_folder, segmap_folder = os.listdir(root)
    
    # if not os.path.exists('./img'):
    #     os.makedirs('img')
    # if not os.path.exists('./img/train'):
    #     os.makedirs('img/train')
    # if not os.path.exists('./img/val'):
    #     os.makedirs('img/val')

    # if not os.path.exists('./seg'):
    #     os.makedirs('seg')
    # if not os.path.exists('./seg/train'):
    #     os.makedirs('seg/train')
    # if not os.path.exists('./seg/val'):
    #     os.makedirs('seg/val')
    
    sub_path_train_img = os.path.join(root, img_folder, './train')
    sub_path_train_seg = os.path.join(root, segmap_folder, './train')

    sub_path_val_img = os.path.join(root, img_folder, './val')
    sub_path_val_seg = os.path.join(root, segmap_folder, './val')
    
    for folder in os.listdir(sub_path_train_seg):
        curr_img_folder = os.path.join(sub_path_train_img, folder)
        curr_seg_folder = os.path.join(sub_path_train_seg, folder)
        
        for file in os.listdir(curr_seg_folder):
            id = file.split('_')[0]
            # print(id)
            if '.' not in id and not os.path.exists('/data/datasets/IDD/gtFine/train/'+folder+'/'+id+'.png'):
                print('/data/datasets/IDD/gtFine/train/'+folder+'/'+id+'.png')
                # print('its new?')
                img = cv2.imread(os.path.join(curr_img_folder, file))
                
                f = open(os.path.join(curr_seg_folder, id+'_gtFine_polygons.json'), 'r')
                data = json.loads(f.read())
                seg_map = np.zeros((data['imgWidth'],data['imgHeight']))
                for obj in data['objects']:
                    label = obj['label']
                    poly = np.array(obj['polygon'])
                    # print('segmap shape :',seg_map.shape)
                    # print(poly.shape)
                    if len(poly.shape) == 2:
                        # print('legal')
                        rr, cc = polygon(poly[:,0], poly[:,1], seg_map.shape)
                        seg_map[rr,cc] = labels[label]
                
                # print('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png')
                print('/data/datasets/IDD/gtFine/train/'+folder+'/'+id+'.png')
                # cv2.imwrite('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png', img)
                cv2.imwrite('/data/datasets/IDD/gtFine/train/'+folder+'/'+id+'.png', seg_map.T)
            
    for folder in os.listdir(sub_path_val_seg):
        curr_img_folder = os.path.join(sub_path_val_img, folder)
        curr_seg_folder = os.path.join(sub_path_val_seg, folder)
        
        for file in os.listdir(curr_seg_folder):
            id = file.split('_')[0]
            if not os.path.exists('/data/datasets/IDD/gtFine/val/'+folder+'/'+id+'.png'):
                img = cv2.imread(os.path.join(curr_img_folder, file))
                
                f = open(os.path.join(curr_seg_folder, id+'_gtFine_polygons.json'), 'r')
                data = json.loads(f.read())
                seg_map = np.zeros((data['imgWidth'],data['imgHeight']), np.uint8)
                for obj in data['objects']:
                    label = obj['label']
                    poly = np.array(obj['polygon'])
                    rr, cc = polygon(poly[:,0], poly[:,1], seg_map.shape)
                    seg_map[rr,cc] = labels[label]
                    
                # print('/data/datasets/IDD/leftImg8bit/val/'+folder+'/'+id+'.png')
                print('/data/datasets/IDD/gtFine/val/'+folder+'/'+id+'.png')
                # cv2.imwrite('/data/datasets/IDD/leftImg8bit/val/'+folder+'/'+id+'.png', img)
                cv2.imwrite('/data/datasets/IDD/gtFine/val/'+folder+'/'+id+'.png', seg_map.T)
            print('exist file')


def get_dataset(dataset, batch, imsize, workers):
    if dataset == 'G':
        train_dataset = GTA5(list_path='./data_list/GTA5', split='train', crop_size=imsize)
        test_dataset = None

    elif dataset == 'C':
        train_dataset = Cityscapes(list_path='./data_list/Cityscapes', split='train', crop_size=imsize)
        test_dataset = Cityscapes(list_path='./data_list/Cityscapes', split='val', crop_size=imsize, train=False)

    elif dataset == 'GtoC':
        train_dataset = GTA5(list_path='./data_list/GtoC', split='train', crop_size=imsize)
        test_dataset = None

    # elif dataset == 'I':
        # train_dataset = 

    elif dataset == 'M':
        train_dataset = dset.MNIST(root='./data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.Scale(imsize),
                                       transforms.Grayscale(3),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        test_dataset = dset.MNIST(root='./data', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Scale(imsize),
                                      transforms.Grayscale(3),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
    elif dataset == 'U':
        train_dataset = dset.USPS(root='./data/usps', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.Scale(imsize),
                                       transforms.Grayscale(3),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        test_dataset = dset.USPS(root='./data/usps', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Scale(imsize),
                                      transforms.Grayscale(3),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
    elif dataset == 'MM':
        train_dataset = MNIST_M(root='./data/mnist_m', train=True,
                                transform=transforms.Compose([
                                    transforms.Scale(imsize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        test_dataset = MNIST_M(root='./data/mnist_m', train=False,
                               transform=transforms.Compose([
                                   transforms.Scale(imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    elif dataset == 'S':
        # train_dataset = dset.SVHN(root='./data/svhn', split='train', download=True,
        #                           transform=transforms.Compose([
        #                               transforms.Scale(imsize),
        #                               transforms.ToTensor(),
        #                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                           ]))
        # test_dataset = dset.SVHN(root='./data/svhn', split='test', download=True,
        #                          transform=transforms.Compose([
        #                              transforms.Scale(imsize),
        #                              transforms.ToTensor(),
        #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                          ]))
        # train_dataset = SyntheticDigits(root='./data/synthetic_digits', train=True,
        #                         transform=transforms.Compose([
        #                             transforms.Scale(imsize),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                         ]))
        # test_dataset = SyntheticDigits(root='./data/synthetic_digits', train=False,
        #                        transform=transforms.Compose([
        #                            transforms.Scale(imsize),
        #                            transforms.ToTensor(),
        #                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                        ]))
        train_dataset = SEMEION(root='./data/semeion',
                                transform=transforms.Compose([
                                    transforms.Scale(imsize),
                                    transforms.Grayscale(3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        test_dataset = None



    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch,
                                                   shuffle=True, num_workers=int(workers), pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch*4,
                                                   shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    # labels = get_all_labels('/data/datasets/IDD')
    labels = {'sky': 0, 'road': 1, 'drivable fallback': 2, 'sidewalk': 3, 'building': 4, 'wall': 5, 'vegetation': 6, 'fence': 7, 'obs-str-bar-fallback': 8, 'curb': 9, 'pole': 10, 'rider': 11, 'truck': 12, 'autorickshaw': 13, 'billboard': 14, 'motorcycle': 15, 'person': 16, 'bus': 17, 'car': 18, 'bicycle': 19, 'vehicle fallback': 20, 'non-drivable fallback': 21, 'fallback background': 22, 'traffic sign': 23, 'animal': 24, 'traffic light': 25, 'polegroup': 26, 'bridge': 27, 'caravan': 28, 'parking': 29, 'trailer': 30, 'guard rail': 31, 'rectification border': 32, 'out of roi': 33, 'train': 34, 'rail track': 35, 'tunnel': 36, 'license plate': 37, 'ground': 38, 'ego vehicle': 39, 'unlabeled': 40}
    create_segmentation_maps('/data/datasets/IDD', labels)