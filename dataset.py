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

def count_img(root):
    cnt = 0
    for mode in ['test', 'train', 'val']:
        img_file = os.path.join(root,'leftImg8bit',mode)
        img_folder = os.listdir(img_file)

        for folder in img_folder:
            cur_folder = os.path.join(img_file, folder)
            cnt += len(os.listdir(cur_folder))
            # print(os.listdir(cur_folder))
        print(mode)
        print(cnt)

        cnt = 0

    return 0

def get_all_labels(root):
    labels = []
    final_labels = {}
    segmap_folder = os.listdir(root)
    
    print(root)
#     sub_path_train_img = os.path.join(root, img_folder, './train')
    sub_path_train_seg = os.path.join(root+'/training', 'v1.2', 'polygons')

#     sub_path_val_img = os.path.join(root, img_folder, './val')
    sub_path_val_seg = os.path.join(root+'/validation', 'v1.2', 'polygons')
    
    print(sub_path_train_seg)
    # extract training labels
    for file in os.listdir(sub_path_train_seg):
        # curr_seg_folder = os.path.join(sub_path_train_seg, folder)
        # for file in os.listdir(curr_seg_folder):
        if 'json' not in file:
            continue
        f = open(os.path.join(sub_path_train_seg, file), 'r')
        data = json.loads(f.read())
        for obj in data['objects']:
            if obj['label'] not in labels:
                labels.append(obj['label'])

    # extract validation labels
    for file in os.listdir(sub_path_val_seg):
        # curr_seg_folder = os.path.join(sub_path_val_seg, folder)
        # for file in os.listdir(curr_seg_folder):
        if 'json' not in file:
            continue
        f = open(os.path.join(sub_path_val_seg, file), 'r')
        data = json.loads(f.read())
        for obj in data['objects']:
            if obj['label'] not in labels:
                labels.append(obj['label'])
                    
    for i in range(len(labels)):
        final_labels[labels[i]] = i
    print(final_labels)
    return final_labels


def create_segmentation_maps(root, labels):
    # test, train, val = os.listdir(root)
    
    if not os.path.exists('./img'):
        os.makedirs('img')
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
    
    sub_path_train_img = os.path.join(root, 'training/images/')
    sub_path_train_seg = os.path.join(root, 'training/v2.0/polygons')

    sub_path_val_img = os.path.join(root, 'validation/images/')
    sub_path_val_seg = os.path.join(root, 'validation/v2.0/polygons')
    
    # for folder in os.listdir(sub_path_train_seg):
    #     curr_img_folder = os.path.join(sub_path_train_img, folder)
    #     curr_seg_folder = os.path.join(sub_path_train_seg, folder)
        
    for file in os.listdir(sub_path_train_seg):
        id = file.split('.')[0]
        print(file)
        if not os.path.exists('/data/datasets/mapillary/training/seg/'+id+'.png'):
            polygoni = open(os.path.join(sub_path_train_seg, file), 'r')
            
            # f = open(os.path.join(curr_seg_folder, id+'_gtFine_polygons.json'), 'r')
            data = json.loads(polygoni.read())
            seg_map = np.zeros((data['width'],data['height']))
            for obj in data['objects']:
                label = obj['label']
                poly = np.array(obj['polygon'])
                # print('segmap shape :',seg_map.shape)
                # print(poly.shape)
                if len(poly.shape) == 2:
                    # print('legal')
                    # print(poly[:,0])
                    # print(poly[:,1])
                    rr, cc = polygon(poly[:,0], poly[:,1], seg_map.shape)
                    seg_map[rr,cc] = labels[label]
            
            # print('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png')
            print('/data/datasets/mapillary/training/'+file)
            # cv2.imwrite('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png', img)
            cv2.imwrite('/data/datasets/mapillary/training/seg/'+id+'.png', seg_map.T)
            print('/data/datasets/mapillary/training/seg/'+id+'.png')

     
    for file in os.listdir(sub_path_val_seg):
        id = file.split('.')[0]
        print(file)
        if not os.path.exists('/data/datasets/mapillary/validation/seg/'+id+'.png'):
            polygoni = open(os.path.join(sub_path_val_seg, file), 'r')
            
            # f = open(os.path.join(curr_seg_folder, id+'_gtFine_polygons.json'), 'r')
            data = json.loads(polygoni.read())
            seg_map = np.zeros((data['width'],data['height']))
            for obj in data['objects']:
                label = obj['label']
                poly = np.array(obj['polygon'])
                # print('segmap shape :',seg_map.shape)
                # print(poly.shape)
                if len(poly.shape) == 2:
                    # print('legal')
                    # print(poly[:,0])
                    # print(poly[:,1])
                    rr, cc = polygon(poly[:,0], poly[:,1], seg_map.shape)
                    seg_map[rr,cc] = labels[label]
            
            # print('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png')
            print('/data/datasets/mapillary/validation/'+file)
            # cv2.imwrite('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png', img)
            cv2.imwrite('/data/datasets/mapillary/validation/seg/'+id+'.png', seg_map.T)
            print('/data/datasets/mapillary/validation/seg/'+id+'.png')


def get_dataset(dataset, batch, imsize, workers, super_class=False):
    if dataset == 'G':
        train_dataset = GTA5(list_path='./data_list/GTA5', split='train', crop_size=imsize, super_class=super_class)
        test_dataset = None

    elif dataset == 'C':
        train_dataset = Cityscapes(list_path='./data_list/Cityscapes', split='train', crop_size=imsize, super_class=super_class)
        test_dataset = Cityscapes(list_path='./data_list/Cityscapes', split='val', crop_size=imsize, train=False, super_class=super_class)

    elif dataset == 'GtoC':
        train_dataset = GTA5(list_path='./data_list/GtoC', split='train', crop_size=imsize)
        test_dataset = None

    elif dataset == 'I':
        train_dataset = IDD(list_path='./data_list/IDD', split='train', crop_size=imsize, super_class=super_class)
        test_dataset = IDD(list_path='./data_list/IDD', split='val', crop_size=imsize, train=False, super_class=super_class)
    
    elif dataset == 'M':
        train_dataset = Mapillary(list_path='./data_list/mapillary', split='train', crop_size=imsize)
        test_dataset = Mapillary(list_path='./data_list/mapillary', split='val', crop_size=imsize, train=False)

    # elif dataset == 'M':
    #     train_dataset = dset.MNIST(root='./data', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.Scale(imsize),
    #                                    transforms.Grayscale(3),
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                ]))
    #     test_dataset = dset.MNIST(root='./data', train=False, download=True,
    #                               transform=transforms.Compose([
    #                                   transforms.Scale(imsize),
    #                                   transforms.Grayscale(3),
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                               ]))
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
                                                   shuffle=False, num_workers=int(workers), pin_memory=True)
    return train_dataloader, test_dataloader

def sort_super_class(cls_list):
    sup_dict = [[], [],[],[],[],[],[],[],[]]
    ret_dict = dict()
    super_list = ['marking', 'marking-only', 'construction', 'object', 'nature', 'human', 'animal', 'void']
    for key, value in enumerate(cls_list):
        print(key, value)
        for i in range(8):
            if value.split('--')[1] == 'flat': 
                if value not in sup_dict[0]: sup_dict[0].append(value)
            elif super_list[i] == value.split('--')[0]:
                sup_dict[i+1].append(value)
    idx = 0
    print(sup_dict)
    for j in range(len(sup_dict)):
        for i in range(len(sup_dict[j])):
            ret_dict[sup_dict[j][i]] = idx
            idx += 1
    
    return ret_dict

def print_keys(cls_dict):
    cls_list = []
    for _, value in enumerate(cls_dict):
        if value.split('--')[0] not in cls_list:
            cls_list.append(value.split('--')[0])

    print(cls_list)
        
def return_keys(cls_dict):
    cls_list = []
    for _, value in enumerate(cls_dict):
        if value not in cls_list:
            cls_list.append(value)

    return(cls_list)

def make_idmatchingdict(d1, d2):
    ret_dict = dict()
    for i in d1.items():
        for j in d2.items():
            # print(key, value1, key2, value2)
            print(i, j)
            # print(d1[i], d2[j])
            if i[0] == j[0]:
                # print(key, ':', key2)
                ret_dict[i[1]] = j[1]
    
    print(ret_dict)
    return ret_dict

if __name__ == '__main__':
    # labels = get_all_labels('/data/datasets/mapillary')
    # labels ={'construction--flat--road': 0, 'construction--flat--sidewalk': 1, 'construction--flat--parking': 2, 'construction--flat--crosswalk-plain': 3, 'construction--flat--curb-cut': 4, 'construction--flat--traffic-island': 5, 'construction--flat--road-shoulder': 6, 'construction--flat--bike-lane': 7, 'construction--flat--driveway': 8, 'construction--flat--parking-aisle': 9, 'construction--flat--rail-track': 10, 'construction--flat--service-lane': 11, 'construction--flat--pedestrian-area': 12, 'marking--discrete--crosswalk-zebra': 13, 'marking--continuous--solid': 14, 'marking--discrete--other-marking': 15, 'marking--discrete--text': 16, 'marking--discrete--stop-line': 17, 'marking--discrete--arrow--straight': 18, 'marking--continuous--dashed': 19, 'marking--discrete--hatched--diagonal': 20, 'marking--discrete--ambiguous': 21, 'marking--discrete--hatched--chevron': 22, 'marking--discrete--arrow--right': 23, 'marking--discrete--arrow--split-right-or-straight': 24, 'marking--discrete--arrow--other': 25, 'marking--discrete--arrow--left': 26, 'marking--discrete--symbol--bicycle': 27, 'marking--continuous--zigzag': 28, 'marking--discrete--give-way-single': 29, 'marking--discrete--give-way-row': 30, 'marking--discrete--arrow--split-left-or-straight': 31, 'marking--discrete--symbol--other': 32, 'marking-only--continuous--dashed': 33, 'marking-only--discrete--other-marking': 34, 'marking-only--discrete--text': 35, 'marking-only--discrete--crosswalk-zebra': 36, 'construction--structure--building': 37, 'construction--barrier--wall': 38, 'construction--barrier--fence': 39, 'construction--structure--bridge': 40, 'construction--barrier--curb': 41, 'construction--barrier--temporary': 42, 'construction--barrier--separator': 43, 'construction--barrier--road-median': 44, 'construction--barrier--guard-rail': 45, 'construction--barrier--road-side': 46, 'construction--structure--tunnel': 47, 'construction--barrier--concrete-block': 48, 'construction--barrier--ambiguous': 49, 'construction--structure--garage': 50, 'construction--barrier--other-barrier': 51, 'object--support--pole': 52, 'object--vehicle--vehicle-group': 53, 'object--vehicle--car': 54, 'object--traffic-sign--back': 55, 'object--traffic-sign--front': 56, 'object--street-light': 57, 'object--support--utility-pole': 58, 'object--sign--store': 59, 'object--sign--advertisement': 60, 'object--traffic-sign--information-parking': 61, 'object--trash-can': 62, 'object--traffic-light--pedestrians': 63, 'object--sign--back': 64, 'object--vehicle--other-vehicle': 65, 'object--bike-rack': 66, 'object--manhole': 67, 'object--catch-basin': 68, 'object--water-valve': 69, 'object--traffic-light--general-upright': 70, 'object--traffic-sign--direction-front': 71, 'object--traffic-sign--ambiguous': 72, 'object--cctv-camera': 73, 'object--vehicle--bicycle': 74, 'object--vehicle--truck': 75, 'object--sign--other': 76, 'object--bench': 77, 'object--traffic-sign--direction-back': 78, 'object--sign--ambiguous': 79, 'object--vehicle--bus': 80, 'object--traffic-light--other': 81, 'object--vehicle--boat': 82, 'object--traffic-cone': 83, 'object--banner': 84, 'object--junction-box': 85, 'object--sign--information': 86, 'object--support--traffic-sign-frame': 87, 'object--support--pole-group': 88, 'object--traffic-light--cyclists': 89, 'object--vehicle--wheeled-slow': 90, 'object--vehicle--motorcycle': 91, 'object--traffic-light--general-single': 92, 'object--vehicle--trailer': 93, 'object--vehicle--on-rails': 94, 'object--traffic-light--general-horizontal': 95, 'object--fire-hydrant': 96, 'object--pothole': 97, 'object--traffic-sign--temporary-back': 98, 'object--parking-meter': 99, 'object--traffic-sign--temporary-front': 100, 'object--phone-booth': 101, 'object--mailbox': 102, 'object--vehicle--caravan': 103, 'nature--vegetation': 104, 'nature--sky': 105, 'nature--terrain': 106, 'nature--water': 107, 'nature--mountain': 108, 'nature--snow': 109, 'nature--sand': 110, 'human--person--individual': 111, 'human--rider--motorcyclist': 112, 'human--person--person-group': 113, 'human--rider--other-rider': 114, 'human--rider--bicyclist': 115, 'animal--bird': 116, 'animal--ground-animal': 117, 'void--ground': 118, 'void--static': 119, 'void--car-mount': 120, 'void--dynamic': 121, 'void--ego-vehicle': 122}
    # create_segmentation_maps('/data/datasets/mapillary', labels)
    count_img('/data/datasets/IDD/')