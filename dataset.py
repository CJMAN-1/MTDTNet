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
    
    sub_path_train_img = os.path.join(root, img_folder, 'train')
    sub_path_train_seg = os.path.join(root, segmap_folder, 'train')

    sub_path_val_img = os.path.join(root, img_folder, 'val')
    sub_path_val_seg = os.path.join(root, segmap_folder, 'val')
    
    for folder in os.listdir(sub_path_train_seg):
        curr_img_folder = os.path.join(sub_path_train_img, folder)
        curr_seg_folder = os.path.join(sub_path_train_seg, folder)
        
        for file in os.listdir(curr_seg_folder):
            id = file.split('_')[0]
            # print(id)
            if '.' not in id and not os.path.exists('/data/datasets/IDD/gtFine/train/'+folder+'/'+id+'.png'):
                # print('/data/datasets/IDD/gtFine/train/'+folder+'/'+id+'.png')
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
        print(curr_seg_folder)
        
        for file in os.listdir(curr_seg_folder):
            # print(curr_seg_folder)
            id = file.split('_')[0]
            if '.' not in id and not os.path.exists('/data/datasets/IDD/gtFine/val/'+folder+'/'+id+'.png'):
                img = cv2.imread(os.path.join(curr_img_folder, file))
                # print(curr_seg_folder+'/'+id+'_gtFine_polygons.json')
                
                f = open(os.path.join(curr_seg_folder, id+'_gtFine_polygons.json'), 'r')
                data = json.loads(f.read())
                seg_map = np.zeros((data['imgWidth'],data['imgHeight']), np.uint8)
                for obj in data['objects']:
                    label = obj['label']
                    poly = np.array(obj['polygon'])
                    if len(poly.shape) == 2:
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

    elif dataset == 'I':
        train_dataset = IDD(list_path='./data_list/IDD', split='train', crop_size=imsize)
        test_dataset = IDD(list_path='./data_list/IDD', split='val', crop_size=imsize, train=False)
    
    # elif dataset == 'MP':
    #     train_dataset = mapillary(list_path='.data_list/mapillary', split='train', crop_size=imsize)
    #     test_dataset = mapillary(list_path='.data_list/mapillary', split='val', crop_size=imsize, train=False)

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
                                                   shuffle=False, num_workers=int(workers), pin_memory=True)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    labels = get_all_labels('/data/datasets/mapillary')
    # labels = {'sky': 0, 'road': 1, 'drivable fallback': 2, 'sidewalk': 3, 'building': 4, 'wall': 5, 'vegetation': 6, 'fence': 7, 'obs-str-bar-fallback': 8, 'curb': 9, 'pole': 10, 'rider': 11, 'truck': 12, 'autorickshaw': 13, 'billboard': 14, 'motorcycle': 15, 'person': 16, 'bus': 17, 'car': 18, 'bicycle': 19, 'vehicle fallback': 20, 'non-drivable fallback': 21, 'fallback background': 22, 'traffic sign': 23, 'animal': 24, 'traffic light': 25, 'polegroup': 26, 'bridge': 27, 'caravan': 28, 'parking': 29, 'trailer': 30, 'guard rail': 31, 'rectification border': 32, 'out of roi': 33, 'train': 34, 'rail track': 35, 'tunnel': 36, 'license plate': 37, 'ground': 38, 'ego vehicle': 39, 'unlabeled': 40}
    {'construction--flat--road': 0 ,'construction--flat--sidewalk': 1, 'construction--structure--building': 2,
    'construction--barrier--wall': 3, 'construction--barrier--fence': 4, 
    'object--support--pole': 5, 'void--ground': 6, 
    'object--vehicle--vehicle-group': 10, 'nature--vegetation': 8, 
    'marking--discrete--crosswalk-zebra': 9, 'nature--sky': 10,  
    'void--static': 11, 'object--vehicle--car': 12, 'human--person--individual': 13, 
    'object--traffic-sign--back': 14, 'object--traffic-sign--front': 15, 
    'object--street-light': 16, 'object--support--utility-pole': 17, 
    'marking--continuous--solid': 18, 'object--sign--store': 19, 
    'object--sign--advertisement': 20, 'animal--bird': 21, 
    'object--traffic-sign--information-parking': 22, 
    'construction--flat--parking': 23, 'nature--terrain': 24, 
    'marking--discrete--other-marking': 25, 'object--trash-can': 26, 
    'marking--discrete--text': 27, 'void--car-mount': 28, 
    'object--traffic-light--pedestrians': 29, 'object--sign--back': 30, 
    'construction--structure--bridge': 31, 'object--vehicle--other-vehicle': 32, 
    'construction--flat--crosswalk-plain': 33, 'construction--flat--curb-cut': 34, 
    'object--bike-rack': 35, 'animal--ground-animal': 36, 'void--dynamic': 37, 
    'object--manhole': 38, 'object--catch-basin': 39, 'object--water-valve': 40, 
    'object--traffic-light--general-upright': 41, 
    'object--traffic-sign--direction-front': 42, 'marking--discrete--stop-line': 43, 
    'marking--discrete--arrow--straight': 44, 'object--traffic-sign--ambiguous': 45, 
    'object--cctv-camera': 46, 'construction--barrier--curb': 47, 
    'human--rider--motorcyclist': 48, 'marking--continuous--dashed': 49, 
    'object--vehicle--bicycle': 50, 'object--vehicle--truck': 51, 
    'object--sign--other': 52, 'object--bench': 53, 
    'object--traffic-sign--direction-back': 54, 'construction--barrier--temporary': 55, 
    'object--sign--ambiguous': 56, 'construction--flat--traffic-island': 57, 
    'object--vehicle--bus': 58, 'object--traffic-light--other': 59, 
    'marking--discrete--hatched--diagonal': 60, 'nature--water': 61, 
    'object--vehicle--boat': 62, 'object--traffic-cone': 63, 'object--banner': 64, 
    'construction--flat--road-shoulder': 65, 'object--junction-box': 66, 
    'construction--flat--bike-lane': 67, 'marking--discrete--ambiguous': 68, 
    'object--sign--information': 69, 'nature--mountain': 70, 'void--ego-vehicle': 71, 
    'construction--barrier--separator': 72, 'construction--barrier--road-median': 73, 
    'object--support--traffic-sign-frame': 74, 'construction--barrier--guard-rail': 75, 
    'construction--barrier--road-side': 76, 'marking--discrete--hatched--chevron': 77, 
    'construction--flat--driveway': 78, 'construction--flat--parking-aisle': 79, 
    'object--support--pole-group': 80, 'object--traffic-light--cyclists': 81, 
    'object--vehicle--wheeled-slow': 82, 'object--vehicle--motorcycle': 83, 
    'object--traffic-light--general-single': 84, 'object--vehicle--trailer': 85, 
    'construction--flat--rail-track': 86, 'object--vehicle--on-rails': 87, 
    'marking--discrete--arrow--right': 88, 'construction--structure--tunnel': 89, 
    'construction--barrier--concrete-block': 90, 
    'object--traffic-light--general-horizontal': 91, 'object--fire-hydrant': 92, 
    'object--pothole': 93, 'human--person--person-group': 94, 'nature--snow': 95, 
    'marking--discrete--arrow--split-right-or-straight': 96, 
    'construction--barrier--ambiguous': 97, 'object--traffic-sign--temporary-back': 98, 
    'nature--sand': 99, 'construction--flat--service-lane': 100, 
    'construction--flat--pedestrian-area': 101, 'object--parking-meter': 102, 'human--rider--other-rider': 103, 'marking--discrete--arrow--other': 104, 'marking--discrete--arrow--left': 105, 'object--traffic-sign--temporary-front': 106, 'marking--discrete--symbol--bicycle': 107, 'human--rider--bicyclist': 108, 'marking--continuous--zigzag': 109, 'object--phone-booth': 110, 'marking--discrete--give-way-single': 111, 'object--mailbox': 112, 'construction--structure--garage': 113, 'marking--discrete--give-way-row': 114, 'construction--barrier--other-barrier': 115, 'marking--discrete--arrow--split-left-or-straight': 116, 'marking--discrete--symbol--other': 117, 'object--vehicle--caravan': 118, 'marking-only--continuous--dashed': 119, 'marking-only--discrete--other-marking': 120, 'marking-only--discrete--text': 121, 'marking-only--discrete--crosswalk-zebra': 122}

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
    # create_segmentation_maps('/data/datasets/IDD', labels)
