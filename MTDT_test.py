from __future__ import print_function
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.backends.cudnn
import numpy as np
import torch.nn.functional as F

from Networks import *
from deeplabv2 import Deeplab

from utils import *
from losses import *

from dataloader.Cityscapes import decode_labels
from dataset import get_dataset
import torchvision.transforms as transforms
from prettytable import PrettyTable


def set_converts(source, targets):
    converts = list()
    task_converts = list()

    for target in targets:
        converts.append(source + '2' + target)
        # converts.append(target + '2' + source)
        task_converts.append(source + '2' + target)

    return converts, task_converts

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.imsize = opt.imsize
        self.best_miou = dict()
        self.min_miou = dict()

        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.opt.datasets:
            train_loader, test_loader = get_dataset(dataset=dset, batch=self.opt.batch,
                                                    imsize=self.imsize, workers=self.opt.workers, super_class=opt.super_class)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader

        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.writer = SummaryWriter('./tensorboard1/%s' % opt.ex)
        self.logger = getLogger()
        self.step = 0
        self.checkpoint = './checkpoint/%s' % opt.ex
        self.datasets = opt.datasets
        self.source = opt.datasets[0]
        self.targets = opt.datasets[1:]
        self.super_class = opt.super_class
        
        if opt.super_class:
            self.n_class = 7
            self.current_best_T = 'pretrained_model/deeplab_city_sc_63.10'
            for target in self.targets:
                if target == 'C':
                    self.best_miou[target] = 63.1
                    self.min_miou[target] = 59.0
                elif target == 'I':
                    self.best_miou[target] = 59.8
                    self.min_miou[target] = 58.0
                elif target == 'M':
                    self.best_miou[target] = 68.4
                    self.min_miou[target] = 59.0
        else:
            self.n_class = 19
            self.current_best_T = 'pretrained_model/deeplab_gta5_36.94'
            for target in self.targets:
                if target == 'C':
                    self.best_miou[target] = 36.9
                    self.min_miou[target] = 33.0
                elif target == 'I':
                    self.best_miou[target] = 39.7
                    self.min_miou[target] = 34.0
                elif target == 'M':
                    self.best_miou[target] = 46.3
                    self.min_miou[target] = 38.0
            
        self.converts, self.task_converts = set_converts(self.source, self.targets)
        self.w, self.h = opt.imsize
        self.loss_fns = Losses(opt)
        self.perceptual = dict()
        self.seg = dict()
        self.last_feature = dict()

        if opt.super_class:
            self.name_classes = [
                "flat", "construction", "object", "nature","sky", "human", "vehicle",
            ]
        else:
            self.name_classes = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "trafflight", "traffsign", "vegetation",
            "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        ]

    def set_default(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

        ## Random Seed ##
        print("Random Seed: ", self.opt.manualSeed)
        seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        if self.opt.cuda:
            torch.cuda.manual_seed_all(self.opt.manualSeed)

        ## Logger ##
        file_log_handler = FileHandler(self.opt.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)


    def set_networks(self):
        self.nets['E'] = Encoder()
        self.nets['G'] = Generator()
        self.nets['SE'] = Style_Encoder()
        self.nets['DT'] = Domain_Transfer(self.targets)
        self.nets['LE'] = Label_Embed()

        step = 4000
        # folder = './checkpoint/MTDT_save/%d_36.90_39.70_46.30/' % step
        folder = './checkpoint/M2CI/%d/' % step
        for key in self.nets.keys():
            self.nets[key] = (torch.load(folder + 'net%s.pth' % (key)))

        if self.opt.cuda:
            for net in self.nets.keys():
                self.nets[net].cuda()

    def set_train(self):
        for net in self.nets.keys():
            self.nets[net].train()

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        dset = self.source
        try:
            batch_data[dset] = batch_data_iter[dset].next()
        except StopIteration:
            batch_data_iter[dset] = iter(self.train_loader[dset])
            batch_data[dset] = batch_data_iter[dset].next()
        return batch_data

    def tensor_board_log(self, imgs, labels):
        nrow = 2
        direct_recon, indirect_recon, contents, styles, D_outputs_fake = dict(), dict(), dict(), dict(), dict()
        converted_imgs, converted_contents, converted_styles = dict(), dict(), dict()
        gamma, beta = dict(), dict()
        features, converted_features = dict(), dict()
        dset = self.source

        features[dset] = self.nets['E'](imgs[dset])
        
        gamma[self.source], beta[self.source] = self.nets['SE'](imgs[self.source])
            
        for convert in self.converts:
            source, target = convert.split('2')
            gamma[convert], beta[convert] = self.nets['DT'](gamma[source], beta[source], target=target)
            converted_imgs[convert] = self.nets['G'](gamma[convert]*self.nets['LE'](labels[source]) + beta[convert])
               
        # Input Images & Recon Images
        
        x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=False, nrow=nrow)
        self.writer.add_image('1_Input_Images/1_%s' % dset, x, self.step)

        # Converted Images
        for convert in converted_imgs.keys():
            x = vutils.make_grid(converted_imgs[convert].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('3_Converted_Images/%s' % convert, x, self.step)
            # x = vutils.make_grid(cycle_recon[convert].detach(), normalize=True, scale_each=False, nrow=nrow)
            # self.writer.add_image('2_Recon_Images/3_cycle_%s' % convert, x, self.step)

        x = vutils.make_grid(imgs[self.opt.datasets[0]].detach(), normalize=True, scale_each=False, nrow=nrow)
        self.writer.add_image('1_Input_Images/1_%s' % self.opt.datasets[0], x, self.step)

    def test(self):
        self.set_default()
        self.set_networks()
        self.set_train()
        batch_data_iter = dict()
        for dset in self.opt.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        for i in range(self.opt.iter):
            self.step += 1
            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            dset = self.source
            imgs[dset], labels[dset] = batch_data[dset]
            if self.opt.cuda:
                imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
            labels[dset] = labels[dset].long()
                        
            # tensorboard
            with torch.no_grad():
                self.tensor_board_log(imgs, labels)
            print(i)
 

if __name__ == '__main__':
    from param import get_params
    opt=get_params()
    trainer = Trainer(opt)
    trainer.test()