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


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.imsize = opt.imsize
        self.best_miou = dict()

        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.opt.datasets:
            train_loader, test_loader = get_dataset(dataset=dset, batch=self.opt.batch,
                                                    imsize=self.imsize, workers=self.opt.workers, super_class=opt.super_class)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader

        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.writer = SummaryWriter('./tensorboard/%s' % opt.ex)
        self.logger = getLogger()
        self.step = 0
        self.checkpoint = './checkpoint/%s' % opt.ex
        self.datasets = opt.datasets
        self.source = opt.datasets[0]
        self.targets = opt.datasets[1:]
        for target in self.targets:
            self.best_miou[target] = 0.
        self.n_class = opt.n_class
        self.w, self.h = opt.imsize
        self.loss_fns = Losses(opt)
        self.perceptual = dict()
        self.seg = dict()
        self.last_feature = dict()
        self.name_classes = [
            "flat",
            "construction",
            "object",
            "nature",
            "sky",
            "human",
            "vehicle",
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

    def save_networks(self):
        if not os.path.exists(self.checkpoint+'/%d' % self.step):
            os.mkdir(self.checkpoint+'/%d' % self.step)
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.opt.datasets:
                    torch.save(self.nets[key][dset].state_dict(),
                               self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, dset))
            elif key == 'T':
                    for cv in self.test_converts:
                        torch.save(self.nets[key][cv].state_dict(),
                                   self.checkpoint + '/%d/net%s_%s.pth' % (self.step, key, cv))
            else:
                torch.save(self.nets[key].state_dict(), self.checkpoint + '/%d/net%s.pth' % (self.step, key))

    def load_networks(self, step):
        self.step = step
        for key in self.nets.keys():
            if key == 'D':
                for dset in self.opt.datasets:
                    self.nets[key][dset].load_state_dict(torch.load(self.checkpoint
                                                                    + '/%d/net%s_%s.pth' % (step, key, dset)))
            elif key == 'T':
                if self.opt.task == 'clf':
                    for cv in self.test_converts:
                        self.nets[key][cv].load_state_dict(torch.load(self.checkpoint
                                                                      + '/%d/net%s_%s.pth' % (step, key, cv)))
            else:
                self.nets[key].load_state_dict(torch.load(self.checkpoint + '/%d/net%s.pth' % (step, key)))

    def set_networks(self):
    
        self.nets['T_source'] = Deeplab(num_classes=self.n_class, restore_from='pretrained_model/deeplab_city_sc_59.83')
        self.nets['T_adapt'] = Deeplab(num_classes=self.n_class, restore_from='pretrained_model/deeplab_multi_72.11')
        # self.nets['P'] = VGG16()
        if self.opt.cuda:
            for net in self.nets.keys():
                self.nets[net].cuda()

    def set_eval(self):
        self.nets['T_source'].eval()
        self.nets['T_adapt'].eval()

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.opt.datasets:
            try:
                batch_data[dset] = batch_data_iter[dset].next()
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = batch_data_iter[dset].next()
        return batch_data

    def tensor_board_log(self, imgs, labels):
        nrow = 2
        test_batch = 8
        # Input Images & Recon Images
        for dset in self.opt.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('1_Input_Images/1_%s' % dset, x, self.step)

        # Segmentation GT, Pred
        self.set_eval()
        preds = dict()
        preds_adapt = dict()
        for dset in self.opt.datasets:
            x = decode_labels(labels[dset].detach(), num_images=test_batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/%s_1_GT' % dset, x, self.step)
            with torch.no_grad():
                preds[dset], _ = self.nets['T_source'](imgs[dset])
                preds_adapt[dset], _ = self.nets['T_adapt'](imgs[dset])

        for key in preds.keys():
            pred = preds[key].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            x = decode_labels(pred, num_images=test_batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/%s_2_Source_Pred' % key, x, self.step)

            pred = preds_adapt[key].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            x = decode_labels(pred, num_images=test_batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/%s_2_Adapt_Pred' % key, x, self.step)


        # 
        x = vutils.make_grid(imgs[self.opt.datasets[0]].detach(), normalize=True, scale_each=False, nrow=nrow)
        self.writer.add_image('1_Input_Images/1_%s' % self.opt.datasets[0], x, self.step)

    def eval(self):
        self.set_default()
        self.set_networks()
        self.set_eval()
        batch_data_iter = dict()
        for dset in self.opt.datasets:
            batch_data_iter[dset] = iter(self.test_loader[dset])

        for i in range(self.opt.iter):
            self.step += 1
            print('step: %d', self.step)
            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.opt.batch*4
            for dset in self.opt.datasets:
                imgs[dset], labels[dset] = batch_data[dset]
                if self.opt.cuda:
                    imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                labels[dset] = labels[dset].long()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.opt.batch*4:
                for dset in self.opt.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]
            # training
            
            # tensorboard
            self.tensor_board_log(imgs, labels)
            

if __name__ == '__main__':
    from param import get_params
    opt = get_params()
    trainer = Trainer(opt)
    trainer.eval()
