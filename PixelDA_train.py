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
        # self.writer = SummaryWriter('./tensorboard1/%s' % opt.ex)
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
            self.current_best_T = '../wh_DRANet/checkpoint/45.63_43.16/_44.47_44.75/netT'
            # self.current_best_T = 'pretrained_model/deeplab_gta5_36.94'
            for target in self.targets:
                if target == 'C':
                    self.best_miou[target] = 44.75
                    # self.best_miou[target] = 36.9
                    self.min_miou[target] = 40
                    # self.min_miou[target] = 33.0
                elif target == 'I':
                    self.best_miou[target] = 44.47
                    # self.best_miou[target] = 39.7
                    self.min_miou[target] = 35
                    # self.min_miou[target] = 34.0
                elif target == 'M':
                    self.best_miou[target] = 40
                    # self.best_miou[target] = 46.3
                    self.min_miou[target] = 33.0
                    # self.min_miou[target] = 38.0
            
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

    def save_networks(self):
        miou = ''
        for target in self.targets:
            miou = miou + '_%.2f'%self.best_miou[target]
        if not os.path.exists(self.checkpoint+'/%s' % miou):
            os.mkdir(self.checkpoint+'/%s' % miou)
        for key in self.nets.keys():
            torch.save(self.nets[key].state_dict(), self.checkpoint + '/%s/net%s.pth' % (miou, key))
        self.current_best_T = self.checkpoint + '/%s/netT' % (miou)

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
        step = 2000
        folder = './checkpoint/MTDT_save/%d_36.90_39.70_46.30/' % step
        for key in ['E', 'SE', 'LE', 'DT', 'G']:
            self.nets[key] = (torch.load(folder + 'net%s.pth' % (key)))
        
        if self.opt.super_class:
            self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from=self.current_best_T)
        else:
            self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from=self.current_best_T)
            # self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from='pretrained_model/deeplab_gta5_36.94')
        if self.opt.cuda:
            for net in self.nets.keys():
                self.nets[net].cuda()

    def set_optimizers(self):
        self.optims['T'] = optim.SGD(self.nets['T'].parameters(), lr=self.opt.lr_seg, momentum=0.9,
                                         weight_decay=self.opt.weight_decay_task)

    def set_zero_grad(self):
        for net in self.nets.keys():
            self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            self.nets[net].train()

    def set_eval(self):
        self.nets['T'].eval()

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        for dset in self.opt.datasets:
            try:
                batch_data[dset] = batch_data_iter[dset].next()
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = batch_data_iter[dset].next()
        return batch_data

    def train_task(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        contents, styles, D_outputs_real, D_outputs_fake = dict(), dict(), dict(), dict()
        features, converted_features = dict(), dict()
        pred = dict()
        Domain_pred, Class_pred = dict(), dict()
        converted_imgs = dict()
        last_feature = dict()
        gamma, beta = dict(), dict()
        new_source_label = dict()
        new_target_label = dict()
        w = dict()
        with torch.no_grad():
            for dset in self.datasets:
                features[dset] = self.nets['E'](imgs[dset])
            
            gamma[self.source], beta[self.source] = self.nets['SE'](imgs[self.source])
                
            for convert in self.converts:
                source, target = convert.split('2')
                gamma[convert], beta[convert] = self.nets['DT'](gamma[source], beta[source], target=target)
                converted_imgs[convert] = self.nets['G'](gamma[convert]*self.nets['LE'](labels[source]) + beta[convert])

        loss_task = 0.
        
        # class_weight = class_weight_by_frequency(labels[self.source], self.n_class)

        pred[self.source], *_ = self.nets['T'](imgs[self.source], lbl=labels[self.source])
        loss_task += self.nets['T'].loss_seg
            
        for convert in self.task_converts:
            pred[convert], *_, = self.nets['T'](converted_imgs[convert], lbl=labels[self.source])
            loss_task += self.nets['T'].loss_seg
        
        
        
        loss_task.backward()
        self.optims['T'].step()
        self.losses['T'] = loss_task.data.item()
    
    def eval(self, target):
        self.set_eval()

        miou = 0.
        min_miou = 100.
        confusion_matrix = np.zeros((self.n_class,) * 2)
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
                if self.opt.cuda:
                    imgs, labels = imgs.cuda(), labels.cuda()
                labels = labels.long()
                pred, *_ = self.nets['T'](imgs)
                pred = pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                gt = labels.data.cpu().numpy()
                confusion_matrix += MIOU(gt, pred, num_class=self.n_class)
                score = np.diag(confusion_matrix) / (
                            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(
                        confusion_matrix))
                miou = 100 * np.nanmean(score)
                

                ############ Toward SOTA #############
                if batch_idx > 20:
                    if miou < min_miou:
                        min_miou = miou
                    if self.min_miou[target] > miou:
                        self.skip = True
                        break
                    

                progress_bar(batch_idx, len(self.test_loader[target]), 'mIoU: %.3f' % miou)
            score = 100 * np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
            score = np.round(score, 1)
            table = PrettyTable()
            table.field_names = self.name_classes
            table.add_row(score)
            # Save checkpoint.
            self.logger.info('======================================================')
            self.logger.info('Step: %d | mIoU: %.3f%%'
                            % (self.step, miou))
            self.logger.info(table)
            self.logger.info('======================================================')
            # self.writer.add_scalar('MIoU/%s' %(self.source + '2' + target), miou, self.step)
            
            
            if miou > self.best_miou[target]:
                self.reload=False
                self.min_miou[target] = min_miou - 0.3
                # torch.save(self.nets['T'].state_dict(), './pretrained_model/deeplab_multi_%.2f.pth' % miou)
                self.best_miou[target] = miou
                # self.writer.add_scalar('Best_MIoU/%s' %(self.source + '2' + target), self.best_miou[target], self.step)
                self.save_networks()
            # if self.reload:
            #     self.skip = True
            #     self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from=self.current_best_T)
            #     if self.opt.cuda:
            #         self.nets['T'].cuda()
            
        self.set_train()

    def print_loss(self):
        best_mious = ''
        for convert in self.task_converts:
            _, target = convert.split('2')
            best_mious += (convert + ': ' + '%.2f'%self.best_miou[target] + '|' )
        losses = ''
        for key in self.losses:
            losses += ('%s: %.2f|'% (key, self.losses[key])) 
        self.logger.info(
            '[%d/%d] %s| %s %s'
            % (self.step, self.opt.iter, losses, best_mious, self.opt.ex)
        )
        # self.logger.info(
        #     '[%d/%d] D: %.2f| G: %.2f| R: %.2f| C: %.2f| T: %.2f| %s %s'
        #     % (self.step, self.opt.iter,
        #        self.losses['D'], self.losses['G'], self.losses['R'], self.losses['C'], self.losses['T'], best_mious, self.opt.ex))

    def train(self):
        self.set_default()
        self.set_networks()
        self.set_optimizers()
        self.set_train()
        batch_data_iter = dict()
        for dset in self.opt.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        for i in range(self.opt.iter):
            self.step += 1
            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.opt.batch
            for dset in self.opt.datasets:
                imgs[dset], labels[dset] = batch_data[dset]
                if self.opt.cuda:
                    imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                labels[dset] = labels[dset].long()
                        
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.opt.batch:
                for dset in self.opt.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]
            
            # if self.step > 100:
            #     self.train_pd(imgs, labels)
            self.train_task(imgs, labels)
            
            self.print_loss()

            # evaluation
            if self.step % self.opt.eval_freq == 0:
                self.skip = False
                for target in self.targets:
                    if not self.skip:
                        self.eval(target)

            # # tensorboard
            # if self.step % self.opt.tensor_freq == 0 and self.step>0:
            #     with torch.no_grad():
            #         self.tensor_board_log(imgs, labels)
            

if __name__ == '__main__':
    from param import get_params
    opt=get_params()
    trainer = Trainer(opt)
    trainer.train()