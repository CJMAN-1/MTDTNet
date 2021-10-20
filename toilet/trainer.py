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


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.imsize = (2 * opt.imsize, opt.imsize)  # (width, height)
        self.best_miou = 0.

        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.opt.datasets:
            train_loader, test_loader = get_dataset(dataset=dset, batch=self.opt.batch,
                                                    imsize=self.imsize, workers=self.opt.workers)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader

        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.writer = SummaryWriter('./tensorboard/%s' % opt.ex)
        self.logger = getLogger()
        self.step = 0
        self.checkpoint = './checkpoint/%s' % opt.ex

        self.loss_fns = Losses(opt)

    def set_default(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
                if self.opt.task == 'clf':
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
        
        self.nets['D'] = Gram_Discriminator()
        # initialization
        if self.opt.load_networks_step is not None:
            self.load_networks(self.opt.load_networks_step)
        init_params(self.nets['D'])
        self.nets['T'] = Deeplab(num_classes=19, restore_from='pretrained_model/gta5')

        if self.opt.cuda:
            for net in self.nets.keys():
                self.nets[net].cuda()

    def set_optimizers(self):
        self.optims['D'] = optim.Adam(self.nets['D'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
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

    def train_dis(self, imgs, labels):  # Train Discriminators(D)
        self.set_zero_grad()
        for dset in self.opt.datasets:
            if dset == 'C':
                with torch.no_grad():
                    # FDA_t2s = FDA_source_to_target(imgs[dset], imgs['G']).cuda()

                    pred = self.nets['T'](imgs[dset])
                    pred = F.softmax(1000*pred, dim=1)
                    parts = make_seg_parts(imgs[dset], pred)  # (B x class) x 3 x H x W
                D_outputs_real = self.nets['D'](parts)
                parts = make_seg_parts(imgs[dset], labels[dset])
                D_outputs_fake = self.nets['D'](parts)
            else:
                # FDA_s2t = FDA_source_to_target(imgs[dset], imgs['C']).cuda()
                # parts = make_seg_parts(FDA_s2t, labels[dset])
                # parts = make_seg_parts(imgs[dset], labels[dset])
                # D_outputs_fake = self.nets['D'](parts)
                pass

        errD = self.loss_fns.dis(D_outputs_real, D_outputs_fake)
        errD.backward()
        self.optims['D'].step()
        self.losses['D'] = errD.data.item()

    def train_task(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        pred = self.nets['T'](imgs['C'])
        pred = F.softmax(1000*pred, dim=1)
        parts = make_seg_parts(imgs['C'], pred)
        D_outputs_fake = self.nets['D'](parts)

        errT = self.loss_fns.gen(D_outputs_fake)
        errT.backward()
        self.optims['T'].step()
        self.losses['T'] = errT.data.item()

        self.set_zero_grad()
        pred = self.nets['T'](imgs['G'], lbl=labels['G'])
        loss_seg_src = self.nets['T'].loss_seg
        loss_seg_src.backward()
        self.optims['T'].step()





    def tensor_board_log(self, imgs, labels):
        nrow = 4
        # Input Images
        for dset in self.opt.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/%s' % dset, x, self.step)

        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

        # Segmentation GT, Pred
        self.set_eval()
        preds = dict()
        for dset in self.opt.datasets:
            x = decode_labels(labels[dset].detach(), num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('2_GT/%s' % dset, x, self.step)
            preds[dset] = self.nets['T'](imgs[dset])

        for key in preds.keys():
            pred = preds[key].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            x = decode_labels(pred, num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('4_Prediction/%s' % key, x, self.step)
        self.set_train()
        x = vutils.make_grid(imgs[self.opt.datasets[0]].detach(), normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('1_Input_Images/%s' % self.opt.datasets[0], x, self.step)

    def eval(self):
        self.set_eval()

        miou = 0.
        confusion_matrix = np.zeros((19,) * 2)
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.test_loader['C']):
                if self.opt.cuda:
                    imgs, labels = imgs.cuda(), labels.cuda()
                labels = labels.long()
                pred = self.nets['T'](imgs)
                pred = pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                gt = labels.data.cpu().numpy()
                confusion_matrix += MIOU(gt, pred)

                score = np.diag(confusion_matrix) / (
                            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(
                        confusion_matrix))
                miou = 100 * np.nanmean(score)

                progress_bar(batch_idx, len(self.test_loader['C']), 'mIoU: %.3f' % miou)
            # Save checkpoint.
            self.logger.info('======================================================')
            self.logger.info('Epoch: %d | Acc: %.3f%%'
                            % (self.step, miou))
            self.logger.info('======================================================')
            self.writer.add_scalar('MIoU/G2C', miou, self.step)
            if miou > self.best_miou:
                self.best_miou = miou
                self.writer.add_scalar('Best MIoU/G2C', self.best_miou, self.step)
                # self.save_networks()
        self.set_train()

    def print_loss(self):
        self.logger.info(
            '[%d/%d] D: %.2f | T: %.2f| %.2f %s'
            % (self.step, self.opt.iter,
               self.losses['D'], self.losses['T'], self.best_miou, self.opt.ex))

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
           
            # training
            self.train_dis(imgs, labels)
            self.train_task(imgs, labels)
            # tensorboard
            if self.step % self.opt.tensor_freq == 0:
                self.tensor_board_log(imgs, labels)
            # evaluation
            if self.step % self.opt.eval_freq == 0:
                self.eval()
            self.print_loss()

