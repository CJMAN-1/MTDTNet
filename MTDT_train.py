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

            
        self.converts, self.task_converts = set_converts(self.source, self.targets)
        self.w, self.h = opt.imsize
        self.loss_fns = Losses(opt)

        if opt.super_class:
            self.n_class = 7
        else:
            self.n_class = 19


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
        if not os.path.exists(self.checkpoint+'/%d' % (self.step)):
            os.mkdir(self.checkpoint+'/%d' % (self.step))
        for key in self.nets.keys():
            torch.save(self.nets[key], self.checkpoint + '/%d/net%s.pth' % (self.step, key))
            # torch.save(self.nets[key].state_dict(), self.checkpoint + '/%d%s/net%s.pth' % (self.step, miou, key))
        self.current_best_T = self.checkpoint + '/%d/netT' % (self.step)

    def set_networks(self):
        self.nets['E'] = Encoder()
        self.nets['G'] = Generator()
        self.nets['SE'] = Style_Encoder()
        self.nets['D'] = Multi_Head_Discriminator(len(self.datasets))
        self.nets['DT'] = Domain_Transfer(self.targets)
        self.nets['LE'] = Label_Embed()
        self.nets['CDCA'] = Cross_Domain_Class_Alignment(self.converts, self.n_class)
        # initialization
        for net in self.nets.keys():
            init_params(self.nets[net])

        self.nets['P'] = VGG19()
        
        # self.nets['P'] = VGG16()
        if self.opt.cuda:
            for net in self.nets.keys():
                self.nets[net].cuda()

    def set_optimizers(self):
        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['D'] = optim.Adam(self.nets['D'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        # self.optims['CAD'] = optim.Adam(self.nets['CAD'].parameters(), lr=self.opt.lr_dra,
        #                                         betas=(self.opt.beta1, 0.999),
        #                                         weight_decay=self.opt.weight_decay)
        self.optims['E'] = optim.Adam(self.nets['E'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['SE'] = optim.Adam(self.nets['SE'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['DT'] = optim.Adam(self.nets['DT'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['LE'] = optim.Adam(self.nets['LE'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)

    def set_zero_grad(self):
        for net in self.nets.keys():
            self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            self.nets[net].train()

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
        contents, styles, D_outputs_real, D_outputs_fake = dict(), dict(), dict(), dict()
        DD_outputs_real, DD_outputs_fake = dict(), dict()
        features, converted_features = dict(), dict()
        converted_imgs = dict()
        seg_last_feature = dict()
        gamma, beta = dict(), dict()
        with torch.no_grad():
            for dset in self.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                if dset in self.targets:
                    self.nets['DT'].update(features[dset], dset)

            gamma[self.source], beta[self.source] = self.nets['SE'](imgs[self.source])
                
            for convert in self.converts:
                source, target = convert.split('2')
                gamma[convert], beta[convert] = self.nets['DT'](gamma[source], beta[source], target=target)
                converted_features[convert] = gamma[convert]*self.nets['LE'](labels[source]) + beta[convert]
                converted_imgs[convert] = self.nets['G'](converted_features[convert])

        for target in self.targets:
            # D_outputs_real[dset] = self.nets['D'](imgs[dset])
            D_outputs_real[target] = self.nets['D'](slice_patches(imgs[target]))
        for convert in self.converts:
            # D_outputs_fake[convert] = self.nets['D'](converted_imgs[convert])
            D_outputs_fake[convert] = self.nets['D'](slice_patches(converted_imgs[convert]))
        
        loss_dis = self.loss_fns.dis_(D_outputs_real, D_outputs_fake)
        loss = loss_dis
        loss.backward()
        self.optims['D'].step()
        self.losses['D'] = loss_dis.data.item()
      
    def train_esg(self, imgs, labels):
        self.set_zero_grad()
        direct_recon, indirect_recon, contents, styles, D_outputs_fake = dict(), dict(), dict(), dict(), dict()
        converted_imgs, converted_contents, converted_styles = dict(), dict(), dict()
        gamma, beta = dict(), dict()
        features, converted_features = dict(), dict()
        vgg_feature = dict()
        for dset in self.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            direct_recon[dset] = self.nets['G'](features[dset])
            vgg_feature[dset] = self.nets['P'](imgs[dset])
        
        gamma[self.source], beta[self.source] = self.nets['SE'](imgs[self.source])
        indirect_recon[self.source] = self.nets['G'](gamma[self.source]*self.nets['LE'](labels[self.source]) + beta[self.source])
            
        for convert in self.converts:
            source, target = convert.split('2')
            gamma[convert], beta[convert] = self.nets['DT'](gamma[source], beta[source], target=target)
            converted_imgs[convert] = self.nets['G'](gamma[convert]*self.nets['LE'](labels[source]) + beta[convert])
            # D_outputs_fake[convert] = self.nets['D'](converted_imgs[convert])
            D_outputs_fake[convert] = self.nets['D'](slice_patches(converted_imgs[convert]))
            vgg_feature[convert] = self.nets['P'](converted_imgs[convert])
            
        G_loss = self.loss_fns.gen_(D_outputs_fake)
        Recon_loss = self.loss_fns.recon(imgs, direct_recon) + self.loss_fns.recon_single(imgs[self.source], indirect_recon[self.source])
        Consis_loss = 0.
        Style_loss = 0.
        # Style_loss = self.loss_fns.region_wise_style_loss(self.nets['P'], seg, imgs, converted_imgs)
        for convert in self.converts:
            source, target = convert.split('2')
            Consis_loss += F.mse_loss(vgg_feature[source][-1], vgg_feature[convert][-1])
            gram_target = [gram(fmap) for fmap in vgg_feature[target]]
            gram_convert = [gram(fmap) for fmap in vgg_feature[convert]]
            for gr in range(len(gram_target)):
                Style_loss += 1e3 * F.mse_loss(gram_target[gr], gram_convert[gr])
            # Consis_loss += self.loss_fns.recon_single(imgs[source], cycle_recon[convert])
            # Consis_loss += self.loss_fns.recon_single(w[dset]*contents[source], cycle_w[convert]*converted_contents[convert])
            # Consis_loss += self.loss_fns.recon_single(styles[target], converted_styles[convert])
            # Consis_loss += self.loss_fns.recon_single(last_feature[source], last_feature[convert]) -> train_task

        loss_esg = G_loss + Recon_loss + Consis_loss + Style_loss
        loss_esg.backward()
        for net in ['E', 'G', 'DT', 'SE', 'LE']:
            self.optims[net].step()
        self.losses['G'] = G_loss.data.item()
        self.losses['R'] = Recon_loss.data.item()
        self.losses['C'] = Consis_loss.data.item()
        self.losses['S'] = Style_loss.data.item()

    def tensor_board_log(self, imgs, labels):
        nrow = 2
        direct_recon, indirect_recon, contents, styles, D_outputs_fake = dict(), dict(), dict(), dict(), dict()
        converted_imgs, converted_contents, converted_styles = dict(), dict(), dict()
        gamma, beta = dict(), dict()
        features, converted_features = dict(), dict()
        w = dict()
        last_feature = dict()
        for dset in self.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            direct_recon[dset] = self.nets['G'](features[dset])
        
        gamma[self.source], beta[self.source] = self.nets['SE'](imgs[self.source])
        indirect_recon[self.source] = self.nets['G'](gamma[self.source]*self.nets['LE'](labels[self.source]) + beta[self.source])
            
        for convert in self.converts:
            source, target = convert.split('2')
            gamma[convert], beta[convert] = self.nets['DT'](gamma[source], beta[source], target=target)
            converted_imgs[convert] = self.nets['G'](gamma[convert]*self.nets['LE'](labels[source]) + beta[convert])
               
        # Input Images & Recon Images
        for dset in self.opt.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('1_Input_Images/1_%s' % dset, x, self.step)
            # x = vutils.make_grid(slice_patches(imgs[dset].detach()), normalize=True, scale_each=False, nrow=4)
            # self.writer.add_image('1_Input_Images/2_slice_%s' % dset, x, self.step)
            x = vutils.make_grid(direct_recon[dset].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('2_Recon_Images/1_direct_%s' % dset, x, self.step)
        x = vutils.make_grid(indirect_recon[self.source].detach(), normalize=True, scale_each=False, nrow=nrow)
        self.writer.add_image('2_Recon_Images/2_indirect_%s' % self.source, x, self.step)


        # Converted Images
        for convert in converted_imgs.keys():
            x = vutils.make_grid(converted_imgs[convert].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('3_Converted_Images/%s' % convert, x, self.step)
            # x = vutils.make_grid(cycle_recon[convert].detach(), normalize=True, scale_each=False, nrow=nrow)
            # self.writer.add_image('2_Recon_Images/3_cycle_%s' % convert, x, self.step)

        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

        self.set_train()

        # 
        x = vutils.make_grid(imgs[self.opt.datasets[0]].detach(), normalize=True, scale_each=False, nrow=nrow)
        self.writer.add_image('1_Input_Images/1_%s' % self.opt.datasets[0], x, self.step)

    def print_loss(self):
        losses = ''
        for key in self.losses:
            losses += ('%s: %.2f|'% (key, self.losses[key])) 
        self.logger.info(
            '[%d/%d] %s| %s'
            % (self.step, self.opt.iter, losses, self.opt.ex)
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
            
            self.train_dis(imgs, labels)
            self.train_esg(imgs, labels)
            
            self.print_loss()

            # tensorboard
            if self.step % self.opt.tensor_freq == 0:
                with torch.no_grad():
                    self.tensor_board_log(imgs, labels)

            if self.step >= 200 and self.step % 100 ==0:
                self.save_networks()

if __name__ == '__main__':
    from param import get_params
    opt=get_params()
    trainer = Trainer(opt)
    trainer.train()