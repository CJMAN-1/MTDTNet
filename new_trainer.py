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

def set_converts(source, targets):
    converts = list()
    task_converts = list()

    for target in targets:
        converts.append(source + '2' + target)
        converts.append(target + '2' + source)
        task_converts.append(source + '2' + target)

    return converts, task_converts

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.imsize = opt.imsize
        self.best_miou = dict()

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
        self.datasets = opt.datasets
        self.source = opt.datasets[0]
        self.targets = opt.datasets[1:]
        for target in self.targets:
            self.best_miou[target] = 0.
        self.n_class = opt.n_class
        self.converts, self.task_converts = set_converts(self.source, self.targets)

        self.loss_fns = Losses(opt, self.source, self.target)

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
        self.nets['E'] = Encoder()
        # TODO : content encoder 추가해야함.
        '''
        self.nets['CE'] = 
        '''
        self.nets['G'] = Generator()
        self.nets['D'] = dict()
        for dset in self.datasets:
            self.nets['D'][dset] = PatchGAN_Discriminator()

        # initialization
        if self.opt.load_networks_step is not None:
            self.load_networks(self.opt.load_networks_step)
        for net in self.nets.keys():
            if net == 'D':
                    for dset in self.datasets:
                        init_params(self.nets[net][dset])
            else:
                init_params(self.nets[net])
        
        self.nets['T'] = Deeplab(num_classes=19, restore_from='pretrained_model/deeplab_gta5_36.94')

        if self.opt.cuda:
            for net in self.nets.keys():
                if net == 'D':
                    for dset in self.datasets:
                        self.nets[net][dset].cuda()
                else:
                    self.nets[net].cuda()

    def set_optimizers(self):
        self.optims['E'] = optim.Adam(self.nets['E'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['CE'] = optim.Adam(self.nets['CE'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['D'] = dict()
        for dset in self.opt.datasets:
            self.optims['D'][dset] = optim.Adam(self.nets['D'][dset].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['T'] = optim.SGD(self.nets['T'].parameters(), lr=self.opt.lr_seg, momentum=0.9,
                                         weight_decay=self.opt.weight_decay_task)

    def set_zero_grad(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.opt.datasets:
                    self.nets[net][dset].zero_grad()
            else:
                self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            if net == 'D':
                for dset in self.opt.datasets:
                    self.nets[net][dset].train()
            else:
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
        features, converted_imgs, D_outputs_fake, D_outputs_real, seg = dict(), dict(), dict(), dict(), dict()
        content_features, gamma, beta= dict(), dict(), dict()
        with torch.no_grad():
            for dset in self.opt.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                content_features[dset] = self.nets['CE'](features[dset])

            seg[self.source] = labels[self.source]
            for target in self.targets:
                seg[target] = pred2seg(self.nets['T'](imgs[target]))
            gamma, beta = self.nets['SEAN'](features, seg)
            
            #Generator
            for convert in self.converts:
                source, target = convert.split('2')
                converted_imgs[convert] = self.nets['G'](gamma[convert]*content_features[source] + beta[convert])

        # TODO : discriminator 수정필요.
        '''
        for dset in self.opt.datasets:
            D_outputs_real[dset] = self.nets['D'][dset](slice_patches(imgs[dset]))
        
        for convert in self.converts:
            D_outputs_fake[convert] = self.nets['D'][](slice_patches(converted_imgs[convert]))
        '''

        loss_dis = self.loss_fns.dis(D_outputs_real, D_outputs_fake)
        loss_dis.backward()
        for dset in self.datasets:
            self.optims['D'][dset].step()
        self.losses['D'] = loss_dis.data.item()

    def train_task(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        features, converted_imgs, D_outputs_fake, D_outputs_real, seg = dict(), dict(), dict(), dict(), dict()
        content_features, gamma, beta= dict(), dict(), dict()
        pred = dict()

        with torch.no_grad():
            for dset in self.opt.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                content_features[dset] = self.nets['CE'](features[dset])

            seg[self.source] = labels[self.source]
            for target in self.targets:
                seg[target] = pred2seg(self.nets['T'](imgs[target]))
            gamma, beta = self.nets['SEAN'](features, seg)
            
            #Generator
            for convert in self.task_converts:
                source, target = convert.split('2')
                converted_imgs[convert] = self.nets['G'](gamma[convert]*content_features[source] + beta[convert])


        pred[self.source] = self.nets['T'](imgs[self.source], lbl=labels[self.source])
        loss_seg_src = self.nets['T'].loss_seg
        loss_task = loss_seg_src
        for convert in self.task_converts:
            pred[convert] = self.nets['T'](converted_imgs[convert], lbl=labels[self.source])
            loss_seg_s2t = self.nets['T'].loss_seg
            loss_task = loss_task + loss_seg_s2t
        
        loss_task.backward()
        self.optims['T'].step()
        self.losses['T'] = loss_task.data.item()

    def train_esg(self, imgs, labels):
        self.set_zero_grad()
        features, converted_imgs, D_outputs_fake, seg = dict(), dict(), dict(), dict()
        content_features, gamma, beta= dict(), dict(), dict()
        direct_recon_imgs, indirect_recon_imgs = dict(), dict()
        converted_content_features = dict()
        # direct recon 
        for dset in self.opt.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            direct_recon_imgs[dset] = self.nets['G'](features[dset])
            content_features[dset] = self.nets['CE'](features[dset])

        seg[self.source] = labels[self.source]
        for target in self.targets:
            seg[target] = pred2seg(self.nets['T'](imgs[target]))
        gamma, beta = self.nets['SEAN'](features, seg)
        
        #Generator
        # indirect recon
        for dset in self.opt.datasets:
            indirect_recon_imgs[dset] = self.nets['G'](gamma[dset]*content_features[dset] + beta[dset])

        for convert in self.converts:
            source, target = convert.split('2')
            converted_imgs[convert] = self.nets['G'](gamma[convert]*content_features[source] + beta[convert])
        '''
        for convert in self.converts:
            D_outputs_fake[convert] = self.nets['D'][](slice_patches(converted_imgs[convert]))
        '''
        
        for convert in self.converts:
            features[convert] = self.nets['E'](converted_imgs[convert])
            converted_content_features[convert] = self.nets['CE'](features[convert])
       
        G_loss = self.loss_fns.gen(D_outputs_fake)
        Recon_loss = self.loss_fns.recon(imgs, direct_recon_imgs) + self.loss_fns.recon(imgs, indirect_recon_imgs)
        # TODO : consis_loss 새로 짜야함.
        Consis_loss = self.loss_fns.recon(content_features, converted_content_features)

        loss_esg = G_loss + Recon_loss + Consis_loss 
        loss_esg.backward()
        for net in ['E', 'CE', 'G']:
            self.optims[net].step()
        self.losses['G'] = G_loss.data.item()
        self.losses['R'] = Recon_loss.data.item()
        self.losses['C'] = Consis_loss.data.item()

    def tensor_board_log(self, imgs, labels):
        nrow = 2
        with torch.no_grad():
            features, converted_imgs, seg = dict(), dict(), dict()
            content_features, gamma, beta= dict(), dict(), dict()
            direct_recon_imgs, indirect_recon_imgs = dict(), dict()
            
            # direct recon 
            for dset in self.opt.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                direct_recon_imgs[dset] = self.nets['G'](features[dset])
                content_features[dset] = self.nets['CE'](features[dset])

            seg[self.source] = labels[self.source]
            for target in self.targets:
                seg[target] = pred2seg(self.nets['T'](imgs[target]))
            gamma, beta = self.nets['SEAN'](features, seg)
            
            #Generator
            # indirect recon
            for dset in self.opt.datasets:
                indirect_recon_imgs[dset] = self.nets['G'](gamma[dset]*content_features[dset] + beta[dset])

            for convert in self.converts:
                source, target = convert.split('2')
                converted_imgs[convert] = self.nets['G'](gamma[convert]*content_features[source] + beta[convert])

        # Input Images & Recon Images
        for dset in self.opt.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/1_%s' % dset, x, self.step)
            x = vutils.make_grid(slice_patches(imgs[dset].detach()), normalize=True, scale_each=True, nrow=4)
            self.writer.add_image('1_Input_Images/2_slice_%s' % dset, x, self.step)
            x = vutils.make_grid(direct_recon_imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('2_Recon_Images/1_direct_%s' % dset, x, self.step)
            x = vutils.make_grid(indirect_recon_imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('2_Recon_Images/2_indirect_%s' % dset, x, self.step)


        # Converted Images
        for convert in converted_imgs.keys():
            x = vutils.make_grid(converted_imgs[convert].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('3_Converted_Images/%s' % convert, x, self.step)

        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

        # Segmentation GT, Pred
        self.set_eval()
        preds = dict()
        for dset in self.opt.datasets:
            x = decode_labels(labels[dset].detach(), num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('4_Segmentation/1_GT_%s' % dset, x, self.step)
            preds[dset] = self.nets['T'](imgs[dset])

        for key in preds.keys():
            pred = preds[key].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            x = decode_labels(pred, num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('4_Segmentation/2_Pred_%s' % key, x, self.step)
        
        self.set_train()

        # 
        x = vutils.make_grid(imgs[self.opt.datasets[0]].detach(), normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('1_Input_Images/1_%s' % self.opt.datasets[0], x, self.step)

    def eval(self, target):
        self.set_eval()

        miou = 0.
        confusion_matrix = np.zeros((19,) * 2)
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
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
            
            if miou > self.best_miou[target]:
                self.best_miou[target] = miou
                self.writer.add_scalar('Best MIoU/G2C', self.best_miou[target], self.step)
                # self.save_networks()
        self.set_train()

    def print_loss(self):
        best_mious = ''
        for target in self.targets:
            best_mious += (target + ': ' + '%.2f'%self.best_miou[target] + '|' )
        self.logger.info(
            '[%d/%d] D: %.2f| G: %.2f| R: %.2f| C: %.2f| T: %.2f| %s %s'
            % (self.step, self.opt.iter,
               self.losses['D'], self.losses['G'], self.losses['R'], self.losses['C'], self.losses['T'], best_mious, self.opt.ex))

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
            # print(F.one_hot(labels['G']+1, num_classes=self.n_class+1).size())  # b x h x w x class
            # print(labels['G'].size())
            self.train_dis(imgs, labels)
            self.train_task(imgs, labels)
            self.train_esg(imgs, labels)
            # tensorboard
            if self.step % self.opt.tensor_freq == 0:
                self.tensor_board_log(imgs, labels)
            # evaluation
            if self.step % self.opt.eval_freq == 0:
                for target in self.targets:
                    self.eval(target)
            self.print_loss()

