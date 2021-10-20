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
        for target in self.targets:
            self.best_miou[target] = 0.
        if self.source == 'G':
            if opt.super_class:
                self.n_class = 7
            else:
                self.n_class = 19
        elif self.source == 'S':
            if opt.super_class:
                self.n_class = 7
            else:
                self.n_class = 16
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
        self.nets['G'] = Generator()
        # self.nets['CSE'] = Content_Style_Encoder()
        # self.nets['CSG'] = Content_Style_Decoder()
        self.nets['CSE'] = SWAEEncoder()
        self.nets['CSG'] = SWAEDecoder()
        # self.nets['D'] = Perceptual_Discriminator(len(self.datasets), self.n_class)
        self.nets['D'] = Multi_Head_Discriminator(len(self.datasets))
        # self.nets['D'] = Multi_Domain_Discriminator(len(self.datasets))
        self.nets['W'] = Domain_Normalization(self.datasets)
        self.nets['M'] = Adaptive_Style_Memory_Bank(self.targets, size=8)
        # initialization
        if self.opt.load_networks_step is not None:
            self.load_networks(self.opt.load_networks_step)
        for net in self.nets.keys():
            init_params(self.nets[net])
        
        if self.opt.super_class:
            self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from='pretrained_model/deeplab_city_sc_63.10')
        else:
            self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from='pretrained_model/deeplab_gta5_36.94')
        # self.nets['P'] = VGG16()
        if self.opt.cuda:
            for net in self.nets.keys():
                self.nets[net].cuda()

    def set_optimizers(self):
        self.optims['CSE'] = optim.Adam(self.nets['CSE'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['CSG'] = optim.Adam(self.nets['CSG'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['D'] = optim.Adam(self.nets['D'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['E'] = optim.Adam(self.nets['E'].parameters(), lr=self.opt.lr_dra,
                                                betas=(self.opt.beta1, 0.999),
                                                weight_decay=self.opt.weight_decay)
        self.optims['W'] = optim.Adam(self.nets['W'].parameters(), lr=self.opt.lr_dra,
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
        contents, styles, D_outputs_real, D_outputs_fake = dict(), dict(), dict(), dict()
        features, converted_features = dict(), dict()
        converted_imgs = dict()
        w = dict()
        with torch.no_grad():
            for dset in self.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                self.nets['W'](features[dset], features[dset], dset, dset, update=True)
                contents[dset], styles[dset] = self.nets['CSE'](features[dset])
                if dset in self.targets:
                    self.nets['M'].update(contents[dset], styles[dset], dset)
            for convert in self.converts:
                source, target = convert.split('2')
                w[convert] = self.nets['W'](features[source], features[target], source, target)
                adaptive_style = self.nets['M'](contents[source], target)
                converted_features[convert] = self.nets['CSG'](w[convert]*contents[source], adaptive_style)
                converted_imgs[convert] = self.nets['G'](converted_features[convert])
                # self.perceptual[convert] = self.nets['P'](slice_patches(converted_imgs[convert]))

        for target in self.targets:
            D_outputs_real[dset] = self.nets['D'](imgs[dset])
            # D_outputs_real[dset] = self.nets['D'](slice_patches(imgs[dset]))
        for convert in self.converts:
            D_outputs_fake[convert] = self.nets['D'](converted_imgs[convert])
            # D_outputs_fake[convert] = self.nets['D'](slice_patches(converted_imgs[convert]))
            
        loss_dis = self.loss_fns.dis_(D_outputs_real, D_outputs_fake)
        loss_dis.backward()
        self.optims['D'].step()
        self.losses['D'] = loss_dis.data.item()

    def train_task(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        contents, styles, D_outputs_real, D_outputs_fake = dict(), dict(), dict(), dict()
        features, converted_features = dict(), dict()
        converted_imgs = dict()
        last_feature = dict()
        w = dict()
        with torch.no_grad():
            for dset in self.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                contents[dset], styles[dset] = self.nets['CSE'](features[dset])
            for convert in self.converts:
                source, target = convert.split('2')
                w[convert] = self.nets['W'](features[source], features[target], source, target)
                adaptive_style = self.nets['M'](contents[source], target)
                converted_features[convert] = self.nets['CSG'](w[convert]*contents[source], adaptive_style)
                converted_imgs[convert] = self.nets['G'](converted_features[convert])

        loss_task = 0.
        
        T_feature = dict()
        converted_T_feature =dict()
        converted_content_features = dict()
        
        class_weight = class_weight_by_frequency(labels[self.source], self.n_class)

        _, last_feature_source = self.nets['T'](imgs[self.source], lbl=labels[self.source], weight=class_weight)
        # loss_task += self.nets['T'].loss_seg
        # loss_task += self.loss_fns.triplet(last_feature, labels[self.source])
            
        for convert in self.task_converts:
            # with torch.no_grad():
            #     converted_T_feature[convert] = self.nets['G']((contents[convert[0]], styles[convert[-1]]), feature_only=True)
            _, last_feature[convert] = self.nets['T'](converted_imgs[convert], lbl=labels[self.source], weight=class_weight)
            # loss_task += self.nets['T'].loss_seg
            loss_task += self.loss_fns.class_align(last_feature_source, last_feature[convert], labels[self.source])
        loss_task.backward()
        self.optims['T'].step()
        self.losses['T'] = loss_task.data.item()

    def train_esg(self, imgs, labels):
        self.set_zero_grad()
        direct_recon, indirect_recon, contents, styles, D_outputs_fake = dict(), dict(), dict(), dict(), dict()
        converted_imgs, converted_contents, converted_styles = dict(), dict(), dict()
        features, converted_features = dict(), dict()
        cycle_converted_features = dict()
        indirect_features = dict()
        cycle_features = dict()
        cycle_recon = dict()
        w = dict()
        cycle_w = dict()
        for dset in self.datasets:
            features[dset] = self.nets['E'](imgs[dset])
            contents[dset], styles[dset] = self.nets['CSE'](features[dset])
            w[dset] = self.nets['W'](features[dset], features[dset], dset, dset)
            indirect_features[dset] = self.nets['CSG'](w[dset]*contents[dset], styles[dset])
            direct_recon[dset] = self.nets['G'](features[dset])
            indirect_recon[dset] = self.nets['G'](indirect_features[dset])
            
        for convert in self.converts:
            source, target = convert.split('2')
            w[convert] = self.nets['W'](features[source], features[target], source, target)
            adaptive_style = self.nets['M'](contents[source], target)
            converted_features[convert] = self.nets['CSG'](w[convert]*contents[source], adaptive_style)
            converted_imgs[convert] = self.nets['G'](converted_features[convert])
            D_outputs_fake[convert] = self.nets['D'](converted_imgs[convert])
            # D_outputs_fake[convert] = self.nets['D'](slice_patches(converted_imgs[convert]))
            cycle_converted_features[convert] = self.nets['E'](converted_imgs[convert])
            cycle_w[convert] = self.nets['W'](cycle_converted_features[convert], features[source], target, source)
            converted_contents[convert], _ = self.nets['CSE'](cycle_converted_features[convert])
            cycle_features[source] = self.nets['CSG'](cycle_w[convert]*converted_contents[convert], styles[source])
            cycle_recon[convert] = self.nets['G'](cycle_features[source])
            
        
        G_loss = self.loss_fns.gen_(D_outputs_fake)
        Recon_loss = self.loss_fns.recon(imgs, direct_recon) + self.loss_fns.recon(imgs, indirect_recon)
        Consis_loss = 0.
        # Style_loss = self.loss_fns.region_wise_style_loss(self.nets['P'], seg, imgs, converted_imgs)
        for convert in self.converts:
            source, target = convert.split('2')
            # Consis_loss += self.loss_fns.recon_single(imgs[source], cycle_recon[convert])
            Consis_loss += self.loss_fns.recon_single(w[dset]*contents[source], cycle_w[convert]*converted_contents[convert])
            # Consis_loss += self.loss_fns.recon_single(styles[target], converted_styles[convert])
            # Consis_loss += self.loss_fns.recon_single(last_feature[source], last_feature[convert]) -> train_task

        loss_esg = G_loss + Recon_loss + Consis_loss # + Style_loss
        loss_esg.backward()
        for net in ['E', 'G', 'CSE', 'CSG', 'W']:
            self.optims[net].step()
        self.losses['G'] = G_loss.data.item()
        self.losses['R'] = Recon_loss.data.item()
        self.losses['C'] = Consis_loss.data.item()
        # self.losses['S'] = Style_loss.data.item()

    def tensor_board_log(self, imgs, labels):
        nrow = 2
        with torch.no_grad():
            direct_recon, indirect_recon, contents, styles, D_outputs_fake = dict(), dict(), dict(), dict(), dict()
            converted_imgs, converted_contents, converted_styles = dict(), dict(), dict()
            features, converted_features = dict(), dict()
            indirect_features = dict()
            cycle_converted_features, cycle_w, cycle_features, cycle_recon = dict(), dict(), dict(), dict()
            w = dict()
            for dset in self.datasets:
                features[dset] = self.nets['E'](imgs[dset])
                contents[dset], styles[dset] = self.nets['CSE'](features[dset])
                w[dset] = self.nets['W'](features[dset], features[dset], dset, dset)
                indirect_features[dset] = self.nets['CSG'](w[dset]*contents[dset], styles[dset])
                direct_recon[dset] = self.nets['G'](features[dset])
                indirect_recon[dset] = self.nets['G'](indirect_features[dset])
                
            for convert in self.converts:
                source, target = convert.split('2')
                w[convert] = self.nets['W'](features[source], features[target], source, target)
                adaptive_style = self.nets['M'](contents[source], target)
                converted_features[convert] = self.nets['CSG'](w[convert]*contents[source], adaptive_style)
                converted_imgs[convert] = self.nets['G'](converted_features[convert])
                cycle_converted_features[convert] = self.nets['E'](converted_imgs[convert])
                cycle_w[convert] = self.nets['W'](cycle_converted_features[convert], features[source], target, source)
                converted_contents[convert], _ = self.nets['CSE'](cycle_converted_features[convert])
                cycle_features[source] = self.nets['CSG'](cycle_w[convert]*converted_contents[convert], styles[source])
                cycle_recon[convert] = self.nets['G'](cycle_features[source])
               
        # Input Images & Recon Images
        for dset in self.opt.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('1_Input_Images/1_%s' % dset, x, self.step)
            # x = vutils.make_grid(slice_patches(imgs[dset].detach()), normalize=True, scale_each=False, nrow=4)
            # self.writer.add_image('1_Input_Images/2_slice_%s' % dset, x, self.step)
            x = vutils.make_grid(direct_recon[dset].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('2_Recon_Images/1_direct_%s' % dset, x, self.step)
            x = vutils.make_grid(indirect_recon[dset].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('2_Recon_Images/2_indirect_%s' % dset, x, self.step)


        # Converted Images
        for convert in converted_imgs.keys():
            x = vutils.make_grid(converted_imgs[convert].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('3_Converted_Images/%s' % convert, x, self.step)
            x = vutils.make_grid(cycle_recon[convert].detach(), normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('2_Recon_Images/3_cycle_%s' % convert, x, self.step)

        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

        # Segmentation GT, Pred
        self.set_eval()
        preds = dict()
        for dset in self.opt.datasets:
            x = decode_labels(labels[dset].detach(), num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/1_GT_%s' % dset, x, self.step)
            with torch.no_grad():
                preds[dset], _ = self.nets['T'](imgs[dset])

        for key in preds.keys():
            pred = preds[key].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            x = decode_labels(pred, num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/2_Pred_%s' % key, x, self.step)

            pred_confidence = pred2seg(preds[key]).data.cpu().numpy()
            x = decode_labels(pred_confidence, num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/3_Pred_%s_confidence' % key, x, self.step)
        
        self.set_train()

        # 
        x = vutils.make_grid(imgs[self.opt.datasets[0]].detach(), normalize=True, scale_each=False, nrow=nrow)
        self.writer.add_image('1_Input_Images/1_%s' % self.opt.datasets[0], x, self.step)

    def eval(self, target):
        self.set_eval()

        miou = 0.
        confusion_matrix = np.zeros((self.n_class,) * 2)
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
                if self.opt.cuda:
                    imgs, labels = imgs.cuda(), labels.cuda()
                labels = labels.long()
                pred, _ = self.nets['T'](imgs)
                pred = pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                gt = labels.data.cpu().numpy()
                confusion_matrix += MIOU(gt, pred, num_class=self.n_class)

                score = np.diag(confusion_matrix) / (
                            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(
                        confusion_matrix))
                miou = 100 * np.nanmean(score)

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
            self.writer.add_scalar('MIoU/%s' %(self.source + '2' + target), miou, self.step)
            
            if miou > self.best_miou[target]:
                # torch.save(self.nets['T'].state_dict(), './pretrained_model/deeplab_multi_%.2f.pth' % miou)
                self.best_miou[target] = miou
                self.writer.add_scalar('Best_MIoU/%s' %(self.source + '2' + target), self.best_miou[target], self.step)
                # self.save_networks()
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
            # training
            
            self.train_dis(imgs, labels)
            for esg in range(2):
                self.train_esg(imgs, labels)
            if self.step > 70:
                self.train_task(imgs, labels)
            # tensorboard
            if self.step % self.opt.tensor_freq == 0:
                self.tensor_board_log(imgs, labels)
            # evaluation
            if self.step % self.opt.eval_freq == 0 and self.step>80:
                for target in self.targets:
                    self.eval(target)
            self.print_loss()
