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

    def save_networks(self):
        miou = ''
        for target in self.targets:
            miou = miou + '_%.2f'%self.best_miou[target]
        if not os.path.exists(self.checkpoint+'/%d%s' % (self.step, miou)):
            os.mkdir(self.checkpoint+'/%d%s' % (self.step, miou))
        for key in self.nets.keys():
            torch.save(self.nets[key], self.checkpoint + '/%d%s/net%s.pth' % (self.step, miou, key))
            # torch.save(self.nets[key].state_dict(), self.checkpoint + '/%d%s/net%s.pth' % (self.step, miou, key))
        self.current_best_T = self.checkpoint + '/%d%s/netT' % (self.step, miou)

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
        self.nets['SE'] = Style_Encoder()
        # self.nets['D'] = Perceptual_Discriminator(len(self.datasets), self.n_class)
        self.nets['D'] = Multi_Head_Discriminator(len(self.datasets))
        # self.nets['D'] = Multi_Domain_Discriminator(len(self.datasets))
        self.nets['DT'] = Domain_Transfer(self.targets)
        # self.nets['C11'] = nn.Conv2d(2048, 64, 1, 1, 0)
        self.nets['LE'] = Label_Embed()
        # self.nets['CAD'] = Class_Alignment_Discriminator(len(self.datasets), self.n_class)
        self.nets['CDCA'] = Cross_Domain_Class_Alignment(self.converts, self.n_class)
        self.nets['PD'] = Prediction_Discriminator(self.n_class)
        # self.nets['DD'] = Domain_Discriminator(len(self.datasets))
        # self.nets['WPN'] = Weight_Prediction_Networks(self.targets, self.n_class)
        # initialization
        if self.opt.load_networks_step is not None:
            self.load_networks(self.opt.load_networks_step)
        for net in self.nets.keys():
            init_params(self.nets[net])

        self.nets['P'] = VGG19()
        
        if self.opt.super_class:
            self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from='pretrained_model/deeplab_city_sc_63.10')
        else:
            self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from='pretrained_model/deeplab_gta5_36.94')
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
        self.optims['PD'] = optim.Adam(self.nets['PD'].parameters(), lr=1e-4,
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
        # self.optims['WPN'] = optim.Adam(self.nets['WPN'].parameters(), lr=self.opt.lr_dra,
        #                                         betas=(self.opt.beta1, 0.999),
        #                                         weight_decay=self.opt.weight_decay)
        # self.optims['C11'] = optim.Adam(self.nets['C11'].parameters(), lr=self.opt.lr_dra,
        #                                         betas=(self.opt.beta1, 0.999),
        #                                         weight_decay=self.opt.weight_decay)
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
        

    def train_pd(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        contents, styles, D_outputs_real, DD_outputs_fake = dict(), dict(), dict(), dict()
        D_outputs_real, D_outputs_fake = dict(), dict()
        features, converted_features = dict(), dict()
        pred = dict()
        Domain_pred, Class_pred = dict(), dict()
        converted_imgs = dict()
        last_feature = dict()
        gamma, beta = dict(), dict()
        w = dict()
        with torch.no_grad():
            for dset in self.datasets:
                features[dset] = self.nets['E'](imgs[dset])
            
            gamma[self.source], beta[self.source] = self.nets['SE'](imgs[self.source])
                
            for convert in self.converts:
                source, target = convert.split('2')
                gamma[convert], beta[convert] = self.nets['DT'](gamma[source], beta[source], target=target)
                converted_imgs[convert] = self.nets['G'](gamma[convert]*self.nets['LE'](labels[source]) + beta[convert])

            pred[self.source], *_ = self.nets['T'](imgs[self.source], lbl=labels[self.source])
            for target in self.targets:
                pred[target], *_ = self.nets['T'](converted_imgs[convert])
            for convert in self.task_converts:
                pred[convert], *_ = self.nets['T'](converted_imgs[convert])

        D_outputs_real[self.source] = self.nets['PD'](pred[self.source])
        for convert in self.task_converts:
            target = convert[-1]
            D_outputs_fake[target] = self.nets['PD'](pred[target])
        
        loss_pd = self.loss_fns.dis_patch(D_outputs_real, D_outputs_fake)
        loss_pd.backward()
        self.optims['PD'].step()
        self.losses['PD'] = loss_pd.data.item()


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

        pred[self.source], _, last_feature[self.source] = self.nets['T'](imgs[self.source], lbl=labels[self.source])
        # loss_task += 0.1 * self.nets['T'].loss_seg
        
        for target in self.targets:
            pred[target], _, last_feature[target] = self.nets['T'](converted_imgs[convert])
            
        for convert in self.task_converts:
            pred[convert], _, last_feature[convert] = self.nets['T'](converted_imgs[convert], lbl=labels[self.source])
            # loss_task += 0.1 * self.nets['T'].loss_seg
        
        # for convert in self.task_converts:
        #     target = convert[-1]
        #     D_outputs_fake[target] = 0.001 * self.nets['PD'](pred[target])
        # loss_task += self.loss_fns.gen_patch(D_outputs_fake)

        

        
        if self.step > 300:
            _, pred[self.source], last_feature[self.source] = self.nets['T'](imgs[self.source], lbl=labels[self.source])
            loss_task += self.nets['T'].loss_seg
            for convert in self.task_converts:
                target = convert[-1]
                seg_target = pred2seg(pred[target])
                w[convert], w[target] = self.nets['CDCA'](last_feature[convert], last_feature[target], labels[self.source], seg_target, convert)
                
                new_source_label[convert] = labels[self.source].clone().detach()
                new_source_label[convert][w[convert]!=new_source_label[convert]] = -1
                new_target_label[target] = seg_target.clone().detach()
                new_target_label[target][w[target]!=new_target_label[target]] = -1
                loss_task += self.nets['T'].CrossEntropy2d(pred[convert], new_source_label[convert])


                D_outputs_fake[target] = 0.001 * self.nets['PD'](pred[target])
                loss_task += self.loss_fns.gen_patch(D_outputs_fake)
                # loss_task += self.nets['CDCA'].feature_align_loss(last_feature[target], new_target_label, seg_target, convert)
                # loss_task += self.nets['CDCA'].feature_align_loss(last_feature[convert], new_source_label, seg_target, convert)
                loss_task += 0.001*self.nets['T'].CrossEntropy2d(pred[target], new_target_label[target])
                
            loss_task.backward()
            self.optims['T'].step()
            self.losses['T'] = loss_task.data.item()
        
        # update centroid
        for convert in self.task_converts:
            target = convert[-1]
            seg_target = pred2seg(pred[target])
            with torch.no_grad():
                if not self.step > 300:
                    self.nets['CDCA'].update(last_feature[convert], labels[self.source], convert)
                    self.nets['CDCA'].update(last_feature[target], seg_target, target)
                else:
                    self.nets['CDCA'].update(last_feature[convert], new_source_label[convert], convert)
                    self.nets['CDCA'].update(last_feature[target], new_target_label[target], target)
            


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

        # Segmentation GT, Pred
        self.set_eval()
        preds = dict()
        for dset in self.opt.datasets:
            x = decode_labels(labels[dset].detach(), num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/1_GT_%s' % dset, x, self.step)
            preds[dset], _, last_feature[dset] = self.nets['T'](imgs[dset])
        for convert in self.converts:
            *_, last_feature[convert] = self.nets['T'](converted_imgs[convert])


        for key in preds.keys():
            pred = preds[key].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            x = decode_labels(pred, num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/2_Pred_%s' % key, x, self.step)

        for convert in self.task_converts:
            target = convert[-1]
            seg_target = pred2seg(preds[target])
            w[convert], w[target] = self.nets['CDCA'](last_feature[convert], last_feature[target], labels[self.source], seg_target, convert)
            new_source_label = labels[self.source].clone().detach()
            new_source_label[w[convert]!=new_source_label] = -1
            new_target_label = seg_target.clone().detach()
            new_target_label[w[target]!=new_target_label] = -1
            seg_convert = new_source_label.data.cpu().numpy()
            seg_target = new_target_label.data.cpu().numpy()
            w_convert = w[convert].data.cpu().numpy()
            w_target = w[target].data.cpu().numpy()

            x = decode_labels(seg_convert, num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/3_Attention_%s' % convert, x, self.step)
            x = decode_labels(seg_target, num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/3_Attention_%s' % target, x, self.step)

            x = decode_labels(w_convert, num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/4_Centroid_Pred_%s' % convert, x, self.step)
            x = decode_labels(w_target, num_images=self.opt.batch, super_class=self.opt.super_class)
            x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            self.writer.add_image('4_Segmentation/4_Centroid_Pred_%s' % target, x, self.step)

            # pred_confidence = pred2seg(preds[key]).data.cpu().numpy()
            # x = decode_labels(pred_confidence, num_images=self.opt.batch, super_class=self.opt.super_class)
            # x = vutils.make_grid(x, normalize=True, scale_each=False, nrow=nrow)
            # self.writer.add_image('4_Segmentation/3_Pred_%s_confidence' % key, x, self.step)


        
        
        self.set_train()

        # 
        x = vutils.make_grid(imgs[self.opt.datasets[0]].detach(), normalize=True, scale_each=False, nrow=nrow)
        self.writer.add_image('1_Input_Images/1_%s' % self.opt.datasets[0], x, self.step)

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
                
                if batch_idx > 10:
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
            self.writer.add_scalar('MIoU/%s' %(self.source + '2' + target), miou, self.step)
            
            
            if miou > self.best_miou[target]:
                self.best_miou[target] = miou
                self.min_miou[target] = min_miou - 0.3
                self.writer.add_scalar('Best_MIoU/%s' %(self.source + '2' + target), self.best_miou[target], self.step)
                self.save_networks()
            
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
                # remove too many sidewalk sample
                # if self.n_class == 19 and dset == self.source:
                #     b, h, w = labels[dset].size()
                #     freq = F.one_hot(labels[dset]+1, num_classes=self.n_class+1).sum(dim=(0,1,2))
                #     freq = freq[1:].float()
                #     side_ratio = freq[1] / (b*h*w)
                #     while side_ratio > 0.1:
                #         print(side_ratio)
                #         try:
                #             batch_data[dset] = batch_data_iter[dset].next()
                #         except StopIteration:
                #             batch_data_iter[dset] = iter(self.train_loader[dset])
                #             batch_data[dset] = batch_data_iter[dset].next()
                #         imgs[dset], labels[dset] = batch_data[dset]
                #         if self.opt.cuda:
                #             imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                #         labels[dset] = labels[dset].long()
                #         b, h, w = labels[dset].size()
                #         freq = F.one_hot(labels[dset]+1, num_classes=self.n_class+1).sum(dim=(0,1,2))
                #         freq = freq[1:].float()
                #         side_ratio = freq[1] / (b*h*w)
                #         print(side_ratio)
                        

                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.opt.batch:
                for dset in self.opt.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]
            
            self.train_dis(imgs, labels)
            for esg in range(1):
                self.train_esg(imgs, labels)
            
            
            # if self.step > 300:
            #     self.train_pd(imgs, labels)
            # if self.step > 150:
            #     self.train_task(imgs, labels)
            
            self.print_loss()

            # # evaluation
            # if self.step % self.opt.eval_freq == 0 and self.step>300:
            #     self.skip = False
            #     for target in self.targets:
            #         if not self.skip:
            #             self.eval(target)

            # # tensorboard
            # if self.step % self.opt.tensor_freq == 0 and self.step>300:
            #     with torch.no_grad():
            #         self.tensor_board_log(imgs, labels)
            

            if self.step >= 200 and self.step % 100 ==0:
                self.save_networks()
