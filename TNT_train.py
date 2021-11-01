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
                    self.best_miou[target] = 40.0
                    # self.best_miou[target] = 46.3
                    self.min_miou[target] = 33.0
                    # self.min_miou[target] = 38.0
            
        self.converts, self.task_converts = set_converts(self.source, self.targets)
        self.w, self.h = opt.imsize
        self.loss_fns = Losses(opt)

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
        miou = '%d'% self.step
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
        self.nets['CDCA'] = Cross_Domain_Class_Alignment(self.converts, self.n_class)
        for net in self.nets.keys():
            init_params(self.nets[net])

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
            dset = self.source
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

        

        
        if self.step > 100:
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


                # D_outputs_fake[target] = 0.001 * self.nets['PD'](pred[target])
                # loss_task += self.loss_fns.gen_patch(D_outputs_fake)
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
                if not self.step > 100:
                    self.nets['CDCA'].update(last_feature[convert], labels[self.source], convert)
                    self.nets['CDCA'].update(last_feature[target], seg_target, target)
                else:
                    self.nets['CDCA'].update(last_feature[convert], new_source_label[convert], convert)
                    self.nets['CDCA'].update(last_feature[target], new_target_label[target], target)
     
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
            if self.reload:
                self.skip = True
                self.nets['T'] = Deeplab(num_classes=self.n_class, restore_from=self.current_best_T)
                if self.opt.cuda:
                    self.nets['T'].cuda()
            
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
            dset = self.source
            imgs[dset], labels[dset] = batch_data[dset]
            if self.opt.cuda:
                imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
            labels[dset] = labels[dset].long()
            
            

            # evaluation - no_reload
            self.train_task(imgs, labels)
            self.print_loss()
            if self.step % self.opt.eval_freq == 0 and self.step>100:
                self.skip = False
                for target in self.targets:
                    if not self.skip:
                        self.eval(target)

            # evalutation - reload
            # self.reload = False
            # while not self.reload:
            #     self.train_task(imgs, labels)
            #     self.print_loss()
            #     self.reload = True

            #     if self.step % self.opt.eval_freq == 0 and self.step>100:
            #         self.skip = False
            #         for target in self.targets:
            #             if not self.skip:
            #                 self.eval(target)

if __name__ == '__main__':
    from param import get_params
    opt=get_params()
    trainer = Trainer(opt)
    trainer.train()