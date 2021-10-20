import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gram


class Losses:
    def __init__(self, opt):
        self.opt = opt
        self.loss_fns = dict()
        self.loss_fns['L1'] = nn.L1Loss()
        self.loss_fns['MSE'] = nn.MSELoss()
        self.loss_fns['BCE'] = nn.BCEWithLogitsLoss(size_average=True)
        self.loss_fns['CE'] = nn.CrossEntropyLoss()
        self.loss_fns['T'] = TripletLoss(self.opt.n_class)
        self.loss_fns['A'] = Class_Align_Loss(self.opt.n_class)
        if self.opt.cuda:
            for fn in self.loss_fns.values():
                fn.cuda()
        self.targets = len(opt.datasets) - 1
        self.alpha_recon = 10
        self.alpha_dis_patch = 1. / self.targets
        self.alpha_dis_domain = 1. / self.targets
        self.alpha_gen_patch = 1. / self.targets
        self.alpha_gen_domain = 1. / self.targets
        self.alpha_style = 1e4

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += self.loss_fns["L1"](imgs[dset], recon_imgs[dset])
        return self.alpha_recon * recon_loss

    def recon_single(self, img, recon_img):
        return self.alpha_recon * self.loss_fns["L1"](img, recon_img)

    def dis_single(self, real, fake):
        dis_loss = 0
        for pred in range(3):
            dis_loss += F.relu(1. - real[pred]).mean()
        for pred in range(3):
            dis_loss += F.relu(1. + fake[pred]).mean()
        return self.alpha_dis_patch*dis_loss

    def gen_single(self, fake):
        gen_loss = 0
        for pred in range(3):
            gen_loss += (-fake[pred].mean())
        return self.alpha_gen_patch*gen_loss

    def dis(self, real, fake):
        dis_loss = 0

        for dset in real.keys():
            for pred in range(3):
                dis_loss += F.relu(1. - real[dset][pred]).mean()

        for convert in fake.keys():
            for pred in range(3):
                dis_loss += F.relu(1. + fake[convert][pred]).mean()

        return self.alpha_dis_patch * dis_loss

    def dis_patch(self, real, fake):
        dis_loss = 0

        for dset in real.keys():
            dis_loss += F.relu(1. - real[dset]).mean()

        for convert in fake.keys():
            dis_loss += F.relu(1. + fake[convert]).mean()

        return self.alpha_dis_patch * dis_loss
    
    def dis_domain(self, real, fake):
        dis_loss = 0

        for dset in real.keys():
            if dset == self.opt.datasets[1]:
                dis_loss += self.loss_fns['BCE'](real[dset], torch.cuda.FloatTensor([1, 0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(real[dset]))
            elif dset == self.opt.datasets[2]:
                dis_loss += self.loss_fns['BCE'](real[dset], torch.cuda.FloatTensor([0, 1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(real[dset]))
        for convert in fake.keys():
            dis_loss += self.loss_fns['BCE'](fake[convert], torch.zeros_like(fake[convert]))
        
        return dis_loss
            

    def dis_(self, real, fake):  # multi-head
        patch_dis_loss, domain_dis_loss = 0, 0
        
        for dset in real.keys():
            patch_real, fc_real = real[dset]
            b = patch_real.size(0)
            patch_dis_loss += F.relu(1. - patch_real).mean()
            if dset == self.opt.datasets[0]:
                domain_dis_loss += self.loss_fns['CE'](fc_real, torch.zeros(b, device=patch_real.device).long())
            elif dset == self.opt.datasets[1]:
                domain_dis_loss += self.loss_fns['CE'](fc_real, torch.ones(b, device=patch_real.device).long())
            else:
                domain_dis_loss += self.loss_fns['CE'](fc_real, 2 * torch.ones(b, device=patch_real.device).long())

        for convert in fake.keys():
            patch_fake, fc_fake = fake[convert]
            patch_dis_loss += F.relu(1. + patch_fake).mean()
            if dset == self.opt.datasets[0]:
                domain_dis_loss += self.loss_fns['CE'](fc_fake, torch.zeros(b, device=patch_fake.device).long())
            elif dset == self.opt.datasets[1]:
                domain_dis_loss += self.loss_fns['CE'](fc_fake, torch.ones(b, device=patch_fake.device).long())
            else:
                domain_dis_loss += self.loss_fns['CE'](fc_fake, 2 * torch.ones(b, device=patch_fake.device).long())

        return self.alpha_dis_patch * patch_dis_loss + self.alpha_dis_domain * domain_dis_loss


    def dis_LSGAN(self, real, fake):  # multi-head
        patch_dis_loss, domain_dis_loss = 0, 0
        
        for dset in real.keys():
            patch_real, fc_real = real[dset]
            b = patch_real.size(0)
            # patch_dis_loss += F.relu(1. - patch_real).mean()
            patch_dis_loss += ((patch_real - 1)**2).mean()
            if dset == self.opt.datasets[0]:
                domain_dis_loss += self.loss_fns['CE'](fc_real, torch.zeros(b, device=patch_real.device).long())
            elif dset == self.opt.datasets[1]:
                domain_dis_loss += self.loss_fns['CE'](fc_real, torch.ones(b, device=patch_real.device).long())
            else:
                domain_dis_loss += self.loss_fns['CE'](fc_real, 2 * torch.ones(b, device=patch_real.device).long())

        for convert in fake.keys():
            patch_fake, fc_fake = fake[convert]
            # patch_dis_loss += F.relu(1. + patch_fake).mean()
            patch_dis_loss += F.relu(patch_fake**2).mean()
            if dset == self.opt.datasets[0]:
                domain_dis_loss += self.loss_fns['CE'](fc_fake, torch.zeros(b, device=patch_fake.device).long())
            elif dset == self.opt.datasets[1]:
                domain_dis_loss += self.loss_fns['CE'](fc_fake, torch.ones(b, device=patch_fake.device).long())
            else:
                domain_dis_loss += self.loss_fns['CE'](fc_fake, 2 * torch.ones(b, device=patch_fake.device).long())

        return self.alpha_dis_patch * patch_dis_loss + self.alpha_dis_domain * domain_dis_loss


    def gen(self, fake):
        gen_loss = 0
        for convert in fake.keys():
            for pred in range(3):
                gen_loss += (-fake[convert][pred].mean())
        return self.alpha_gen_patch * gen_loss

    def gen_patch(self, fake):
        gen_loss = 0
        for convert in fake.keys():
            gen_loss += (-fake[convert].mean())
        return self.alpha_gen_patch * gen_loss

    def gen_(self, fake):
        patch_gen_loss, domain_gen_loss = 0, 0
        for convert in fake.keys():
            patch_fake, fc_fake = fake[convert]
            b = patch_fake.size(0)
            patch_gen_loss += -patch_fake.mean()
            if convert[-1] == self.opt.datasets[0]:
                domain_gen_loss += self.loss_fns['CE'](fc_fake, torch.zeros(b, device=patch_fake.device).long())
            elif convert[-1] == self.opt.datasets[1]:
                domain_gen_loss += self.loss_fns['CE'](fc_fake, torch.ones(b, device=patch_fake.device).long())
            else:
                domain_gen_loss += self.loss_fns['CE'](fc_fake, 2 * torch.ones(b, device=patch_fake.device).long())
        return self.alpha_gen_patch * patch_gen_loss + self.alpha_gen_domain * domain_gen_loss

    def gen_LSGAN(self, fake):
        patch_gen_loss, domain_gen_loss = 0, 0
        for convert in fake.keys():
            patch_fake, fc_fake = fake[convert]
            b = patch_fake.size(0)
            patch_gen_loss += ((patch_fake - 1)**2).mean()
            if convert[-1] == self.opt.datasets[0]:
                domain_gen_loss += self.loss_fns['CE'](fc_fake, torch.zeros(b, device=patch_fake.device).long())
            elif convert[-1] == self.opt.datasets[1]:
                domain_gen_loss += self.loss_fns['CE'](fc_fake, torch.ones(b, device=patch_fake.device).long())
            else:
                domain_gen_loss += self.loss_fns['CE'](fc_fake, 2 * torch.ones(b, device=patch_fake.device).long())
        return self.alpha_gen_patch * patch_gen_loss + self.alpha_gen_domain * domain_gen_loss


    def gen_domain(self, fake):
        gen_loss = 0
        for convert in fake.keys():
            if convert[-1] == self.opt.datasets[1]:
                gen_loss += self.loss_fns['BCE'](fake[convert], torch.cuda.FloatTensor([1, 0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(fake[convert]))
            if convert[-1] == self.opt.datasets[2]:
                gen_loss += self.loss_fns['BCE'](fake[convert], torch.cuda.FloatTensor([0, 1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(fake[convert]))
        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            sdset, tdset = cv.split('2')
            content_perceptual_loss += self.loss_fns['MSE'](perceptual[sdset][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            sdset, tdset = cv.split('2')
            for gr in range(len(style_gram[tdset])):
                style_percptual_loss += self.alpha['style'][cv] * self.loss_fns['MSE'](style_gram[tdset][gr], style_gram_converted[cv][gr])
        return style_percptual_loss

    # def region_wise_style_loss(self, net, seg, imgs, converted_imgs):
    #     total_loss = 0.
    #     for convert in converted_imgs.keys():
    #         source, target = convert.split('2')
    #         mask_source = F.one_hot(seg[source] + 1, num_classes=self.opt.n_class+1).permute(0,3,1,2).float()  # B x (cls+1) x H x W
    #         mask_source = mask_source[:, 1:, :, :]
    #         freq_source = torch.sum(mask_source, dim=(0,2,3))
    #         mask_target = F.one_hot(seg[target] + 1, num_classes=self.opt.n_class+1).permute(0,3,1,2).float()  # B x (cls+1) x H x W
    #         mask_target = mask_target[:, 1:, :, :]
    #         freq_target = torch.sum(mask_target, dim=(0,2,3))
    #         for cls in range(freq_source.size(0)):
    #             if freq_source[cls] > 0 and freq_target[cls] > 0:
    #                 vgg_feature_target = net(mask_target[:, cls, :, :].unsqueeze(1) * imgs[target])
    #                 vgg_feature_s2t = net(mask_source[:, cls, :, :].unsqueeze(1) * converted_imgs[convert])
    #                 gram_target = [gram(fmap) for fmap in vgg_feature_target]
    #                 gram_s2t = [gram(fmap) for fmap in vgg_feature_s2t]
    #                 for gr in range(len(gram_target)):
    #                     total_loss += (self.alpha_style * self.loss_fns['MSE'](gram_target[gr], gram_s2t[gr]))
    #     return total_loss / self.opt.n_class

    def region_wise_style_loss(self, net, seg, imgs, converted_imgs):
        total_loss = 0.
        for convert in converted_imgs.keys():
            source, target = convert.split('2')
            mask_source = F.one_hot(downsample(seg[source] + 1, 'nearest'), num_classes=self.opt.n_class+1).permute(0,3,1,2).float()  # B x (cls+1) x H x W
            mask_source = mask_source[:, 1:, :, :]
            freq_source = torch.sum(mask_source, dim=(0,2,3))
            mask_target = F.one_hot(downsample(seg[target] + 1, 'nearest'), num_classes=self.opt.n_class+1).permute(0,3,1,2).float()  # B x (cls+1) x H x W
            mask_target = mask_target[:, 1:, :, :]
            freq_target = torch.sum(mask_target, dim=(0,2,3))
            for cls in range(freq_source.size(0)):
                if freq_source[cls] > 0 and freq_target[cls] > 0:
                    vgg_feature_target = net(mask_target[:, cls, :, :].unsqueeze(1) * downsample(imgs[target], 'bilinear'))
                    vgg_feature_s2t = net(mask_source[:, cls, :, :].unsqueeze(1) * downsample(converted_imgs[convert], 'bilinear'))
                    gram_target = [gram(fmap) for fmap in vgg_feature_target]
                    gram_s2t = [gram(fmap) for fmap in vgg_feature_s2t]
                    for gr in range(len(gram_target)):
                        total_loss += (self.alpha_style * self.loss_fns['MSE'](gram_target[gr], gram_s2t[gr]))
        return total_loss / self.opt.n_class


    def consistency(self, contents, styles, contents_converted, styles_converted):
        consistency_loss = 0
        for cv in contents_converted.keys():
            sdset, tdset = cv.split('2')
            consistency_loss += self.loss_fns['L1'](contents[cv], contents_converted[cv])
            consistency_loss += self.loss_fns["L1"](styles[tdset], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

    def triplet(self, f, seg):
        return self.loss_fns['T'](f, seg)

    def class_align(self, f1, f2, seg):
        return self.loss_fns['A'](f1, f2, seg)


def downsample(img, mode):
    if mode == 'nearest':
        h, w = img.shape[1:]
        return F.interpolate(img.unsqueeze(1).float(), size=(h//4, w//4), mode=mode).squeeze(1).long()
    else:
        h, w = img.shape[2:]
        return F.interpolate(img, size=(h//4, w//4), mode=mode)


class TripletLoss(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.n = dict()
        # self.center = dict()
        
        
    def forward(self, f, seg):
        loss = 0.
        f_norm = F.normalize(f, p=2, dim=1)
        b, c, h, w = f.size()
        # update center
        seg_ = F.interpolate(seg.detach().unsqueeze(1).float(), size=f.shape[2:], mode='nearest').squeeze(1)
        mask = []
        center = dict()
        for cls in range(self.n_class):
            self.n[cls] = 0
            mask.append(1* (seg_==cls).unsqueeze(1))  # B x 1 x H x W
            if mask[cls].sum() != 0:
                self.n[cls] = 1
                new_sample = ((mask[cls]*f).mean(dim=(0, 2, 3), keepdim=True))  # 1 x C x 1 x 1
                new_sample = F.normalize(new_sample, p=2)
                
                center[str(cls)] = new_sample
                # else:
                #     self.center[cls] += ((new_sample - self.center[cls])/self.n[cls])
            # if self.n[cls] != 0:
            #     f_differ[str(cls)] = f_norm - center[str(cls)]  # B x 2048 x 65 x 129
        # compute loss
        for cls in center.keys():
            f_cls_differ = torch.norm(mask[int(cls)] *(f_norm - center[str(cls)]), p=2, dim=1)  # B x H x W
            for other_cls in center.keys():
                if cls != other_cls: 
                    f_other_cls_differ = torch.norm(mask[int(cls)] * (f_norm - center[str(other_cls)]), p=2, dim=1)        
                    loss = loss + F.relu(f_cls_differ-f_other_cls_differ+0.2).sum()
                    
        return loss / (b*c*h*w)


class Class_Align_Loss(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
    def forward(self, f_source, f_convert, seg):
        f_norm_source = F.normalize(f_source, p=2, dim=1)
        f_norm_convert = F.normalize(f_convert, p=2, dim=1)
        b, c, h, w = f_source.size()
        seg_ = F.interpolate(seg.detach().unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1)
        mask = []
        center_source = []
        center_convert = []
        for cls in range(self.n_class):
            mask_tmp = 1* (seg_==cls).unsqueeze(1)
            
            if mask_tmp.sum() != 0:
                mask.append(mask_tmp)  # B x 1 x H x W
                center_source.append(F.normalize(((mask_tmp*f_norm_source).mean(dim=(0, 2, 3))).unsqueeze(0), p=2))  # 1xC
                center_convert.append(F.normalize(((mask_tmp*f_norm_convert).mean(dim=(0, 2, 3))).unsqueeze(0), p=2))  # 1xC
                
        exist_class = len(mask)
        
        anchor = torch.cat([center_source[cls].repeat(exist_class-1, 1) for cls in range(exist_class)], dim=0)  # (ex(ex-1))xC
        positive = torch.cat([center_convert[cls].repeat(exist_class-1, 1) for cls in range(exist_class)], dim=0)  # (ex(ex-1))xC
        
        negative = torch.cat([center_convert[cls] for cls in range(exist_class)], dim=0)  # exxC
        negative = torch.cat([torch.cat([negative[:cls], negative[cls+1:]], dim=0) for cls in range(exist_class)], dim=0)


        return F.triplet_margin_loss(anchor, positive, negative, margin=0.2, p=2)