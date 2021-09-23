import torch
import torch.nn as nn
import torch.nn.functional as F


class Losses:
    def __init__(self, opt, source, target):
        self.opt = opt
        self.loss_fns = dict()
        self.loss_fns['L1'] = nn.L1Loss()
        self.loss_fns['MSE'] = nn.MSELoss()
        self.loss_fns['BCE'] = nn.BCEWithLogitsLoss(size_average=True)
        self.loss_fns['CE'] = nn.CrossEntropyLoss()
        if self.opt.cuda:
            for fn in self.loss_fns.values():
                fn.cuda()
        self.alpha_recon = 30
        self.alpha_dis_patch = 0.5 / 3
        self.alpha_dis_domain = 0.25
        self.alpha_gen_patch = 1 / 3
        self.alpha_gen_domain = 0.5
        self.source = source
        self.target = target

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += self.loss_fns["L1"](imgs[dset], recon_imgs[dset])
        return self.alpha_recon * recon_loss

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

    def dis_(self, real, fake):
        patch_dis_loss, domain_dis_loss = 0, 0

        for dset in real.keys():
            patch_real, fc_real = real[dset]
            patch_dis_loss += F.relu(1. - patch_real).mean()
            if dset == self.source:
                domain_dis_loss += self.loss_fns['BCE'](fc_real, torch.zeros_like(fc_real))
            else:
                domain_dis_loss += self.loss_fns['BCE'](fc_real, torch.ones_like(fc_real))

        for convert in fake.keys():
            patch_fake, fc_fake = fake[convert]
            patch_dis_loss += F.relu(1. + patch_fake).mean()
            if convert[-1] == self.source:
                domain_dis_loss += self.loss_fns['BCE'](fc_fake, torch.zeros_like(fc_fake))
            else:
                domain_dis_loss += self.loss_fns['BCE'](fc_fake, torch.ones_like(fc_fake))

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
            patch_gen_loss += -patch_fake.mean()
            if convert[-1] == self.source:
                domain_gen_loss += self.loss_fns['BCE'](fc_fake, torch.zeros_like(fc_fake))
            else:
                domain_gen_loss += self.loss_fns['BCE'](fc_fake, torch.ones_like(fc_fake))
        return self.alpha_gen_patch * patch_gen_loss + self.alpha_gen_domain * domain_gen_loss

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

    def consistency(self, contents, styles, contents_converted, styles_converted):
        consistency_loss = 0
        for cv in contents_converted.keys():
            sdset, tdset = cv.split('2')
            consistency_loss += self.loss_fns['L1'](contents[cv], contents_converted[cv])
            consistency_loss += self.loss_fns["L1"](styles[tdset], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

