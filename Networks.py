import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.padding import ReflectionPad2d
from torchvision import models
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from batchinstancenorm import BatchInstanceNorm2d as Normlayer
import functools
from functools import partial
import torchvision.transforms as ttransforms
from utils import *
import torch.nn.init as init


# def normalize(x):
#     image_transforms = ttransforms.Compose([
#         ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
#     ])
#     for b in range(x.size(0)):
#         x[b] = image_transforms(x[b])
#     return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        # bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            # batch_size x in_channels x 64 x 64
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
            # batch_size x filters x 64 x 64
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, inputs):
        output = self.main(inputs)
        output += self.shortcut(inputs)
        return output


class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        # bin = functools.partial(Normlayer, affine=True)
        self.Encoder_Conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        # output = F.normalize(self.Encoder_Conv(inputs), dim=0) # batch_size x 512 x 10 x 6
        output = self.Encoder_Conv(inputs)
        return output


class Style_Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Style_Encoder, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        # bin = functools.partial(Normlayer, affine=True)
        self.Encoder_Conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.gamma = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )
        self.beta = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, inputs):
        output = self.Encoder_Conv(inputs) # batch_size x 512 x 10 x 6
        # gamma = F.normalize(self.gamma(output), dim=0)
        # beta = F.normalize(self.beta(output), dim=0)
        gamma = self.gamma(output)
        beta = self.beta(output)
        return gamma, beta


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Decoder_Conv = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            # batch_size x 3 x 1280 x 768
            spectral_norm(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.Tanh()
        )
    def forward(self, x):
        return self.Decoder_Conv(x)


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2,7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7,12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12,21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21,25):
            self.to_relu_4_2.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_4_2)
        return out


class PatchGAN_Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(PatchGAN_Discriminator, self).__init__()
        # self.Conv = nn.Sequential(
        #     # batch_size x 32 x 640 x 384
        #     spectral_norm(nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     spectral_norm(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),            
        #     spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     spectral_norm(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)),
        # )
        self.Conv = nn.Sequential(
            # batch_size x 32 x 640 x 384
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=True)),
        )

    def forward(self, inputs):
        patch_output = self.Conv(inputs)
        return patch_output


class Multi_Head_Discriminator(nn.Module):
    def __init__(self, num_domains, channels=3):
        super(Multi_Head_Discriminator, self).__init__()
        self.Conv = nn.Sequential(
            # input size: 256x256
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            # nn.InstanceNorm2d(64),
            # nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.Patch = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            # nn.InstanceNorm2d(256),
            # nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)), # 
            # nn.InstanceNorm2d(512),
            # nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=True)),
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(64*64*128, 500)),
            nn.ReLU(),
            spectral_norm(nn.Linear(500, num_domains))
        )

    def forward(self, inputs):
        conv_output = self.Conv(inputs)
        patch_output = self.Patch(conv_output)
        fc_output = self.fc(conv_output.view(conv_output.size(0), -1))
        # fc_output = self.fc(conv_output).view(conv_output.size(0), -1)
        return (patch_output, fc_output)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_2 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4,8):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(8,13):
            self.to_relu_3_2.add_module(str(x), features[x])
        for x in range(13,20):
            self.to_relu_4_2.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        # h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        # h_relu_2_2 = h
        h = self.to_relu_3_2(h)
        # h_relu_3_2 = h
        # h = self.to_relu_4_2(h)
        # h_relu_4_2 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_2, h_relu_4_2)
        return h


class Label_Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv2d(1, 64, 1, 1, 0)
    
    def forward(self, seg):
        return self.embed(F.interpolate(seg.unsqueeze(1).float(), size=(256, 512), mode='nearest'))


class D_AdaIN_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, targets):
        super().__init__()
        mid_ch = min(in_ch, out_ch)
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, mid_ch, 3, 1, 1))
        self.D_adain1 = D_AdaIN(mid_ch, targets)
        self.conv2 = spectral_norm(nn.Conv2d(mid_ch, out_ch, 3, 1, 1))
        self.D_adain2 = D_AdaIN(out_ch, targets)
        
        self.conv_s = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
        self.D_adain_s = D_AdaIN(out_ch, targets)

    def forward(self, feature, target_mean, target_std, target):
        x_s = self.D_adain_s(self.conv_s(feature), target_mean, target_std, target)  # shortcut
        dx = self.conv1(feature)
        dx = self.conv2(F.relu(self.D_adain1(dx, target_mean, target_std, target)))
        dx = self.D_adain2(dx, target_mean, target_std, target)
        return F.relu(x_s + dx)


class D_AdaIN(nn.Module):
    def __init__(self, in_ch, targets):
        super().__init__()
        self.IN = nn.InstanceNorm2d(in_ch)
        self.mlp_mean = nn.ModuleDict()
        self.mlp_std = nn.ModuleDict()
        for dset in targets:
            self.mlp_mean[dset] = nn.Linear(64, 64)
            self.mlp_std[dset] = nn.Linear(64, 64)

    def forward(self, feature, target_mean, target_std, target):
        return self.mlp_std[target](target_std).unsqueeze(-1).unsqueeze(-1)*self.IN(feature) + self.mlp_mean[target](target_mean).unsqueeze(-1).unsqueeze(-1)


class Domain_Transfer(nn.Module):
    def __init__(self, targets):
        super().__init__()
        self.n = dict()
        self.m = dict()
        self.s = dict()
        self.w = 1

        for dset in targets:
            self.n[dset] = 0
            self.m[dset] = 0
            self.s[dset] = 0
        
        self.gamma_res1 = D_AdaIN_ResBlock(64, 64, targets)
        self.gamma_res2 = D_AdaIN_ResBlock(64, 64, targets)
        self.beta_res1 = D_AdaIN_ResBlock(64, 64, targets)
        self.beta_res2 = D_AdaIN_ResBlock(64, 64, targets)

    def forward(self, gamma, beta, target):
        # Domain mean, std
        target_mean = self.m[target].mean(dim=(0,2,3)).unsqueeze(0)
        target_std = ((self.s[target].mean(dim=(0,2,3)))/self.n[target]).sqrt()
        gamma_convert = self.gamma_res1(gamma, target_mean, target_std, target)
        gamma_convert = self.gamma_res2(gamma_convert, target_mean, target_std, target)
        beta_convert = self.gamma_res1(beta, target_mean, target_std, target)
        beta_convert = self.gamma_res2(beta_convert, target_mean, target_std, target)
        
        return gamma_convert, beta_convert
    
    def update(self, feature, target):
        self.n[target] += 1
        if self.n[target] == 1:
            self.m[target] = feature
            self.s[target] = (feature - self.m[target].mean(dim=(0,2,3), keepdim=True)) ** 2
        else:
            prev_m = self.m[target].mean(dim=(0,2,3), keepdim=True)  # 1 x C x 1 x 1
            self.m[target] += self.w * (feature - self.m[target]) / self.n[target]  # B x C x H x W
            curr_m = self.m[target].mean(dim=(0,2,3), keepdim=True)  # 1 x C x 1 x 1
            self.s[target] += self.w * (feature - prev_m) * (feature - curr_m)  # B x C x H x W


class Class_Alignment_Discriminator(nn.Module):
    def __init__(self, n_domain, n_class, channels=2048):
        super().__init__()
        self.n_domain = n_domain
        self.n_class = n_class
        self.shared = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 256, kernel_size=1, stride=1, padding=0)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.domain_dis_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.domain_dis_fc = nn.Sequential(
            nn.Linear(512*8*16, 500),
            nn.ReLU(),
            nn.Linear(500, n_domain)
        )
        self.class_dis_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.class_dis_fc = nn.Sequential(
            nn.Linear(512*8*16, 500),
            nn.ReLU(),
            nn.Linear(500, n_class+1)
        )

    def forward(self, feature, pred):
        f = self.shared(feature)  # B x 64 x H x W
        domain_out = self.domain_dis_conv(f)  # B x 512 x 8 x 16
        domain_out = self.domain_dis_fc(domain_out.view(f.size(0), -1))
        class_out = []
        for cls in range(self.n_class):
            class_attention_map = f * pred[:, cls, :, :].unsqueeze(1)  # B x 64 x H x W
            class_out_ = self.class_dis_conv(class_attention_map)
            class_out.append(self.class_dis_fc(class_out_.view(f.size(0), -1)))
        return domain_out, class_out


class Cross_Domain_Class_Alignment(nn.Module):
    def __init__(self, converts, n_class):
        super().__init__()
        self.n_class = n_class
        self.centroid = dict()
        self.n_sample = dict()
        for convert in converts:
            target = convert[-1]
            self.centroid[target] = 0
            self.n_sample[target] = 0
            self.centroid[convert] = 0
            self.n_sample[convert] = 0

    def forward(self, feature_s2t, feature_target, seg_s2t, seg_target, convert, get_loss=False):
        target = convert[-1]
        # B x cls x 2048 x H x W    
        feature_s2t_cls = feature_s2t.unsqueeze(1).expand(-1, self.centroid[convert].size(0), -1, -1, -1)
        feature_target_cls = feature_target.unsqueeze(1).expand(-1, self.centroid[target].size(0), -1, -1, -1)
        centroid_s2t = self.centroid[convert].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        centroid_target = self.centroid[target].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        centroid_s2t = centroid_s2t.expand_as(feature_s2t_cls)
        centroid_target = centroid_target.expand_as(feature_target_cls)
        # B x cls x H x W
        # distance_target_target = torch.norm(feature_target_cls-centroid_target, p=2, dim=2)
        # B x H x W
        mask_s2t_target = torch.argmin(torch.norm(feature_s2t_cls-centroid_target, p=2, dim=2), dim=1)
        mask_target_s2t= torch.argmin(torch.norm(feature_target_cls-centroid_s2t, p=2, dim=2), dim=1)
        mask_s2t_target = F.interpolate(mask_s2t_target.float().unsqueeze(1), size=seg_s2t.size()[1:], mode='nearest').squeeze(1).long()
        mask_target_s2t = F.interpolate(mask_target_s2t.float().unsqueeze(1), size=seg_target.size()[1:], mode='nearest').squeeze(1).long()

        # if get_loss:
        #     # pred_target_s2t = F.softmax(distance_target_s2t, dim=1)
        #     x = F.interpolate(distance_s2t_target, size=seg_s2t.size()[1:], mode='bilinear', align_corners=True)
        #     self.loss = self.CrossEntropy2d(x, seg_s2t)
           
        return mask_s2t_target, mask_target_s2t

    def region_wise_pooling(self, codes, seg):

        segmap = F.one_hot(seg+1, num_classes=self.n_class+1).permute(0,3,1,2)
        segmap = F.interpolate(segmap.float(), size=codes.size()[2:], mode='nearest')

        b_size = codes.shape[0]
        # h_size = codes.shape[2]
        # w_size = codes.shape[3]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

        return codes_vector.mean(dim=0)[1:]

    def update(self, feature, seg, domain):
        self.n_sample[domain] += 1
        new_centroid = self.region_wise_pooling(feature, seg)  # cls x 2048
        if self.n_sample[domain] == 1:
            self.centroid[domain] = new_centroid
        else:
            self.centroid[domain] += ((new_centroid - self.centroid[domain]) / self.n_sample[domain])

    def feature_align_loss(self, feature, seg, original_seg, domain):
        loss = 0.
        # f_norm = F.normalize(feature, p=2, dim=(1, 2))  # B x H x W
        # f_norm = feature
        # update center
        seg_ = F.interpolate(seg.detach().unsqueeze(1).float(), size=feature.shape[2:], mode='nearest').squeeze(1)
        # ignore label (repulsive force)
        hole = original_seg - seg
        hole_ = F.interpolate(hole.detach().unsqueeze(1).float(), size=feature.shape[2:], mode='nearest').squeeze(1)
        mask = [] 
        mask_hole = []
        for i in range(self.n_class):
            mask_hole.append((1*(hole_==i).unsqueeze(1)))
            mask.append((1*(seg_==i).unsqueeze(1)))  # B x 1 x H x W
        
        pos = torch.zeros(self.n_class, device=torch.device('cuda'))
        neg = torch.zeros(self.n_class, device=torch.device('cuda'))
        # TODO : only pos contrastive loss
        for cls in range(self.n_class):
            if mask[cls].sum() != 0:
                centroid = self.centroid[domain][cls].unsqueeze(1).unsqueeze(1).expand(-1, 65, 129)
                f_cls_differ = torch.norm(mask[cls] *(feature - centroid), p=2, dim=(1, 2, 3))  # B x H x W
                pos[cls] = f_cls_differ
                # loss = loss + f_cls_differ.mean()
            if mask_hole[cls].sum() != 0:
                centroid = self.centroid[domain][cls].unsqueeze(1).unsqueeze(1).expand(-1, 65, 129)
                f_cls_differ = torch.norm(mask_hole[cls] * (feature - centroid), p=2, dim=(1, 2, 3))  # B x H x W
                neg[cls] = f_cls_differ
                # print('cls', cls,':', float(f_cls_differ.mean()))
                # loss = loss + math.pow((margin - f_cls_differ.mean()), 2)

        margin = 1/self.n_class
        lambda_neg = 0.2


        pos = self.softmax(pos)
        neg = torch.pow((margin - self.softmax(neg).mean()), 2)
        loss = loss + pos.mean()
        loss = loss + lambda_neg * neg
                    
        return loss

    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        
        return loss

class Prediction_Discriminator(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(n_class, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)),
        )
    
    def forward(self, x):
        return self.conv(x)





