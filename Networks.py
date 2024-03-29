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
from SWAE.upfirdn2d import *
from SWAE.fused_act import *
from collections import OrderedDict


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

class SPADE(nn.Module):
    def __init__(self, nc_norm, n_class, nc_embed=50, kernel_size=3, nc_label=1): #input 채널 갯수, label 채널 갯수
        super().__init__()
        # self.normalize = norm_layer(nc_norm)
        # self.embed = nn.Conv2d(nc_label, nc_embed, kernel_size=1, stride=1)
        self.n_class = n_class
        self.nc_output = nc_norm
        self.embed = nn.Embedding(n_class+1, nc_embed, padding_idx=0)
        
        # self.mlp_shared = nn.Sequential(
        #     nn.Conv2d(nc_embed, nc_embed, kernel_size=kernel_size, padding=1),
        #     nn.ReLU()
        # )
        # self.gamma = nn.Conv2d(nc_embed, nc_norm, kernel_size=kernel_size, padding=1)
        # self.beta = nn.Conv2d(nc_embed, nc_norm, kernel_size=kernel_size, padding=1)

        # input : (b * 1 * h * w)
        self.nc_output = nc_norm
        self.FC = nn.Sequential(
            nn.Linear(nc_embed, nc_embed),
            nn.ReLU(),
            nn.Linear(nc_embed, nc_embed),
            nn.ReLU(),
        )
        self.FC_gamma = nn.Sequential(
            nn.Linear(nc_embed, self.nc_output),
            nn.ReLU(),
        )
        self.FC_betta = nn.Sequential(
            nn.Linear(nc_embed, self.nc_output),
            nn.ReLU(),
        )

    def forward(self, feature, label): #input, segmentation map(input이랑 같은 사이즈)
        
        # normalized_input = self.normalize(input)
        segmap = label.detach()
        segmap = F.interpolate(segmap.unsqueeze(1).float(), size=feature.size()[2:], mode='nearest')
        segmap = segmap.squeeze(1).long()
        # if len(segmap.size())==2:
        #     segmap = segmap.unsqueeze(0)
        embeded = self.embed(segmap + 1)   # b, h, w, c
        b, h, w, c = embeded.shape
        embeded = embeded.contiguous()
        inputs = embeded.view([b*h*w, c])
        
        actv = self.FC(embeded)

        gamma = self.FC_gamma(actv)
        betta = self.FC_betta(actv)
        gamma_ = gamma.view([b, h, w, self.nc_output]).permute(0,3,1,2)
        betta_ = betta.view([b, h, w, self.nc_output]).permute(0,3,1,2)

        out = feature*gamma_ + betta_
        return out

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


class Separator(nn.Module):
    def __init__(self, imsize, dsets, ch=64):
        super(Separator, self).__init__()
        self.Conv = nn.Sequential(
            # batch_size x 256 x 4 x 4
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
        )
        self.w = nn.ParameterDict()
        h, w = imsize
        for dset in dsets:
            self.w[dset] = nn.Parameter(torch.ones(1, ch, h, w), requires_grad=True)

    def forward(self, features, converts=None):
        contents, styles, S_F = dict(), dict(), dict()  # S_F = S(F_X), see equation (2) in our paper.
        for key in features.keys():
            S_F[key] = self.Conv(features[key])
            if '2' in key:  # to disentangle the features of converted images. (and to compute consistency loss)
                source, target = key.split('2')
                contents[key] = self.w[target] * S_F[key]
                styles[key] = features[key] - contents[key]
            else:  # to disentangle the features of input images.
                contents[key] = self.w[key] * S_F[key]
                styles[key] = features[key] - contents[key]
        if converts is not None:  # for generating converted features.
            for cv in converts:
                source, target = cv.split('2')
                contents[cv] = self.w[target] * S_F[source]
        return contents, styles


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


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.Decoder_Conv = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.ReLU(True),
#             ResidualBlock(32, 32),
#             ResidualBlock(32, 32),
#             nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.Tanh()
#         )
#     def forward(self, c, s):
#         return self.Decoder_Conv(c+s)


# class Domain_Normalization_Parameter(nn.Module):
#     def __init__(self, datasets, h, w):
#         super(Domain_Normalization_Parameter, self).__init__()
#         self.w = nn.ParameterDict()
#         for dset in datasets:
#             self.w[dset] = nn.Parameter(torch.ones(1, 64, h//2, w//2), requires_grad=True)
    
#     def forward(self, x, domain):
#         return self.w[domain] * x

class Classifier(nn.Module):
    def __init__(self, channels=1, num_classes=10):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            # batch_size x source_channels x 512 x 1024
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            # batch_size x 32 x 512 x 1024
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # batch_size x 32 x 256 x 512
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # batch_size x 64 x 256 x 512
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # batch_size x 64 x 128 x 256
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # batch_size x 128 x 128 x 256
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # batch_size x 128 x 64 x 128
        )

        self.fc = nn.Sequential(
            # batch_size x (128*64*128 = 1048576)
            nn.Linear(1048576, 1000),
            nn.ReLU(True),
            # batch_size x 100
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            # batch_size x 100
            nn.Linear(1000, 1),
            # nn.Sigmoid(),
        )
        self.sig = nn.Sigmoid()

    def forward(self, inputs):
        inputs_ = inputs.unsqueeze(1)
        output = self.conv(inputs_)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.sig(output)
        return output

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

class Residual_PatchGAN_Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Residual_PatchGAN_Discriminator, self).__init__()
        self.Conv = nn.Sequential(
            # batch_size x 32 x 640 x 384
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),            
            spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, inputs):
        patch_output = self.Conv(inputs)
        return patch_output

class Gram_Discriminator(nn.Module):
    def __init__(self, channels=4):
        super(Gram_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Sequential(
            nn.Linear(64*64, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
        )

    def forward(self, parts):
        b, c, h, w = parts.size()
        out = self.conv1(parts)
        out = gram(out)  # B x 64 x 64
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

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


class Deep_Multi_Head_Discriminator(nn.Module):
    def __init__(self, num_domains, channels=3):
        super(Deep_Multi_Head_Discriminator, self).__init__()
        self.Conv = nn.Sequential(
            # input size: 256x256
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),  # 128x128
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)),  # 128x128
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),  # 64x64
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)),  # 64x64
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.Patch = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*64*128, 500),
            nn.ReLU(),
            nn.Linear(500, num_domains),
        )

    def forward(self, inputs):
        conv_output = self.Conv(inputs)
        patch_output = self.Patch(conv_output)
        fc_output = self.fc(conv_output.view(conv_output.size(0), -1))
        return (patch_output, fc_output)


class Multi_Domain_Discriminator(nn.Module):
    def __init__(self, num_domains, channels=3):
        super(Multi_Domain_Discriminator, self).__init__()
        self.Conv = nn.Sequential(
            # input size: 256x256
            spectral_norm(nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)), # 
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, num_domains-1, kernel_size=3, stride=1, padding=1, bias=True)),
        )

    def forward(self, inputs):
        return self.Conv(inputs)

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

class Perceptual_Discriminator(nn.Module):
    def __init__(self):
        super(Perceptual_Discriminator, self).__init__()
        # self.n_class = n_class
        self.vgg = VGG16()
        # CGL block
        self.CGL = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # CLC layer
        self.CLC = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=True)),
        )
        # Embedding
        # self.embedding = nn.Conv2d(n_class+1, 256, kernel_size=1)

        # self.fc = nn.Sequential(
        #     spectral_norm(nn.Linear(64*64*256, 500)),
        #     nn.ReLU(True),
        #     spectral_norm(nn.Linear(500, n_domain))
        # )

    def forward(self, img, extract_features=False):
        feature_vgg = self.vgg(img)
        feature_CGL = self.CGL(feature_vgg)  # B x 256 x 64 x 64 
        feature_CLC = self.CLC(feature_CGL)
        # segmap = F.one_hot(seg.squeeze(1) + 1, num_classes=self.n_class+1).permute(0,3,1,2)
        # segmap = F.interpolate(segmap.float(), size=feature_CGL.shape[2:], mode='nearest')
        # segmap = self.embedding(segmap)
        # patch_out = feature_CLC + segmap * feature_CGL
        patch_out = feature_CLC
        # fc_out = self.fc(feature_vgg.view(feature_vgg.size(0), -1))
        if extract_features:
            return patch_out, feature_vgg
        else:
            return patch_out


class Domain_Normalization(nn.Module):
    def __init__(self, datasets):
        super().__init__()
        self.n = dict()
        self.m = dict()
        self.s = dict()
        self.mlp_mean = nn.ModuleDict()
        self.mlp_std = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        
        for dset in datasets:
            self.n[dset] = 0
            self.m[dset] = 0
            self.s[dset] = 0
            self.w = 1
            self.mlp_mean[dset] = nn.Sequential(
                nn.Linear(64, 64),
            )
            self.mlp_std[dset] = nn.Sequential(
                nn.Linear(64, 64),
            )
            self.conv[dset] = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(1, 1, kernel_size=1, stride=1),
            )
        
    def forward(self, fs, ft, source, target, update=False):
        # Domain mean, std
        if update:
            self.n[target] += 1
            if self.n[target] == 1:
                self.m[target] = ft
                self.s[target] = (ft - self.m[target].mean(dim=(0,2,3), keepdim=True)) ** 2
            else:
                prev_m = self.m[target].mean(dim=(0,2,3), keepdim=True)  # 1 x C x 1 x 1
                self.m[target] += self.w * (ft - self.m[target]) / self.n[target]  # B x C x H x W
                curr_m = self.m[target].mean(dim=(0,2,3), keepdim=True)  # 1 x C x 1 x 1
                self.s[target] += self.w * (ft - prev_m) * (ft - curr_m)  # B x C x H x W
        else:
            beta = self.mlp_mean[target](self.m[target].mean(dim=(0,2,3)).unsqueeze(0))  # 1 x C
            gamma = self.mlp_std[target](((self.s[target].mean(dim=(0,2,3)))/self.n[target]).sqrt().unsqueeze(0))  # 1 x C

            # f_position_mean = fs.mean(dim=1, keepdim=True)  # B x 1 x H x W
            # f_position_std = fs.std(dim=1, keepdim=True) # B x 1 x H x W
            f_position = torch.cat([fs.mean(dim=1, keepdim=True), fs.std(dim=1, keepdim=True)], dim=1)  # B x 2 x H x W
            f_position = self.conv[source](f_position).permute(0,2,3,1).contiguous() # B x H x W x 1
            f_position = f_position.matmul(gamma) + beta.unsqueeze(0).unsqueeze(0)  # B x H x W x C
            
            return f_position.permute(0,3,1,2).contiguous()


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



class Adaptive_Style_Memory_Bank(nn.Module):
    def __init__(self, targets, size=64):
        super().__init__()
        self.content = dict()
        self.style = dict()
        self.n_target = len(targets)
        self.size = size
    
    def forward(self, content, target):
        content_source = F.normalize(content.view(content.size(0), -1), dim=1)  # B1 x N
        content_target = F.normalize(self.content[target].view(-1, self.content[target].size(0)), dim=0)  # N x B2
        sim_mat = content_source.mm(content_target)  # B1 x B2
        adaptive_style = self.style[target][sim_mat.argmax(dim=1)]
        if self.content[target].size(0) >= self.size:
            self.content[target] = self.content[target][content.size(0):]
            self.style[target] = self.style[target][content.size(0):]
            # remove_index = torch.unique(sim_mat.argmin(dim=1))
            # if remove_index.size(0) == content_source.size(0):
                
        return adaptive_style

    def update(self, content, style, target):
        if len(self.content) < self.n_target:
            self.content[target] = content
            self.style[target] = style
        else:
            self.content[target] = torch.cat([self.content[target], content], dim=0)
            self.style[target] = torch.cat([self.style[target], style], dim=0)


class Content_Style_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.Content = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        self.Style = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(),
        )

        self.add_module(
            "ToGlobalCode",
            nn.Sequential(
                EqualLinear(512, 512)
            )
        )

    def forward(self, x, extract_features=False):
        c = self.Content(x)
        s = self.Style(x)
        s = s.mean(dim=(2, 3))
        s = self.ToGlobalCode(s)

        return c, s


class AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim=512):
        super().__init__()
        self.IN = nn.InstanceNorm2d(in_channel)
        self.gamma = nn.Linear(style_dim, in_channel)
        self.beta = nn.Linear(style_dim, in_channel)

    def forward(self, content, style):
        return self.gamma(style).unsqueeze(-1).unsqueeze(-1) * self.IN(content) + self.beta(style).unsqueeze(-1).unsqueeze(-1)


class SMResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim=512):
        super().__init__()
        self.learned_shortcut = False
        mid_channel = min(in_channel, out_channel)
        self.adain1 = AdaIN(in_channel, style_dim=style_dim)
        self.conv1 = spectral_norm(nn.Conv2d(in_channel, mid_channel, 3, 1, 1))
        self.adain2 = AdaIN(mid_channel, style_dim=style_dim)
        self.conv2 = spectral_norm(nn.Conv2d(mid_channel, out_channel, 3, 1, 1))
        self.shortcut = nn.Sequential()
        if in_channel != out_channel:
            self.learned_shortcut = True
            self.conv_s = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False))
            self.adain_s = AdaIN(in_channel, style_dim=style_dim)
            

    def shortcut(self, c, s):
        if self.learned_shortcut:
            x_s = self.conv_s(F.relu(self.adain_s(c, s)))
        else:
            x_s = c
        return x_s

    def forward(self, c, s):
        x = self.shortcut(c, s)
        dx = self.conv1(F.relu(self.adain1(c, s)))
        dx = self.conv2(F.relu(self.adain2(dx, s)))
        return x + dx

class Content_Style_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.block1 = SMResBlock(256, 256)
        # self.block2 = SMResBlock(256, 128)
        # self.block3 = nn.Sequential(
        #     spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1)),
        #     nn.ReLU(),
        # )
        self.adain = AdaIN(64)
        self.block1 = SMResBlock(64, 64)
        self.block2 = SMResBlock(64, 64)

    def forward(self, c, s):
        # s: B x 512, c: B x 256 x 128 x 256
        x = self.adain(c, s)
        # x = self.block1(c, s)
        # x = self.block2(x, s)
        # x = self.block3(x)
        # x = self.block4(x, s)
        # x = self.block5(x, s)
        return x


