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


def normalize(x):
    image_transforms = ttransforms.Compose([
        ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    for b in range(x.size(0)):
        x[b] = image_transforms(x[b])
    return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
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
        output = self.Encoder_Conv(inputs) # batch_size x 512 x 10 x 6
        return output

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

class Style_Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Style_Encoder, self).__init__()
        bin = functools.partial(nn.GroupNorm, 4)
        self.Encoder_Conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            bin(32),
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
        gamma = self.gamma(output)
        beta = self.beta(output)
        return gamma, beta

class Content_Style_Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Style_Encoder, self).__init__()
        self.C = nn.Sequential(
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
        gamma = self.gamma(output)
        beta = self.beta(output)
        return gamma, beta

class Content_Extractor(nn.Module):
    def __init__(self, ch=64):
        super(Content_Extractor, self).__init__()
        # self.Conv = nn.Sequential(
        #     spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.ReLU(True),
        #     spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.ReLU(True),
        # )
        self.Conv = nn.Sequential(
            ResidualBlock(ch, ch),
            ResidualBlock(ch, ch),
            ResidualBlock(ch, ch),
            ResidualBlock(ch, ch),
        )

    def forward(self, inputs):
        return self.Conv(inputs)

class Content_Generator(nn.Module):
    def __init__(self):
        super(Content_Generator, self).__init__()
        # input: HxW = 65x129
        self.Decoder_Conv = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=True)), # 130x258
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True)), # 130x258
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)), # 260x516
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)), # 260x516
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)), # 260x516
            nn.ReLU(True),
        )
        # self.Decoder_Conv = nn.Sequential(
        #     spectral_norm(nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=True)), # 130x258
        #     nn.ReLU(True),
        #     spectral_norm(nn.ConvTranspose2d(512, 64, kernel_size=4, stride=2, padding=1, bias=True)), # 260x516
        #     nn.ReLU(True),
        #     ResidualBlock(64, 64),
        #     ResidualBlock(64, 64),
        # )
    def forward(self, x):
        return self.Decoder_Conv(x)

class Content_Encoder(nn.Module):
    def __init__(self, n_class, channels=3):
        super(Content_Encoder, self).__init__()
        self.n_class = n_class
        self.conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.InstanceNorm2d(64, affine=False)
        )
    
        self.embedding = nn.Conv2d(n_class+1, 64, kernel_size=1)

    def forward(self, seg):
        segmap = F.one_hot(seg.squeeze(1) + 1, num_classes=self.n_class+1).permute(0,3,1,2)
        segmap = self.embedding(segmap.float())
        return self.conv(segmap)

class Class_wise_Separator(nn.Module):
    def __init__(self, source, target, n_class, ch=64):
        super(Class_wise_Separator, self).__init__()
        self.Conv = nn.Sequential(
            # batch_size x 64 x 4 x 4
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            # nn.InstanceNorm2d(ch, affine=False),
        )
        self.n_class = n_class
        # self.st_matrix = dict()
        self.wc = dict()
        self.mem = Memory_Network(n_class, source, target)
        self.source, self.target = source, target
        for dset in [source, target]:
            self.wc[dset] = SPADE(ch, n_class).cuda()
            # self.st_matrix[dset] = dict()
            # for cl in range(n_class):
            #     self.st_matrix[dset][str(cl)] = dict()

    def forward(self, features, seg, converts=None):
        contents, styles = dict(), dict()
        for key in features.keys():
            if '2' in key:
                source, target = key.split('2')
                contents[key] = self.Conv(features[key])
                styles[key] = features[key] - self.wc[target](contents[key], seg[source])
                styles[source] = self.mem(contents[key], seg[source], target)
                contents[source] = self.wc[source](contents[key], seg[source])
            else:
                contents[key] = self.Conv(features[key])
                styles[key] = features[key] - self.wc[key](contents[key], seg[key])
        
        if converts is not None:
            for cv in converts:
                source, target = cv.split('2')
                styles[cv] = self.mem(contents[source], seg[source], source)
                contents[cv] = self.wc[target](contents[source], seg[source])
            for key in features.keys():
                contents[key] = self.wc[source](contents[key], seg[source])
        return contents, styles
    
    # def update_style_matrix(self, styles):
    #     style_vectors = dict()
    #     for key in styles.keys():
    #         # batch x n_class x channel
    #         style_vectors[key] = self.region_wise_pooling(styles[key])
    #         for cl in range(self.n_class):
    #             if len(self.st_matrix[key][str(cl)]) == 0:
    #                 # batch x channel
    #                 self.st_matrix[key][str(cl)]['mean'] = style_vectors[key][:, n_class, :]
    #                 self.st_matrix[key][str(cl)]['sd'] = torch.zeros_like(style_vectors[key][:, n_class, :])
    #                 self.st_matrix[key][str(cl)]['sample'] = 1
    #             else:
    #                 m = 

    # def region_wise_pooling(self, codes, segmap):
    #     segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

    #     b_size = codes.shape[0]
    #     # h_size = codes.shape[2]
    #     # w_size = codes.shape[3]
    #     f_size = codes.shape[1]

    #     s_size = segmap.shape[1]

    #     codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

    #     for i in range(b_size):
    #         for j in range(s_size):
    #             component_mask_area = torch.sum(segmap.bool()[i, j])

    #             if component_mask_area > 0:
    #                 codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
    #                 codes_vector[i][j] = codes_component_feature

    #     return codes_vector

    # def style_sampling(self, seg, ):

"""TODO : build module that extract gamma, beta tensor from image and segmap
    INPUT : feature dictionary, segmap dictionary 
        -> e.g. feature[src] = F_x, segmap[tgt] = seg_y
    OUTPUT : Affine tensor gamma beta dictionary
    1. REGION WISE POOLING -> Matrix1 : B*cls*C 
    2. GLOBAL POOLING -> Matrix2 : B*1*C
    3. concat Mat1, Mat2 : B*(cls+1)*C
    4. MLP B*(cls+1)*C -> B*(cls+1)*C'
    5. broadcasting : B*(cls+1)*C' -> B*C'*H*W
    5. Multi-head convolution (like SPADE) : gamma = conv1(F), beta = conv2(F)
    """
class StyleEncoder(nn.Module):
    def __init__(self, n_class, datasets, converts, channel=64):
        super(StyleEncoder, self).__init__()
        self.n_class = n_class
        self.channel = channel
        self.converts = converts
        self.datasets = datasets
        self.FC = dict()
        for cls in range(n_class + 1):
            self.FC[str(cls)] = nn.Sequential(
                nn.Linear(channel, channel),
                nn.ReLU(True),
            ).cuda()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channel, channel, 3,1,1),
            nn.ReLU(True),
        )
        self.conv_beta= nn.Sequential(
            nn.Conv2d(channel, channel, 3,1,1),
            nn.ReLU(True),
        )

    def forward(self, feature_dict, segmap_dict):
        mat, gamma, beta = dict(), dict(), dict()
        segmap_dict_ = dict()
        b, c, h, w = feature_dict[self.datasets[0]].shape
        for dataset in self.datasets:
            segmap_dict_[dataset] = F.one_hot(segmap_dict[dataset] + 1, num_classes=self.n_class+1).permute(0,3,1,2)
            segmap_dict_[dataset] = F.interpolate(segmap_dict_[dataset].float(), size=(h,w), mode='nearest')
            rpmat = self.region_wise_pooling(feature_dict[dataset], segmap_dict_[dataset])
            gpmat = feature_dict[dataset].mean(dim=(2,3)).unsqueeze(1)
            mat[dataset] = torch.cat((rpmat,gpmat), dim=1)  # b * cls * n
            # shape : b*(cls+1)*c
            # n = m1.shape[1] # == cls+1
            # m1 = m1.view(b*n, c)
            # mat[dataset] = self.FC(m1).view(b,n,c)
            # broadcasting
            f = self.broadcasting(segmap_dict_[dataset], mat[dataset])
            # output shape : b*c*h*w
            gamma[dataset] = self.conv_gamma(f)
            beta[dataset] = self.conv_beta(f)

        for convert in self.converts:
            # [G2C, C2G]
            source, target = convert.split('2') # G C
            f = self.broadcasting(segmap_dict_[source], mat[target])

            gamma[convert] = self.conv_gamma(f)
            beta[convert] = self.conv_beta(f)

        return gamma, beta

    def broadcasting(self, segmap, style_code):
        b, cls, h, w = segmap.shape
        c = style_code.shape[-1]
        middle_avg = torch.zeros(b, c, h, w).cuda()
        for i in range(b):
            for j in range(cls):
                mask = segmap.bool()[i, j]
                component_mask_area = torch.sum(mask)

                if component_mask_area > 0:
                    middle_mu = style_code[i][j] # c
                    if middle_mu.sum() != 0 and j != 0:
                        middle_mu = self.FC[str(j)](middle_mu)
                        component_mu = middle_mu.reshape(c, 1).expand(c, component_mask_area)
                        middle_avg[i].masked_scatter_(mask, component_mu) 
                    else:
                        middle_mu = style_code[i][-1]
                        middle_mu = self.FC[str(0)](middle_mu)
                        component_mu = middle_mu.reshape(c, 1).expand(c, component_mask_area)

                        middle_avg[i].masked_scatter_(mask, component_mu) 

        return middle_avg


    def region_wise_pooling(self, codes, segmap):
        

        b_size = codes.shape[0]
        # h_size = codes.shape[2]
        # w_size = codes.shape[3]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]
        """segmap instance """

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

        # shape : b*cls*c
        return codes_vector
'''
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
'''

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Decoder_Conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )
    def forward(self, c, s):
        return self.Decoder_Conv(c+s)

class CST(nn.Module):
    def __init__(self, targets, n_class):
        super(CST, self).__init__()
        self.embed = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        self.R = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),
        )
        for target in targets:
            self.embed[target] = nn.Embedding(n_class+1, 64)
            self.conv[target] = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)
            
    def forward(self, style, seg, target):
        # segmap = F.interpolate((seg+1).unsqueeze(1).float(), size=style.shape[2:], mode='nearest').squeeze(1)
        # segmap = self.embed[target](segmap.long()).permute(0,3,1,2)
        # return self.R(self.conv[target](segmap) * style)
        return self.R(style)

class Domain_Normalization_Parameter(nn.Module):
    def __init__(self, datasets, h, w):
        super(Domain_Normalization_Parameter, self).__init__()
        self.w = nn.ParameterDict()
        for dset in datasets:
            self.w[dset] = nn.Parameter(torch.ones(1, 64, h//2, w//2), requires_grad=True)
    
    def forward(self, x, domain):
        return self.w[domain] * x

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
        self.Conv = nn.Sequential(
            # batch_size x 32 x 640 x 384
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),            
            spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
            spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=True)),  # 128x128
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),  # 64x64
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.Patch = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
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

class Memory_Network(nn.Module):
    def __init__(self, n_class, source, target, channel=64, n=20):
        super(Memory_Network, self).__init__()
        self.n_class = n_class
        self.source = source
        self.target = target
        self.mem = nn.ParameterDict()
        self.mem['key'] = nn.Parameter(torch.ones(n_class + 1, channel, n), requires_grad=True)
        self.mem[source] = nn.Parameter(torch.ones(n_class + 1, n, channel), requires_grad=True) # value X
        self.mem[target] = nn.Parameter(torch.ones(n_class + 1, n, channel), requires_grad=True) # value Y
        for m in self.mem.values():
            init.xavier_normal_(m)

    def forward(self, content, seg, domain):
        b, c, h, w = content.size()
        seg_ = F.interpolate(seg.unsqueeze(1).float(), size=(h,w), mode='nearest')
        seg_ = F.one_hot(seg_.squeeze(1).long()+1, num_classes=self.n_class+1)  # b x h x w x class
        seg_ = seg_.view(-1, self.n_class+1).float() # (bxhxw) x class
        content_ = content.permute(0,2,3,1).contiguous().view(-1, c) # (bxhxw) x c
        content_ = torch.bmm(content_.unsqueeze(-1), seg_.unsqueeze(1)) # (bxhxw) x c x class
        content_ = content_.view(b, h*w, c, self.n_class+1).permute(0,3,1,2).contiguous().view(b*(self.n_class+1),h*w,c) # (bxclass) x (hxw) x c
        key = self.mem['key'].unsqueeze(0).repeat(b,1,1,1).view(b*(self.n_class+1), c, -1) # (b x class) x c x n
        content_ = F.normalize(content_, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=1) 
        weight = torch.bmm(content_, key) # consine similarity (b x class) x (hxw) x n
        weight = F.softmax(weight, dim=-1)
        value = self.mem[domain].unsqueeze(0).repeat(b,1,1,1).view(b*(self.n_class+1), -1, c) # (b x class) x n x c
        adaptive_style = torch.bmm(weight, value) # (b x class) x (h x w) x c
        adaptive_style = adaptive_style.view(b, self.n_class+1, h, w, c) # b x class x h x w x c
        adaptive_style = torch.sum(adaptive_style, dim=1) # b x h x w x c
        return adaptive_style.permute(0, 3, 1, 2)

# CC-FPSE : Feature-pyramid Semantics Embedding Discriminator
class FPSE_Discriminator(nn.Module):
    def __init__(self, n_class, channels=3):
        embed_dim = 128
        super(FPSE_Discriminator, self).__init__()
        self.n_class = n_class
        # self.embed  = nn.Conv2d(3, embed_dim, kernel_size=1, stride=1)
        self.embed = nn.Embedding(n_class+1, embed_dim, padding_idx=0)
        
        # nf = 64
        # C64-C128-C256-C512-C512
        self.enc1 = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)            # batch_size x 256 x 80 x 48
        )
        self.enc4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # U-net : lateral connections
        self.lat2 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.lat3 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 256, kernel_size=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.lat4 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 256, kernel_size=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.lat5 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 256, kernel_size=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # final layers
        self.final2 = nn.Sequential(
                    spectral_norm(nn.Conv2d(256, 128, kernel_size=3, padding=1)), 
                    nn.LeakyReLU(0.2, True)
        )
        self.final3 = nn.Sequential(
                    spectral_norm(nn.Conv2d(256, 128, kernel_size=3, padding=1)), 
                    nn.LeakyReLU(0.2, True)
        )
        self.final4 = nn.Sequential(
                    spectral_norm(nn.Conv2d(256, 128, kernel_size=3, padding=1)), 
                    nn.LeakyReLU(0.2, True)
        )
        
        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(128, 1, kernel_size=1)
        self.seg = nn.Conv2d(128, 128, kernel_size=1)

        # TO DO : Conv 안에 layer 연결하고 forward implementation
        # embedding parameter 조정

    def forward(self, img, label):

        x1 = self.enc1(img)     # img : fake or real image
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        # U-net
        b5 = self.lat5(x5)
        b4 = self.lat4(x4) + self.up(b5)
        b3 = self.lat3(x3) + self.up(b4)
        b2 = self.lat2(x2) + self.up(b3)

        # final prediction layers
        f2 = self.final2(b2)
        f3 = self.final3(b3)
        f4 = self.final4(b4)

        pred2 = self.tf(f2)
        pred3 = self.tf(f3)
        pred4 = self.tf(f4)
        seg2 = self.seg(f2)
        seg3 = self.seg(f3)
        seg4 = self.seg(f4)
        
        # intermediate feature matching loss
        # feats = [x2, x3, x4, x5]
        # Loss_fm is used for generator

        # segmentation map embedding
        # segmap = label.detach()
        # segmap = F.interpolate(segmap.unsqueeze(1).float(), size=feature.size()[2:], mode='nearest')
        # segmap = segmap.squeeze(1).long()
        embed = self.embed(label+1)
        b, h, w, c = embed.shape
        embed = embed.contiguous().view(b,c,h,w)
        segemb = F.avg_pool2d(embed, kernel_size=2, stride=2)
        segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)
        segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)

        # semantics embedding discriminatro score
        pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)
        # print(pred2.shape, pred3.shape, pred4.shape)

        # pred3 = torch.reshape(pred3, (1,1,pred2.shape[2], int(pred2.shape[2]/4)))
        # pred4 = torch.reshape(pred4, (1,1,pred2.shape[2], int(pred2.shape[2]/16)))
        
        # print(pred3.shape, pred4.shape)
        #concat results from multiple resolutions
        # results = torch.cat((pred2, pred3, pred4), dim=3)

        return (pred2, pred3, pred4)

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
    def __init__(self, n_domain, n_class):
        super(Perceptual_Discriminator, self).__init__()
        self.n_class = n_class
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
        self.embedding = nn.Conv2d(n_class+1, 256, kernel_size=1)

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(16*16*256, 500)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(500, n_domain))
        )

    def forward(self, patches, seg, feature_vgg):
        feature_CGL = self.CGL(feature_vgg)  # B x 256 x 64 x 64 
        feature_CLC = self.CLC(feature_CGL)
        # segmap = F.one_hot(seg.squeeze(1) + 1, num_classes=self.n_class+1).permute(0,3,1,2)
        # segmap = F.interpolate(segmap.float(), size=feature_CGL.shape[2:], mode='nearest')
        # segmap = self.embedding(segmap)
        # patch_out = feature_CLC + segmap * feature_CGL
        patch_out = feature_CLC
        fc_out = self.fc(feature_vgg.view(feature_vgg.size(0), -1))

        return (patch_out, fc_out)

## custom Residualblock
''' 창재가 짠거
    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=3, downsample=False):
            super().__init__()

            self.conv1 = Conv2d(in_channels= in_ch, out_channels=in_ch, kernel_size=kernel, stride=1, padding=1)
            if downsample:
                self.conv2 = Conv2d(in_channels= in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1)
                self.skip = Conv2d(in_channels = in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1)
            else :
                self.conv2 = Conv2d(in_channels= in_ch, out_channels=out_ch, kernel_size=kernel, stride=1, padding=1)
                self.skip = Conv2d(in_channels = in_ch, out_channels=out_ch, kernel_size=kernel, stride=1, padding=1)
            self.activation = nn.ReLU(True)
                
        def forward(self, input):
            out = self.activation(self.conv1(input))
            out = self.activation(self.conv2(out))
            skip = self.skip(input)
            out = (out + skip) / math.sqrt(2)

            return out

    ## Swapping Autoencoder

    class SWAEEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            n_down = 3
            n_down_st = 2
            nc_initial = 32
            nc_content = 512

            self.fromRGB = Conv2d(in_channels=3, out_channels=nc_initial, kernel_size=1)
            self.DowntoContent = nn.Sequential()
            for i in range(n_down):
                self.DowntoContent.add_module(f'ResDown{i}',
                ResBlock(in_ch= nc_initial* 2**(i), out_ch= nc_initial* 2**(i+1), downsample=True)
                )

            n_ch = nc_initial * 2**(n_down)
            self.ContentE = nn.Sequential(
                Conv2d(in_channels= n_ch, out_channels=n_ch, kernel_size=1),
                nn.ReLU(True),
                Conv2d(in_channels= n_ch, out_channels=nc_content, kernel_size=1)
            )
            
            n_chin_style = nc_initial * 2**(n_down+n_down_st)
            self.StyleE = nn.Sequential()
            for i in range(n_down_st):
                self.StyleE.add_module(f'StyleDown{i}', Conv2d(in_channels= n_ch, out_channels= n_ch * 2**(i+1), kernel_size=4, stride=2, padding=1))
                self.StyleE.add_module(f'StyleDownReLU{i}', nn.ReLU(True))
            self.StyleE.add_module('adaptivePool', nn.AdaptiveAvgPool2d(1))
            self.StyleE.add_module('styleLinear', nn.Linear(n_chin_style, n_chin_style * 2))
            

        def forward(self, input):
            x = self.fromRGB(input)
            midpoint = self.DowntoContent(x)
            content = self.ContentE(midpoint) # B X 512 X W/8 X H/8
            style = self.StyleE(midpoint) # B X 2048 X 1 X 1

            return content, style
'''

# 논문 깃헙
class SWAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.netE_num_downsampling_sp = 4
        self.netE_num_downsampling_gl = 2
        self.netE_nc_steepness = 2.0
        self.netE_scale_capacity = 1.0
        self.spatial_code_ch = 8
        self.global_code_ch = 2048

        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1]# if self.opt.use_antialias else [1]

        self.add_module("FromRGB", ConvLayer(3, self.nc(0), 1))

        self.DownToSpatialCode = nn.Sequential()
        for i in range(self.netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True)
            )

        # Spatial Code refers to the Structure Code, and
        # Global Code refers to the Texture Code of the paper.
        nchannels = self.nc(self.netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",
            nn.Sequential(
                ConvLayer(nchannels, nchannels, 1, activate=True, bias=True),
                ConvLayer(nchannels, self.spatial_code_ch, kernel_size=1,
                          activate=False, bias=True)
            )
        )

        self.DownToGlobalCode = nn.Sequential()
        for i in range(self.netE_num_downsampling_gl):
            idx_from_beginning = self.netE_num_downsampling_sp + i
            self.DownToGlobalCode.add_module(
                "ConvLayerDownBy%d" % (2 ** idx_from_beginning),
                ConvLayer(self.nc(idx_from_beginning),
                          self.nc(idx_from_beginning + 1), kernel_size=3,
                          blur_kernel=[1], downsample=True, pad=0)
            )

        nchannels = self.nc(self.netE_num_downsampling_sp +
                            self.netE_num_downsampling_gl)
        self.add_module(
            "ToGlobalCode",
            nn.Sequential(
                EqualLinear(nchannels, self.global_code_ch)
            )
        )

    def nc(self, idx):
        nc = self.netE_nc_steepness ** (5 + idx)
        nc = nc * self.netE_scale_capacity
        # nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)

    def forward(self, x, extract_features=False):
        x = self.FromRGB(x)
        midpoint = self.DownToSpatialCode(x)
        sp = self.ToSpatialCode(midpoint)

        if extract_features:
            padded_midpoint = F.pad(midpoint, (1, 0, 1, 0), mode='reflect')
            feature = self.DownToGlobalCode[0](padded_midpoint)
            assert feature.size(2) == sp.size(2) // 2 and \
                feature.size(3) == sp.size(3) // 2
            feature = F.interpolate(
                feature, size=(7, 7), mode='bilinear', align_corners=False)

        x = self.DownToGlobalCode(midpoint)
        x = x.mean(dim=(2, 3))
        gl = self.ToGlobalCode(x)
        sp = normalize(sp)
        gl = normalize(gl)
        if extract_features:
            return sp, gl, feature
        else:
            return sp, gl

class SWAEDecoder(nn.Module):
    """ The Generator (decoder) architecture described in Figure 18 of
        Swapping Autoencoder (https://arxiv.org/abs/2007.00653).
        
        At high level, the architecture consists of regular and 
        upsampling residual blocks to transform the structure code into an RGB
        image. The global code is applied at each layer as modulation.
        
        Here's more detailed architecture:
        
        1. SpatialCodeModulation: First of all, modulate the structure code 
        with the global code.
        2. HeadResnetBlock: resnets at the resolution of the structure code,
        which also incorporates modulation from the global code.
        3. UpsamplingResnetBlock: resnets that upsamples by factor of 2 until
        the resolution of the output RGB image, along with the global code
        modulation.
        4. ToRGB: Final layer that transforms the output into 3 channels (RGB).
        
        Each components of the layers borrow heavily from StyleGAN2 code,
        implemented by Seonghyeon Kim.
        https://github.com/rosinality/stylegan2-pytorch
    """
    def __init__(self):
        super().__init__()
        self.netG_scale_capacity = 1.0
        self.netG_num_base_resnet_layers = 2
        self.netG_resnet_ch = 256
        self.netE_num_downsampling_sp = 4
        self.num_classes = 0
        self.global_code_ch = 2048
        self.spatial_code_ch = 8

        num_upsamplings = self.netE_num_downsampling_sp
        blur_kernel = [1, 3, 3, 1]# if opt.use_antialias else [1]

        self.global_code_ch = self.global_code_ch + self.num_classes

        self.add_module(
            "SpatialCodeModulation",
            GeneratorModulation(self.global_code_ch, self.spatial_code_ch))

        in_channel = self.spatial_code_ch
        for i in range(self.netG_num_base_resnet_layers):
            # gradually increase the number of channels
            out_channel = (i + 1) / self.netG_num_base_resnet_layers * self.nf(0)
            out_channel = max(self.spatial_code_ch, round(out_channel))
            layer_name = "HeadResnetBlock%d" % i
            new_layer = ResolutionPreservingResnetBlock(
                in_channel, out_channel, self.global_code_ch)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        for j in range(num_upsamplings):
            out_channel = self.nf(j + 1)
            layer_name = "UpsamplingResBlock%d" % (2 ** (4 + j))
            new_layer = UpsamplingResnetBlock(
                in_channel, out_channel, self.global_code_ch,
                blur_kernel, False)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        last_layer = ToRGB(out_channel, self.global_code_ch,
                           blur_kernel=blur_kernel)
        self.add_module("ToRGB", last_layer)

    def nf(self, num_up):
        ch = 128 * (2 ** (self.netE_num_downsampling_sp - num_up))
        ch = int(min(512, ch) * self.netG_scale_capacity)
        return ch

    def forward(self, spatial_code, global_code):
        spatial_code = normalize(spatial_code)
        global_code = normalize(global_code)

        x = self.SpatialCodeModulation(spatial_code, global_code)
        for i in range(self.netG_num_base_resnet_layers):
            resblock = getattr(self, "HeadResnetBlock%d" % i)
            x = resblock(x, global_code)

        for j in range(self.netE_num_downsampling_sp):
            key_name = 2 ** (4 + j)
            upsampling_layer = getattr(self, "UpsamplingResBlock%d" % key_name)
            x = upsampling_layer(x, global_code)
        rgb = self.ToRGB(x, global_code, None)

        return rgb

## SWAE copy
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        pad=None,
        reflection_pad=False,
    ):
        layers = []

        if downsample:
            factor = 2
            if pad is None:
                pad = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (pad + 1) // 2
            pad1 = pad // 2

            layers.append(("Blur", Blur(blur_kernel, pad=(pad0, pad1), reflection_pad=reflection_pad)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2 if pad is None else pad
            if reflection_pad:
                layers.append(("RefPad", nn.ReflectionPad2d(self.padding)))
                self.padding = 0


        layers.append(("Conv",
                       EqualConv2d(
                           in_channel,
                           out_channel,
                           kernel_size,
                           padding=self.padding,
                           stride=stride,
                           bias=bias and not activate,
                       ))
        )

        if activate:
            if bias:
                layers.append(("Act", FusedLeakyReLU(out_channel)))

            else:
                layers.append(("Act", ScaledLeakyReLU(0.2)))

        super().__init__(OrderedDict(layers))

    def forward(self, x):
        out = super().forward(x)
        return out

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, lr_mul=1.0,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], reflection_pad=False, pad=None, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, reflection_pad=reflection_pad, pad=pad)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel, reflection_pad=reflection_pad, pad=pad)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, blur_kernel=blur_kernel, activate=False, bias=False
        )

    def forward(self, input):
        #print("before first resnet layeer, ", input.shape)
        out = self.conv1(input)
        #print("after first resnet layer, ", out.shape)
        out = self.conv2(out)
        #print("after second resnet layer, ", out.shape)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

def normalize(v):
    if type(v) == list:
        return [normalize(vv) for vv in v]

    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class UpsamplingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim, blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True, blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False, use_noise=use_noise)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=True, bias=True)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)

class ResolutionPreservingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim):
        super().__init__()
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=False)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=False, bias=False)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = self.skip(x)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)

class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        use_noise=True,
        lr_mul=1.0,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.use_noise = use_noise
        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = nn.ReLU(True)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        if self.use_noise:
            out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))
        self.fixed_noise = None
        self.image_size = None

    def forward(self, image, noise=None):
        if self.image_size is None:
            self.image_size = image.shape

        if noise is None and self.fixed_noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        elif self.fixed_noise is not None:
            noise = self.fixed_noise
            # to avoid error when generating thumbnails in demo
            if image.size(2) != noise.size(2) or image.size(3) != noise.size(3):
                noise = F.interpolate(noise, image.shape[2:], mode="nearest")
        else:
            pass  # use the passed noise

        return image + self.weight * noise

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.new_demodulation = True

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if style.dim() > 2:
            style = F.interpolate(style, size=(input.size(2), input.size(3)), mode='bilinear', align_corners=False)
            style = self.modulation(style).unsqueeze(1)
            if self.demodulate:
                style = style * torch.rsqrt(style.pow(2).mean([2], keepdim=True) + 1e-8)
            input = input * style
            weight = self.scale * self.weight
            weight = weight.repeat(batch, 1, 1, 1, 1)
        else:
            style = style.view(batch, style.size(1))
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            if self.new_demodulation:
                style = style[:, 0, :, :, :]
                if self.demodulate:
                    style = style * torch.rsqrt(style.pow(2).mean([1], keepdim=True) + 1e-8)
                input = input * style
                weight = self.scale * self.weight
                weight = weight.repeat(batch, 1, 1, 1, 1)
            else:
                weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, reflection_pad=False):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.reflection = reflection_pad
        if self.reflection:
            self.reflection_pad = nn.ReflectionPad2d((pad[0], pad[1], pad[0], pad[1]))
            self.pad = (0, 0)

    def forward(self, input):
        if self.reflection:
            input = self.reflection_pad(input)
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.dim() == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            if input.dim() > 2:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale)
            else:
                out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            if input.dim() > 2:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale,
                               bias=self.bias * self.lr_mul
                )
            else:
                out = F.linear(
                    input, self.weight * self.scale, bias=self.bias * self.lr_mul
                )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
