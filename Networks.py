import torch
import torch.nn as nn
import torch.nn.functional as F
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


# class Style_Encoder(nn.Module):
#     def __init__(self, channels=3):
#         super(Style_Encoder, self).__init__()
#         bin = functools.partial(nn.GroupNorm, 4)
#         self.Encoder_Conv = nn.Sequential(
#             nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
#             bin(32),
#             nn.ReLU(True),
#             ResidualBlock(32, 32),
#             ResidualBlock(32, 32),
            
#         )
#         self.gamma = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.ReLU(True),
#         )
#         self.beta = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.ReLU(True),
#         )

#     def forward(self, inputs):
#         output = self.Encoder_Conv(inputs) # batch_size x 512 x 10 x 6
#         gamma = self.gamma(output)
#         beta = self.beta(output)
#         return gamma, beta



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


# class Content_Encoder(nn.Module):
#     def __init__(self, n_class, channels=3):
#         super(Content_Encoder, self).__init__()
#         self.n_class = n_class
#         self.conv = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.ReLU(True),
#             nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),
#             nn.ReLU(True),
#             ResidualBlock(128, 128),
#             ResidualBlock(128, 128),
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.ReLU(True),
#             ResidualBlock(64, 64),
#             ResidualBlock(64, 64),
#             nn.InstanceNorm2d(64, affine=False)
#         )
    
#         self.embedding = nn.Conv2d(n_class+1, 64, kernel_size=1)

#     def forward(self, seg):
#         segmap = F.one_hot(seg.squeeze(1) + 1, num_classes=self.n_class+1).permute(0,3,1,2)
#         segmap = self.embedding(segmap.float())
#         return self.conv(segmap)


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


#////////////////////////////////////////////////////////////////
"""TODO : build module that extract gamma, beta tensor from image and segmap"""
"""INPUT : feature dictionary, segmap dictionary 
        -> e.g. feature[src] = F_x, segmap[tgt] = seg_y"""
"""OUTPUT : Affine tensor gamma beta dictionary"""
""" 1. REGION WISE POOLING -> Matrix1 : B*cls*C 
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
#////////////////////////////////////////////////////////////////

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.Decoder_Conv = nn.Sequential(
#             spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
#             nn.ReLU(True),
#             ResidualBlock(32, 32),
#             ResidualBlock(32, 32),
#             # batch_size x 3 x 1280 x 768
#             spectral_norm(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True)),
#             nn.Tanh()
#         )
#     def forward(self, x):
#         return self.Decoder_Conv(x)


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
