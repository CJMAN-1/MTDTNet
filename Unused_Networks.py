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

def count_img(root):
    cnt = 0
    for mode in ['test', 'train', 'val']:
        img_file = os.path.join(root,'leftImg8bit',mode)
        img_folder = os.listdir(img_file)

        for folder in img_folder:
            cur_folder = os.path.join(img_file, folder)
            cnt += len(os.listdir(cur_folder))
            # print(os.listdir(cur_folder))
        print(mode)
        print(cnt)

        cnt = 0

    return 0

def get_all_labels(root):
    labels = []
    final_labels = {}
    segmap_folder = os.listdir(root)
    
    print(root)
#     sub_path_train_img = os.path.join(root, img_folder, './train')
    sub_path_train_seg = os.path.join(root+'/training', 'v1.2', 'polygons')

#     sub_path_val_img = os.path.join(root, img_folder, './val')
    sub_path_val_seg = os.path.join(root+'/validation', 'v1.2', 'polygons')
    
    print(sub_path_train_seg)
    # extract training labels
    for file in os.listdir(sub_path_train_seg):
        # curr_seg_folder = os.path.join(sub_path_train_seg, folder)
        # for file in os.listdir(curr_seg_folder):
        if 'json' not in file:
            continue
        f = open(os.path.join(sub_path_train_seg, file), 'r')
        data = json.loads(f.read())
        for obj in data['objects']:
            if obj['label'] not in labels:
                labels.append(obj['label'])

    # extract validation labels
    for file in os.listdir(sub_path_val_seg):
        # curr_seg_folder = os.path.join(sub_path_val_seg, folder)
        # for file in os.listdir(curr_seg_folder):
        if 'json' not in file:
            continue
        f = open(os.path.join(sub_path_val_seg, file), 'r')
        data = json.loads(f.read())
        for obj in data['objects']:
            if obj['label'] not in labels:
                labels.append(obj['label'])
                    
    for i in range(len(labels)):
        final_labels[labels[i]] = i
    print(final_labels)
    return final_labels


def create_segmentation_maps(root, labels):
    # test, train, val = os.listdir(root)
    
    if not os.path.exists('./img'):
        os.makedirs('img')
    # if not os.path.exists('./img/train'):
    #     os.makedirs('img/train')
    # if not os.path.exists('./img/val'):
    #     os.makedirs('img/val')

    # if not os.path.exists('./seg'):
    #     os.makedirs('seg')
    # if not os.path.exists('./seg/train'):
    #     os.makedirs('seg/train')
    # if not os.path.exists('./seg/val'):
    #     os.makedirs('seg/val')
    
    sub_path_train_img = os.path.join(root, 'training/images/')
    sub_path_train_seg = os.path.join(root, 'training/v2.0/polygons')

    sub_path_val_img = os.path.join(root, 'validation/images/')
    sub_path_val_seg = os.path.join(root, 'validation/v2.0/polygons')
    
    # for folder in os.listdir(sub_path_train_seg):
    #     curr_img_folder = os.path.join(sub_path_train_img, folder)
    #     curr_seg_folder = os.path.join(sub_path_train_seg, folder)
        
    for file in os.listdir(sub_path_train_seg):
        id = file.split('.')[0]
        print(file)
        if not os.path.exists('/data/datasets/mapillary/training/seg/'+id+'.png'):
            polygoni = open(os.path.join(sub_path_train_seg, file), 'r')
            
            # f = open(os.path.join(curr_seg_folder, id+'_gtFine_polygons.json'), 'r')
            data = json.loads(polygoni.read())
            seg_map = np.zeros((data['width'],data['height']))
            for obj in data['objects']:
                label = obj['label']
                poly = np.array(obj['polygon'])
                # print('segmap shape :',seg_map.shape)
                # print(poly.shape)
                if len(poly.shape) == 2:
                    # print('legal')
                    # print(poly[:,0])
                    # print(poly[:,1])
                    rr, cc = polygon(poly[:,0], poly[:,1], seg_map.shape)
                    seg_map[rr,cc] = labels[label]
            
            # print('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png')
            print('/data/datasets/mapillary/training/'+file)
            # cv2.imwrite('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png', img)
            cv2.imwrite('/data/datasets/mapillary/training/seg/'+id+'.png', seg_map.T)
            print('/data/datasets/mapillary/training/seg/'+id+'.png')

     
    for file in os.listdir(sub_path_val_seg):
        id = file.split('.')[0]
        print(file)
        if not os.path.exists('/data/datasets/mapillary/validation/seg/'+id+'.png'):
            polygoni = open(os.path.join(sub_path_val_seg, file), 'r')
            
            # f = open(os.path.join(curr_seg_folder, id+'_gtFine_polygons.json'), 'r')
            data = json.loads(polygoni.read())
            seg_map = np.zeros((data['width'],data['height']))
            for obj in data['objects']:
                label = obj['label']
                poly = np.array(obj['polygon'])
                # print('segmap shape :',seg_map.shape)
                # print(poly.shape)
                if len(poly.shape) == 2:
                    # print('legal')
                    # print(poly[:,0])
                    # print(poly[:,1])
                    rr, cc = polygon(poly[:,0], poly[:,1], seg_map.shape)
                    seg_map[rr,cc] = labels[label]
            
            # print('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png')
            print('/data/datasets/mapillary/validation/'+file)
            # cv2.imwrite('/data/datasets/IDD/leftImg8bit/train/'+folder+'/'+id+'.png', img)
            cv2.imwrite('/data/datasets/mapillary/validation/seg/'+id+'.png', seg_map.T)
            print('/data/datasets/mapillary/validation/seg/'+id+'.png')


class Domain_Style_Transfer(nn.Module):
    def __init__(self, datasets):
        super().__init__()
        self.n = dict()
        self.m = dict()
        self.s = dict()
        self.mlp_mean = nn.ModuleDict()
        self.mlp_std = nn.ModuleDict()
        
        for dset in datasets:
            self.n[dset] = 0
            self.m[dset] = 0
            self.s[dset] = 0
            self.w = 1
            self.mlp_mean[dset] = nn.Sequential(
                nn.Linear(1024, 1024),
            )
            self.mlp_std[dset] = nn.Sequential(
                nn.Linear(1024, 1024),
            )
        
    def forward(self, style, target, update=False):
        # Domain mean, std
        if update:
            self.n[target] += 1
            if self.n[target] == 1:
                self.m[target] = style  # B x C
                self.s[target] = (style - self.m[target].mean(dim=0, keepdim=True)) ** 2
            else:
                prev_m = self.m[target].mean(dim=0, keepdim=True)  # 1 x C
                self.m[target] += self.w * (style - self.m[target]) / self.n[target]  # B x C
                curr_m = self.m[target].mean(dim=0, keepdim=True)  # 1 x C
                self.s[target] += self.w * (style - prev_m) * (style - curr_m)  # B x C
        else:
            beta = self.mlp_mean[target](self.m[target].mean(dim=0).unsqueeze(0))  # 1 x C
            gamma = self.mlp_std[target](((self.s[target].mean(dim=0))/self.n[target]).sqrt().unsqueeze(0))  # 1 x C
            
            return gamma * style + beta


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

# LRU policy : hit -> change, miss -> stay
# class Adaptive_Style_Memory_Bank(nn.Module):
#     def __init__(self, targets, size=8):
#         super().__init__()
#         self.content = dict()
#         self.style = dict()
#         self.n_target = len(targets)
#         self.size = size
#         self.n_usage = dict()
#         for i in range(self.n_target):
#             self.n_usage[targets[i]] = [0 for i in range(8)]
    
#     # 6, 7th memory is register, 0~5 is saved content & style
#     def forward(self, content, target):
#         b = content.shape[0]
#         content_source = F.normalize(content.view(b, -1), dim=1)  # B1 x N
#         content_target = F.normalize(self.content[target].view(-1, self.content[target].size(0)), dim=0)  # N x B2
#         sim_mat = content_source.mm(content_target)  # B1 x B2
#         idx = sim_mat.argmax(dim=1)
#         adaptive_style = self.style[target][idx]
#         for i in range(len(idx)):
#             self.n_usage[target][int(idx[i])] += 1
#         # self.print_memory()
#         # print('memory size :', self.content[target].size(0))
#         # print(self.content[target].shape)
#         for i in range(b):
#             if self.content[target].size(0) > self.size - b:
#                 # LRU policy : 
#                 #   if idx != 7 -> miss
#                 if int(idx[i]) != self.content[target].size(0)-1:
#                     self.content[target] = self.content[target][:-1]
#                     self.style[target] = self.style[target][:-1]
#                 #   if idx == 6 or 7 -> hit : select least used style is removed
#                 else:
#                     trash_idx = self.n_usage[target].index(min(self.n_usage[target][:-b]))
#                     self.n_usage[target][trash_idx] = 1
#                     # print(trash_idx)
#                     # print(int(idx[-i-1]))
#                     # print(self.content[target][trash_idx].shape) 
#                     # print(self.content[target][idx].shape)
#                     self.content[target][trash_idx] = self.content[target][int(idx[i])]
#                     # remove style (num : batch)
#                     self.content[target] = self.content[target][:-1]
#                     self.style[target] = self.style[target][:-1]
#                 # remove_index = torch.unique(sim_mat.argmin(dim=1))
#                 # if remove_index.size(0) == content_source.size(0):
                
#         return adaptive_style

#     def print_memory(self):
#         print('N_usage - (target) : (index_list)', )
#         print(self.n_usage)

#     def update(self, content, style, target):
#         if len(self.content) < self.n_target:
#             self.content[target] = content
#             self.style[target] = style
#         else:
#             self.content[target] = torch.cat([self.content[target], content], dim=0)
#             self.style[target] = torch.cat([self.style[target], style], dim=0)

        # if len(self.content) < self.n_target:
        #     self.content[target] = content
        #     self.style[target] = style
        # else:
        #     self.content[target] = torch.cat([self.content[target], content], dim=0)
        #     self.style[target] = torch.cat([self.style[target], style], dim=0)

class Weight_Prediction_Networks(nn.Module):
    def __init__(self, targets, n_class):
        super().__init__()
        self.n_class = n_class
        self.n = dict()
        self.centroid = dict()
        self.mlp = nn.ModuleDict()
        self.conv = nn.ModuleDict()

        for target in targets:
            self.n[target] = 0
            self.mlp[target] = nn.ModuleDict()
            self.conv[target] = nn.Conv2d(64, 64, 1, 1, 0)
            for cls in range(n_class):
                self.mlp[target][str(cls)] = nn.Linear(64, 64)
        
    def forward(self, feature, target):
        mlp_centroid = torch.zeros_like(self.centroid[target].detach())  # Class x N
        for cls in range(self.n_class):
            mlp_centroid[cls] = self.mlp[target][str(cls)](self.centroid[target][cls])
        conv_feature = F.normalize(self.conv[target](feature), dim=1)  # B x C x H x W
        score_map = torch.matmul(conv_feature.permute(0,2,3,1), F.normalize(mlp_centroid, dim=1).permute(1, 0)).permute(0,3,1,2)  # B x Class x H x W
        # print(score_map)
        return F.softmax(10*score_map, dim=1)

    def update(self, feature, seg, target):
        self.n[target] += 1
        new_centroid = self.region_wise_pooling(feature, seg)
        if self.n[target] == 1:
            self.centroid[target] = new_centroid
        else:
            self.centroid[target] += ((new_centroid - self.centroid[target]) / self.n[target])

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


class Weight_Prediction_Networks(nn.Module):
    def __init__(self, targets, n_class):
        super().__init__()
        self.n_class = n_class
        self.n = dict()
        self.centroid = dict()
        self.mlp = nn.ModuleDict()
        self.conv = nn.ModuleDict()

        for target in targets:
            self.n[target] = 0
            self.mlp[target] = nn.ModuleDict()
            self.conv[target] = nn.Conv2d(2048, 256, 1, 1, 0)
            for cls in range(n_class):
                self.mlp[target][str(cls)] = nn.Linear(2048, 256)
        
    def forward(self, feature, target):
        mlp_centroid = torch.zeros_like(self.centroid[target].detach())  # Class x N
        for cls in range(self.n_class):
            mlp_centroid[cls] = self.mlp[target][str(cls)](self.centroid[target][cls])
        conv_feature = F.normalize(self.conv[target](feature), dim=1)  # B x C x H x W
        score_map = torch.matmul(conv_feature.permute(0,2,3,1), F.normalize(mlp_centroid, dim=1).permute(1, 0)).permute(0,3,1,2)  # B x Class x H x W
        # print(score_map)
        return F.softmax(10*score_map, dim=1)

    def update(self, feature, seg, target):
        self.n[target] += 1
        new_centroid = self.region_wise_pooling(feature, seg)
        if self.n[target] == 1:
            self.centroid[target] = new_centroid
        else:
            self.centroid[target] += ((new_centroid - self.centroid[target]) / self.n[target])

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


class Domain_Discriminator(nn.Module):
    def __init__(self, num_domains, channels=2048):
        super().__init__()
        self.Conv = nn.Sequential(
            # input size: 256x256
            spectral_norm(nn.Conv2d(channels, 1024, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)),  # 
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(256*8*16,500)),
            nn.ReLU(),
            spectral_norm(nn.Linear(500, 2))
        )
    
    def forward(self, feature):
        out = self.Conv(feature)
        return self.fc(out.view(out.size(0), -1))


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
