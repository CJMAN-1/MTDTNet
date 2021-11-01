'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import logging
import random

from Networks import *
from torch.nn.parameter import Parameter


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # init.normal(m.weight, std=1e-3)
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def safe_load_state_dict(net, state_dict):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. Any params in :attr:`state_dict`
    that do not match the keys returned by :attr:`net`'s :func:`state_dict()`
    method or have differing sizes are skipped.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
    """
    own_state = net.state_dict()
    skipped = []
    for name, param in state_dict.items():
        if name not in own_state:
            skipped.append(name)
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if own_state[name].size() != param.size():
            skipped.append(name)
            continue
        own_state[name].copy_(param)

    if skipped:
        logging.info('Skipped loading some parameters: {}'.format(skipped))

# models = {}
# def register_model(name):
#     def decorator(cls):
#         models[name] = cls
#         return cls
#     return decorator


def gram(x):
    (b, c, h, w) = x.size()
    f = x.view(b, c, h*w)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (c*w*h)
    return G


def gram_content(x, y):
    s1 = nn.Softmax(dim=1)
    s2 = nn.Softmax(dim=0)
    (b, c, h, w) = x.size()
    fx = x.view(b, c*h*w)
    fy = y.view(b, c*h*w)
    fy = fy.transpose(0, 1)
    G = fx @ fy
    G = G / torch.norm(G, p=2)
    G1 = s1(G)
    G2 = s2(G)
    return G1, G2


def cadt(source, target, style, W=10):
    s1 = nn.Softmax(dim=1)
    (b, c, h, w) = source.size()
    fs = source.view(b, c * h * w)
    ft = target.view(b, c * h * w)
    ft = ft.transpose(0, 1)
    H = fs @ ft
    H = H / torch.norm(H, p=2)
    H = s1(W * H)
    adaptive_style = style.view(b, c * h * w)
    adaptive_style = H @ adaptive_style
    adaptive_style = adaptive_style.view(b, c, h, w)
    return H, adaptive_style


def cadt_gram(gram, con_sim):
    adaptive_gram = []
    for g in range(len(gram)):
        (b, n1, n2) = gram[g].size()
        fg = gram[g].view(b, n1 * n2)
        adaptive_gram.append(con_sim @ fg)
        adaptive_gram[g] = adaptive_gram[g].view(b, n1, n2)
    return adaptive_gram



def re_gram(GS, GT, G1, G2):
    batch = GS.size(0)
    new_GS = torch.zeros_like(GS)
    new_GT = torch.zeros_like(GT)
    for b in range(batch):
        weighted_GS = GS * G1[b].view(batch, 1, 1)
        weighted_GT = GT * G2[:,b].view(batch, 1, 1)
        weighted_GS = weighted_GS.sum(dim=0)
        weighted_GT = weighted_GT.sum(dim=0)
        new_GS[b] = weighted_GS.squeeze(0)
        new_GT[b] = weighted_GT.squeeze(0)
    return new_GS, new_GT


def re_style(SS, ST, G1, G2):
    batch = SS.size(0)
    new_SS = torch.zeros_like(SS)
    new_ST = torch.zeros_like(ST)
    for b in range(batch):
        weighted_SS = SS * G1[b].view(batch, 1, 1, 1)
        weighted_ST = ST * G2[:,b].view(batch, 1, 1, 1)
        weighted_SS = weighted_SS.sum(dim=0)
        weighted_ST = weighted_ST.sum(dim=0)
        new_SS[b] = weighted_SS.squeeze(0)
        new_ST[b] = weighted_ST.squeeze(0)
    return new_SS, new_ST


def MIOU(GT, Pred, num_class=19):
    confusion_matrix = np.zeros((num_class,) * 2)
    mask = (GT >= 0) & (GT < num_class)
    label = num_class * GT[mask].astype('int') + Pred[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix += count.reshape(num_class, num_class)
    return confusion_matrix


def MIOU_score(GT, Pred, num_class=19):
    confusion_matrix = np.zeros((num_class,) * 2)
    mask = (GT >= 0) & (GT < num_class)
    label = num_class * GT[mask].astype('int') + Pred[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix += count.reshape(num_class, num_class)
    score = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
    score = np.nanmean(score)
    return score


def content_sim(cs, ct):
    s1 = nn.Softmax(dim=1)
    s2 = nn.Softmax(dim=0)
    (b, c, h, w) = cs.size()
    content_s = cs
    content_t = ct
    content_s = content_s.view(b, c*h*w)
    (b, c, h, w) = ct.size()
    content_t = content_t.view(b, c*h*w)
    content_s = content_s.div((torch.norm(content_s, p=2, dim=1, keepdim=True)).expand_as(content_s))
    content_t = content_t.div((torch.norm(content_t, p=2, dim=1, keepdim=True)).expand_as(content_t))
    content_t = content_t.transpose(0, 1)
    G = content_s @ content_t

    # G = 15 * G
    # G1 = s1(G)
    # G2 = s2(G)

    # print(np.round(G.cpu().numpy(), 2))
    return G


def content_aware_style_loss(gt_gram, convert_gram, G, MSE):
    batch = gt_gram.size(0)
    loss = 0
    for s_ in range(batch):
        for t_ in range(batch):
            loss += MSE(gt_gram[t_], convert_gram[s_]) * G[s_, t_]
    return loss


# def make_seg_parts(img, seg):
#     for c in range(seg.size(1)):
#         if len(seg.size()) == 4:
#             mask = seg[:,c,:,:].unsqueeze(1)
#         else:
#             mask = (seg==c).unsqueeze(1)
#         cl = c * torch.ones_like(mask)
#         if c==0:
#             out = img * mask
#             out = torch.cat([out, cl], dim=1)
#         else:
#             if torch.equal(mask, torch.zeros_like(mask)):
#                 pass
#             else:
#                 out = torch.cat([out, torch.cat([img * mask, cl], dim=1)], dim=0) # (Bx19) x (3+1) x H x W
#     return out


def make_seg_parts(img, seg):
    for c in range(seg.size(1)):
        if len(seg.size()) == 4:
            mask = seg[:,c,:,:].unsqueeze(1)  # B x 1 x H x W
        else:
            cl = c * torch.ones_like(mask)
        if c==0:
            out = img * mask
            out = torch.cat([out, cl], dim=1)
        else:
            if torch.equal(mask, torch.zeros_like(mask)):
                pass
            else:
                out = torch.cat([out, torch.cat([img * mask, cl], dim=1)], dim=0) # (Bx19) x (3+1) x H x W
    return out


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg


def pred2seg(pred, n_class=19):
    seg = torch.argmax(pred, dim=1)
    return seg


def pred2conf(pred, n_class=19):
    pred_ = F.softmax(pred, dim=1)
    log_pred = F.log_softmax(pred, dim=1)
    e = -(1/(n_class*math.log(n_class))) * torch.sum(pred_*log_pred, dim=1)
    seg = torch.argmax(pred, dim=1)
    seg[e>0.01] = -1
    return seg


def slice_patches(imgs, hight_slice=2, width_slice=4):
    b, c, h, w = imgs.size()
    h_patch, w_patch = int(h / hight_slice), int(w / width_slice)
    patches = imgs.unfold(2, h_patch, h_patch).unfold(3, w_patch, w_patch)
    patches = patches.contiguous().view(b, c, -1, h_patch, w_patch)
    patches = patches.transpose(1,2)
    patches = patches.reshape(-1, c, h_patch, w_patch)
    return patches


def check_road_patches(seg):
    b = seg.size(0)
    road_mask = (seg==0).unsqueeze(1)  # b x 1 x h x w
    # road_img = road_mask * img
    road_mask_patch = slice_patches(road_mask)
    pair =[[] for batch in range(b)]
    for batch in range(b):
        for p in range(4):
            if (1.*road_mask_patch[8*batch+p+4,:,:,:]).mean() > 0.9:
                pair[batch].append(p+4)
    return pair


def class_weight_by_frequency(seg, n_class):
    b, h, w = seg.size()
    freq = F.one_hot(seg + 1, num_classes=n_class+1).permute(0,3,1,2)  # B x (cls+1) x H x W
    freq = torch.sum(freq, dim=(0, 2, 3))  # cls+1
    freq = freq[1:].float()  # cls, 0: ignore label
    for cls in range(n_class):
        if freq[cls] > 0:
            ratio = freq[cls] / (b*h*w)
            if ratio > 1/n_class:
                freq[cls] = (1-ratio)**2
            else:
                freq[cls] = 1+0.5
    return freq


# def CST(gamma, beta, seg, converts, n_class):
#     dim_ = (2,3)
#     for convert in converts:
#         source, target = convert.split('2')
#         h, w = gamma[source].shape[2:]
#         segmap_source = F.one_hot(seg[source] + 1, num_classes=n_class+1).permute(0,3,1,2)
#         segmap_target = F.one_hot(seg[source] + 1, num_classes=n_class+1).permute(0,3,1,2)
#         segmap_source = F.interpolate(segmap_source.float(), size=(h,w), mode='nearest')
#         segmap_target = F.interpolate(segmap_target.float(), size=(h,w), mode='nearest')  # B x cls x H x W
#         # global mean, std
#         gamma_source_mean = gamma[source].mean(dim=dim_, keepdim=True)  # B x C x 1 x 1
#         gamma_source_std = gamma[source].std(dim=dim_, keepdim=True)
#         gamma_target_mean = gamma[target].mean(dim=dim_, keepdim=True)
#         gamma_target_std = gamma[target].std(dim=dim_, keepdim=True)
#         beta_source_mean = beta[source].mean(dim=dim_, keepdim=True)  # B x C x 1 x 1
#         beta_source_std = beta[source].std(dim=dim_, keepdim=True)
#         beta_target_mean = beta[target].mean(dim=dim_, keepdim=True)
#         beta_target_std = beta[target].std(dim=dim_, keepdim=True)
        
#         # new_gamma = (gamma_target_std * ((gamma[source] - gamma_source_mean) / (gamma_source_std + 1e-8)) + gamma_target_mean)
#         # new_beta = (beta_target_std * ((beta[source] - beta_source_mean) / (beta_source_std + 1e-8)) + beta_target_mean)

#         new_gamma = torch.zeros_like(gamma[source])
#         new_beta = torch.zeros_like(beta[source])

#         for cls in range(segmap_source.size(1)):
#             gamma_source_cls = segmap_source[:,cls,:,:].unsqueeze(1) * gamma[source]
#             gamma_target_cls = segmap_target[:,cls,:,:].unsqueeze(1) * gamma[target]
#             beta_source_cls = segmap_source[:,cls,:,:].unsqueeze(1) * beta[source]
#             beta_target_cls = segmap_target[:,cls,:,:].unsqueeze(1) * beta[target]

#             if cls == 0:
#                 # global mean, std
#                 new_gamma += (gamma_target_std * ((gamma_source_cls - gamma_source_mean) / (gamma_source_std + 1e-8)) + gamma_target_mean)
#                 new_beta += (beta_target_std * ((beta_source_cls - beta_source_mean) / (beta_source_std + 1e-8)) + beta_target_mean)
#             else: 
#             # class-wise mean, std
#                 gamma_source_mean_cls = gamma_source_cls.mean(dim=dim_, keepdim=True)  # B x C x 1 x 1
#                 gamma_source_std_cls = gamma_source_cls.std(dim=dim_, keepdim=True)
#                 gamma_target_mean_cls = gamma_target_cls.mean(dim=dim_, keepdim=True)
#                 gamma_target_std_cls = gamma_target_cls.std(dim=dim_, keepdim=True)
#                 beta_source_mean_cls = beta_source_cls.mean(dim=dim_, keepdim=True)  # B x C x 1 x 1
#                 beta_source_std_cls = beta_source_cls.std(dim=dim_, keepdim=True)
#                 beta_target_mean_cls = beta_target_cls.mean(dim=dim_, keepdim=True)
#                 beta_target_std_cls = beta_target_cls.std(dim=dim_, keepdim=True)
                
#                 new_gamma += (gamma_target_std_cls * ((gamma_source_cls - gamma_source_mean_cls) / (gamma_source_std_cls + 1e-8)) + gamma_target_mean_cls)
#                 new_beta += (beta_target_std_cls * ((beta_source_cls - beta_source_mean_cls) / (beta_source_std_cls + 1e-8)) + beta_target_mean_cls)
#         gamma[convert] = new_gamma
#         beta[convert] = new_beta
#     return gamma, beta



