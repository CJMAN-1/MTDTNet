from __future__ import print_function
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.backends.cudnn
import numpy as np

from Networks import *
from deeplabv2 import Deeplab

from utils import *
from losses import *

from dataloader.Cityscapes import decode_labels
from dataset import get_dataset
from param import get_params
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


opt = get_params()
print(opt)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

## Logger ##
logger = logging.getLogger()
file_log_handler = logging.FileHandler(opt.logfile)
logger.addHandler(file_log_handler)

stderr_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stderr_log_handler)

logger.setLevel('INFO')
formatter = logging.Formatter()
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)

# imageSize = (opt.Height, opt.Width)

## Data Loaders ##
source_train_loader, _ = get_dataset(dataset='G', batch=4,
                                                    imsize=(1024, 512), workers=opt.workers)
target_train_loader, target_test_loader = get_dataset(dataset='C', batch=4, imsize=(1024, 512), workers=opt.workers)



# netT = resnet101(pretrained=True)
# netT = DeeplabMulti(num_classes=19, pretrained=False)
# netT = DeeplabMulti(num_classes=19, pretrained=True)
netT = Deeplab(num_classes=19, restore_from='./pretrained_model/gta5')

criterion_T = nn.CrossEntropyLoss(weight=None, ignore_index=-1)
L1 = nn.L1Loss()
MSE = nn.MSELoss()
BCE = nn.BCELoss()
BCE_M = nn.BCEWithLogitsLoss()
# SSIM = pytorch_msssim.SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
# MS_SSIM = pytorch_msssim.MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)

if opt.cuda:
    netT.cuda()
    criterion_T.cuda()
    L1.cuda()
    MSE.cuda()
    BCE.cuda()
    BCE_M.cuda()
    # SSIM.cuda()
    # MS_SSIM.cuda()
    # inputs, label = inputs.cuda(), label.cuda()

optimizerT = optim.SGD(netT.parameters(), lr=2.5e-4, momentum=0.9, weight_decay=5e-4)



def test(epoch, test_loader, best_acc, save=True, dataset="target", is_plot=False):
    netT.eval()
    test_loss = 0
    miou = 0
    confusion_matrix = np.zeros((19,) * 2)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if opt.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            targets = targets.long()
            pred = netT(inputs, lbl=targets)
            # loss = criterion_T(pred, targets)
            loss_seg_src = netT.loss_seg
            loss_ent_src = netT.loss_ent
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            gt = targets.data.cpu().numpy()
            confusion_matrix += MIOU(gt, pred)

            score = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
            miou = np.nanmean(score)

            test_loss += loss_seg_src.data.item()
            # total += 1
            # print(targets.size(0))
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%%'
                         % (test_loss/(batch_idx+1), 100.*miou))
        # Save checkpoint.
        miou = 100.*miou
        logger.info('======================================================')
        logger.info('Epoch: %d | Loss: %.3f | Acc: %.3f%%'
                    % (epoch, test_loss/len(test_loader), miou))
        logger.info('======================================================')
        if dataset == 'source' and miou > best_acc:
            torch.save(netT.state_dict(), './pretrained_model/deeplab_gta5.pth')
            return miou
        else:
            return best_acc


best_miou = 0
iterations = 0
step = 0
n_iter = 100


for epoch in range(n_iter):
    netT.train()
    for i, source_data in enumerate(source_train_loader, 0):
        step += 1

        ## Source Batch ##
        source_cpu, source_label = source_data
        if opt.cuda:
            source_cpu, source_label = source_cpu.cuda(), source_label.cuda()

        source_label = source_label.long()
        # i_, j, h, w = transforms.RandomCrop.get_params(source_cpu, output_size = (128, 128))
        # input_s, source_label = \
        # TF.crop(source_cpu,i_,j,h,w), TF.crop(source_label,i_,j,h,w)
        input_s = source_cpu
        

        netT.zero_grad()

        pred_s = netT(input_s, lbl=source_label)
        # errT = criterion_T(pred_s, source_label)
        # errT.backward()
        loss_seg_src = netT.loss_seg
        # loss_ent_src = netT.loss_ent
        loss_all = loss_seg_src # + loss_ent_src

        loss_all.backward()
        optimizerT.step()

        pred = pred_s.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        gt = source_label.data.cpu().numpy()
        miou = MIOU_score(gt, pred)


        logger.info(
            '[%d/%d][%d/%d]  T: %.4f | M: %.4f | Best: %.4f | %s'
            % (epoch, n_iter, i, len(source_train_loader),
                loss_all.data.item(), miou, best_miou, opt.ex))


    best_miou = test(-1, target_test_loader, best_miou, save=False, dataset="source")