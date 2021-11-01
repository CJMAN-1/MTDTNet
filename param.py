import argparse
import os
import random
from data_list import make_list
from Networks import *


def check_dirs(dirs):
    dirs = [dirs] if type(dirs) not in [list, tuple] else dirs
    for d in dirs:
        try:
            os.makedirs(d)
        except OSError:
            pass
    return


def get_params():
    parser = argparse.ArgumentParser()

    ## Common Parameters ##
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='G/C/S (GTA5/Cityscapes/Synthia)')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--n_class', type=int, default=19)
    parser.add_argument('--super_class', type=bool, default=False)
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--imsize', default=(1024, 512), help='the height of the input image to networks')
    parser.add_argument('--iter', type=int, default=10000000, help='number of iterations to train for')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--manualSeed', type=int, default=5688, help='manual seed')
    parser.add_argument('--ex', help='Experiment name')
    parser.add_argument('--logfile', type=str, help='Log file name (including path)')
    parser.add_argument('--tensor_freq', type=int, default=10, help='frequency of showing results on tensorboard during training.')
    parser.add_argument('--eval_freq', type=int, default=100, help='frequency of evaluation during training.')
    parser.add_argument('--train_T_with_source', type=bool, default=True)
    parser.add_argument('--CADT', type=bool, default=False)

    ## Optimizers Parameters ##
    parser.add_argument('--lr_dra', type=float, default=0.001)
    parser.add_argument('--lr_clf', type=float, default=5e-4)
    parser.add_argument('--lr_seg', type=float, default=2.5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_step', type=int, default=20000)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--weight_decay_task', type=float, default=5e-4)

    ## Saved model and images and checkpoint paths ##
    parser.add_argument('--load_networks_step', type=int, help="input the iteration of trained networks")

    opt = parser.parse_args()
    check_dirs(['checkpoint/' + opt.ex])
    opt.logfile = './checkpoint/' + opt.ex + '/' + opt.ex + '.log'
    check_dirs(['data_list/GTA5', 'data_list/Cityscapes'])
    # make_list()
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    return opt
