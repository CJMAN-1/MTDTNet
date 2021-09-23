from __future__ import print_function
from param import get_params
# from full_trainer import Trainer
from full_trainer2 import Trainer

if __name__ == '__main__':
    opt = get_params()
    trainer = Trainer(opt)
    trainer.train()
