import torch
import random
import numpy as np
from trainer import Trainer
from util.config import cfg

def main():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
