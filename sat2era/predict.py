import torch
import argparse
import os
import re
import yaml
from tqdm import tqdm
from skimage.transform import rescale, resize

import numpy as np
import PIL
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from accelerate import Accelerator
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

from c_dataset import Radars
from unet import UNetModel


def training_function(args, config):
    epoch_num     = args.num_epoch
    batch_size    = args.batch_size 
    num_workers   = args.num_workers
    batch_size    = args.batch_size 
    fake          = args.fake
    log_time      = args.log_time
    learning_rate = config['lr']

    if args.fake == False:
        test_files = np.load(args.filenames + '/test.npy')

        
    test_ds = Radars(test_files, fake) 

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = UNetModel(config)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator(log_with="all", project_dir='logs_era')
    model, optimizer, test_loader = accelerator.prepare(model, optimizer, test_loader)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load("./logs_era/checkpoint_0129/best/pytorch_model.bin"))

    for epoch in range(epoch_num):
        unwrapped_model.eval()
        accurate = 0
        num_elems = 0
        for _, (x, y, c) in enumerate(test_loader):
            with torch.no_grad():
                out = unwrapped_model(x, c)
                a = torch.abs(out - y)
                a = torch.mean(a.view(69, -1), axis=1)

                print(a.shape)
                #loss = criterion(out, y)
                num_elems += 1
                accurate = accurate + a
    
        eval_metric = accurate / num_elems
        np.save('sat2era_69_channel_loss.npy', eval_metric.cpu().numpy())
        print(eval_metric)




def main(): 
    parser = argparse.ArgumentParser('Era5 to satelite Unet', add_help=True)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--batch_size', type=int, default=1, help="batch size for single GPU")
    parser.add_argument('--num_epoch', type=int, default=2, help="epochs")
    parser.add_argument('--num_workers', type=int, default=8, help="num workers")
    parser.add_argument('--log_time', type=str, default='1222', help="log time name")
    parser.add_argument('--filenames', type=str, help="data filenames")
    parser.add_argument('--fake', type=bool, default=False, help="if fake data")
    args, unparsed = parser.parse_known_args()

    with open(args.cfg, "r") as f:
        res = yaml.load(f, Loader=yaml.FullLoader)

    config = res[0]['config']

    training_function(args, config)

if __name__ == '__main__':
    main()
