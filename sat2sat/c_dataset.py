import torch
import os
import time
from skimage.transform import rescale, resize
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor
import gcsfs

class Radars(Dataset):
    def __init__(self, filenames, fake=False):
        super(Radars, self).__init__()

        self.list = filenames 
        self.fake = fake

    def __getitem__(self, index):

        if self.fake!=True:
            t0 = np.load(self.list[index][0][3:])
            t1 = np.load(self.list[index][1][3:])
            t2 = np.load(self.list[index][2][3:])

            now = (self.list[index][0].split('/')[-1][:-4]).split('_')[2]
            r = time.strptime(now,'%Y%m%d%H%M')

            cond = np.zeros((1,40), dtype=np.float16)

            t_mon = r.tm_mon
            t_day = r.tm_hour
            cond[:, t_mon] = 1
            cond[:, t_day+12] = 1

            sat = np.concatenate((t0, t1), axis=0)
            sat = sat.astype(np.float16)
            obs = t2.astype(np.float16)
        else:
            sat = np.ones((20, 256, 256), dtype=np.float16)
            obs = np.ones((10, 256, 256), dtype=np.float16)
            cond = np.ones((1,40), dtype=np.float16)

        return sat, obs, cond

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    #filename = np.load('data/train/train.npy')
    filename = np.arange(1000)
    a = Radars(filenames=filename, fake=True)
    #a = Radars(filenames=filename, fake=False)

    train_loader = DataLoader(a, batch_size=16, shuffle=True, num_workers=4)
    for x in train_loader:
        print(x[0].shape, x[1].shape, x[2].shape)
        print(x[0].dtype, x[1].dtype, x[2].dtype)
        break
