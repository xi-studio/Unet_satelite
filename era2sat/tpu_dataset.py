import torch
import os
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
            data  = np.load(self.list[index])
            sat = data['sat']
            obs = data['obs']

            #sat = sat.astype(np.float16)
            #obs = obs.astype(np.float16)
        else:
            sat = np.ones((10, 256, 256), dtype=np.float16)
            obs = np.ones((69, 256, 256), dtype=np.float16)

        return obs, sat

    def __len__(self):
        return len(self.list)

if __name__ == '__main__':
    filename = np.load('data/train/train.npy')
    #filename = np.arange(1000)
    #a = Radars(filenames=filename, fake=True)
    a = Radars(filenames=filename, fake=False)

    train_loader = DataLoader(a, batch_size=16, shuffle=True, num_workers=4)
    for x in train_loader:
        print(x[0].shape, x[1].shape)
        print(x[0].dtype, x[1].dtype)
        break
