import glob
import numpy as np
import time
import os

res = glob.glob('../data/fp16/*.npz')
res.sort()

train_list = []
valid_list = []
test_list = []

for x in res:
    print(x)
    name = x.split('/')[-1]
    print(name[:-4])

    r = time.strptime(name[:-4], '%Y-%m-%dT%H')
    print(r)
    if r.tm_year==2020 and r.tm_mday>=30:
        test_list.append(x[3:])
    elif r.tm_year==2020 and r.tm_mday>=28 and r.tm_mday < 30:
        valid_list.append(x[3:])
    else:
        train_list.append(x[3:])

np.save('../data/train/train.npy', np.array(train_list))
np.save('../data/train/test.npy', np.array(test_list))
np.save('../data/train/valid.npy', np.array(valid_list))

