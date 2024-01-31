import glob
import numpy as np
import time
import os

res = glob.glob('/home/tree/data/pangu_72_fp16/*.npy')
res.sort()

mlist = []
for x in res:
    print(x)
    name = x.split('/')[-1]

    fname = '../../era2sat/data/fp16/%s.npz' % name[:-4]

    pre = np.load(x)
    if os.path.isfile(fname):
        data = np.load(fname)
        sate = data['sat']
        obs  = data['obs'][:4]
    
        input_data = np.concatenate((sate, pre), axis=0)
    
        np.savez('../data/pan_72_train/%s.npz' % name[:-4], pre=input_data, obs=obs)


