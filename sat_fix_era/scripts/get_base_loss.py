import glob
import numpy as np
import time
import os

#res = glob.glob('/home/tree/data/pangu_72_fp16/*.npy')
#res.sort()
#res = np.load('../data/train_72/test.npy')
res = np.load('../data/train_72/train.npy')

mlist = []
for x in res:
    print(x)
    
    fname = '../' + x 

    data = np.load(fname)
    pre = data['pre'][10:14]
    obs = data['obs']
    a = np.abs(pre-obs)
    a = a.reshape((4, 256*256))
    loss = np.mean(a, axis=1)
    #loss = np.mean(np.abs(pre-obs))
    print(loss)
#    if os.path.isfile(fname):
#        data = np.load(fname)
#        sate = data['sat']
#        obs  = data['obs'][:4]
#    
#        input_data = np.concatenate((sate, pre), axis=0)
#    
#        np.savez('../data/pan_72_train/%s.npz' % name[:-4], pre=input_data, obs=obs)
#

