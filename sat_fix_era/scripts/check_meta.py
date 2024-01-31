import glob
import numpy as np
import time
import os

res = glob.glob('/home/tree/data/pangu_72_fp16/*.npy')
res.sort()

base_dir = '/home/tree/data/satelite/H09_IR_%s_china_0p25.npy'

mlist = []
for x in res:
    print(x)
    name = x.split('/')[-1]

    r = time.strptime(name[:-4], '%Y-%m-%dT%H')
    sname = time.strftime('%Y%m%d%H%M', r)
    print(sname)
    filename = base_dir % sname
    if os.path.isfile(filename):
        y = np.load(filename)
        print(y.shape)
        mlist.append((x, filename))

np.save('data/pangu_72_fp16_meta.npy', np.array(mlist))
