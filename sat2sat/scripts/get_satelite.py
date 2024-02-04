import numpy as np
import glob
from skimage.transform import rescale, resize
import time



res = glob.glob('../data/satelite_fp16/*.npy')
res.sort()

data_list = []
for i, x in enumerate(res[:-5]):
    name_t0 = x
    name_t1 = res[i+1]
    name_t2 = res[i+2]

    time_t0 = (name_t0.split('/')[-1][:-4]).split('_')[2]
    time_t1 = (name_t1.split('/')[-1][:-4]).split('_')[2]
    time_t2 = (name_t2.split('/')[-1][:-4]).split('_')[2]

    t0 = time.mktime(time.strptime(time_t0,'%Y%m%d%H%M'))
    t1 = time.mktime(time.strptime(time_t1,'%Y%m%d%H%M'))
    t2 = time.mktime(time.strptime(time_t2,'%Y%m%d%H%M'))
    if (t1 - t0) == 3600 and (t2 - t1) == 3600:
        data_list.append((name_t0, name_t1, name_t2))
        print(i)

np.save('../data/sat_time_meta.npy', np.array(data_list))

        


    
    

        
    
