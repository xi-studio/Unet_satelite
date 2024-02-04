import numpy as np
import glob
from skimage.transform import rescale, resize
import time



res = glob.glob('/home/tree/data/satelite/*.npy')
res.sort()

for x in res:
    try:
        sate = np.load(x)
        sate = np.nan_to_num(sate, nan=255)
        sate = (sate - 180.0) / (375.0 - 180.0)
        sate = resize(sate, (10, 256, 256))
        sate = sate.astype(np.float16)
        
        name = x.replace('/home/tree/data/satelite/', '../data/satelite_fp16/')
        np.save(name, sate)
        print(name)
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")
        
    #np.savez(name, sat=sate, obs=obs)
    
