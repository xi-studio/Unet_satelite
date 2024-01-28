import numpy as np
import glob
from skimage.transform import rescale, resize


statis = np.load('../data/pangu_fp16_statis.npz')
m = np.ones((241*281, 69)) * statis['mean']
s = np.ones((241*281, 69)) * statis['std']
mean = (m.T).reshape((69, 241, 281))
std  = (s.T).reshape((69, 241, 281))


#res = glob.glob('/home/tree/data/fp16/*npz')
#res.sort()
res = np.load('../data/pan_fp16_meta.npy')

for x in res:
    print(x[0])
    obs = np.load(x[0])
    sur = obs['sur']
    upp = obs['upper']

    N, C, W, H = upp.shape
    upp = upp.reshape((N*C, W, H))

    obs = np.concatenate((sur, upp), axis=0)
    obs = (obs - mean) / std

    obs = resize(obs, (69, 256, 256))
    obs = obs.astype(np.float16)

    sate = np.load(x[1])
    sate = np.nan_to_num(sate, nan=255)
    sate = (sate - 180.0) / (375.0 - 180.0)
    sate = resize(sate, (10, 256, 256))
    sate = sate.astype(np.float16)
    
    name = x[0].replace('/home/tree/data/fp16/', '../data/fp16/')
    np.savez(name, sat=sate, obs=obs)
    
