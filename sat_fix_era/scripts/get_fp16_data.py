import numpy as np
import time
import glob
from skimage.transform import rescale, resize

statis = np.load('pangu_fp16_statis.npz')
m = np.ones((241*281, 69)) * statis['mean']
s = np.ones((241*281, 69)) * statis['std']
mean = (m.T).reshape((69, 241, 281))
std  = (s.T).reshape((69, 241, 281))


res = glob.glob('72/*/*npz')
res.sort()


def load(sur, upp):
    sur[0] = sur[0]/10000.0
    upp[0] = upp[0]/10000.0
    N, C, W, H = upp.shape
    upp = upp.reshape((N*C, W, H))

    obs = np.concatenate((sur, upp), axis=0)
    obs = (obs - mean) / std
    obs = resize(obs, (69, 256, 256))
    obs = obs.astype(np.float16)

    return obs


base = 0
num = 0
for x in res:
    name = x.split('/')[-1]
    print(name[:-4])
    #r = time.strptime(name[:-4], '%Y-%m-%dT%H')
    #if r.tm_year==2020 and r.tm_mday>=28 and r.tm_mday < 30:
    #print(r)
    data = np.load(x)
    pre = load(data['pre_sur'], data['pre_upper'])
    #obs = load(data['obs_sur'], data['obs_upper'])
    #print(pre)
    #print(obs)


    fname = './pangu_72_fp16/%s.npy' % name[:-4]
    print(fname)
    np.save(fname, pre)
    break
