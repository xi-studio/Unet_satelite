import glob
import numpy as np
import time
import os


data_list = np.load('../data/sat_time_meta.npy')
train_list = data_list[ : -4000]
valid_list = data_list[-4000 : -2000]
test_list  = data_list[-2000: ]

np.save('../data/train_sat/train.npy', np.array(train_list))
np.save('../data/train_sat/test.npy', np.array(test_list))
np.save('../data/train_sat/valid.npy', np.array(valid_list))

