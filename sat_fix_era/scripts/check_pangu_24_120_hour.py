import torch
import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr
import dask
import numpy as np
from typing import OrderedDict
import yaml
import time
from skimage.transform import rescale, resize


model_24 = onnx.load('pangu_weather_24.onnx')

options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = 1

cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

statis = np.load('pangu_fp16_statis.npz')
m = np.ones((241*281, 69)) * statis['mean']
s = np.ones((241*281, 69)) * statis['std']
mean = (m.T).reshape((69, 241, 281))
std  = (s.T).reshape((69, 241, 281))

def load(sur, upp):
    sur = sur[:, 30*4:90*4+1, 70*4:140*4+1]
    upp = upp[:, :, 30*4:90*4+1, 70*4:140*4+1]
    sur[0] = sur[0]/10000.0
    upp[0] = upp[0]/10000.0
    N, C, W, H = upp.shape
    upp = upp.reshape((N*C, W, H))

    obs = np.concatenate((sur, upp), axis=0)
    obs = (obs - mean) / std
    obs = resize(obs, (69, 256, 256))
    obs = obs.astype(np.float32)

    return obs

def check(obs_sur, obs_upp, pre_sur, pre_upp):
    obs = load(obs_sur, obs_upp)
    pre = load(pre_sur, pre_upp)
    a = np.abs(obs - pre)
    a = a.reshape((69, 256*256))

    loss = np.mean(a, axis=1)

    return loss



def load_vars(var_config_file):
    with open(var_config_file, "r") as f:
        var_config = yaml.load(f, Loader=yaml.Loader)

    return var_config

def main():

    ar_full_37_1h = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
    )
    print("Model surface dataset size {} TiB".format(ar_full_37_1h.nbytes/(1024**4)))

    variables = OrderedDict(load_vars('./pangu_config.yaml'))
    surface_data = ar_full_37_1h[variables['input']['surface']]
    high_data = ar_full_37_1h[variables['input']['high']].sel(level=variables['input']['levels'])
    sur  = surface_data.sel(time=slice('2020-01-01', '2020-01-31'))
    hig  = high_data.sel(time=slice('2020-01-01', '2020-01-31'))
    sur_dat = xr.concat([sur['mean_sea_level_pressure'],sur['10m_u_component_of_wind'],sur['10m_v_component_of_wind'], sur['2m_temperature']],"var")
    hig_dat = xr.concat([hig['geopotential'],hig['specific_humidity'],hig['temperature'],hig['u_component_of_wind'],hig['v_component_of_wind']],'var')

    res_sur  = sur_dat.transpose('time','var','latitude','longitude')
    res_high = hig_dat.transpose('time','var','level','latitude','longitude')


    list_24 = []
    list_48 = []
    list_72 = []
    list_96 = []
    list_120 = []
    for i, _ in enumerate(res_sur[:-80]):
        now = str(res_sur[i]['time'].to_numpy())
        now = now.split(":")[0]
        print(now)

        sur = res_sur[i].to_numpy()
        high = res_high[i].to_numpy()

        input = high.astype(np.float32)
        surface = sur.astype(np.float32)
        output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':surface})
        pre_sur_24 = output_surface
        pre_upp_24 = output
        output, output_surface = ort_session_24.run(None, {'input':output, 'input_surface':output_surface})
        pre_sur_48 = output_surface
        pre_upp_48 = output
        output, output_surface = ort_session_24.run(None, {'input':output, 'input_surface':output_surface})
        pre_sur_72 = output_surface
        pre_upp_72 = output
        output, output_surface = ort_session_24.run(None, {'input':output, 'input_surface':output_surface})
        pre_sur_96 = output_surface
        pre_upp_96 = output
        output, output_surface = ort_session_24.run(None, {'input':output, 'input_surface':output_surface})
        pre_sur_120 = output_surface
        pre_upp_120 = output


        sur24 = res_sur[i+24].to_numpy()
        high24 = res_high[i+24].to_numpy()

        sur48 = res_sur[i+48].to_numpy()
        high48 = res_high[i+48].to_numpy()

        sur72 = res_sur[i+72].to_numpy()
        high72 = res_high[i+72].to_numpy()

        sur96 = res_sur[i+96].to_numpy()
        high96 = res_high[i+96].to_numpy()

        sur120 = res_sur[i+120].to_numpy()
        high120 = res_high[i+120].to_numpy()

        loss_24 = check(sur24, high24, pre_sur_24, pre_upp_24)
        list_24.append(loss_24)
        print('24hour:', loss_24)
        loss_48 = check(sur48, high48, pre_sur_48, pre_upp_48)
        list_48.append(loss_48)
        print('48hour:', loss_48)
        loss_72 = check(sur72, high72, pre_sur_72, pre_upp_72)
        list_72.append(loss_72)
        print('72hour:', loss_72)
        loss_96 = check(sur96, high96, pre_sur_96, pre_upp_96)
        list_96.append(loss_96)
        print('96hour:', loss_96)
        loss_120 = check(sur120, high120, pre_sur_120, pre_upp_120)
        list_120.append(loss_120)
        print('120hour:', loss_120)

        if i > 10:
            break

    mean_24  = np.mean(np.array(list_24), axis=0)
    mean_48  = np.mean(np.array(list_48), axis=0)
    mean_72  = np.mean(np.array(list_72), axis=0)
    mean_96  = np.mean(np.array(list_96), axis=0)
    mean_120  = np.mean(np.array(list_120), axis=0)

    np.save('./mean/mean_24.npy', mean_24)
    np.save('./mean/mean_48.npy', mean_48)
    np.save('./mean/mean_72.npy', mean_72)
    np.save('./mean/mean_96.npy', mean_96)
    np.save('./mean/mean_120.npy', mean_120)


if __name__ == '__main__':
    main()

#    import pandas as pd

#array =np.ones(10)
#df = pd.DataFrame (array)

#filepath = 'my_excel_file.xlsx'

#df.to_excel(filepath, index=False)
