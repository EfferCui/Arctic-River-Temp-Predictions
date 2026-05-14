# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:21:05 2025

@author: sxc6234
"""

import timeit

start = timeit.default_timer()
from pathlib import Path
from typing import Dict
import random 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
#import seaborn as sns
from neuralhydrology.datasetzoo import get_dataset, camelsus
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import pandas as pd
#import geopandas as gpd
import numpy as np
#import shapely
import matplotlib.pyplot as plt

import numpy as np
torch.cuda._lazy_init() 

#%%

#import contextily as cx
config_file=Path("./Data/Hyperparameter_{}.yml".format(18))
cudalstm_config = Config(config_file)

# create a new model instance with random weights
cuda_lstm = CudaLSTM(cfg=cudalstm_config)

run_dir = Path("./runs/Final_20260211_1205_132934/")

# load the trained weights into the new model.
model_path = run_dir / 'model_epoch008.pt'

model_weights = torch.load(str(model_path), map_location=torch.device('cuda:0'))
#%%
cuda_lstm.load_state_dict(model_weights)
cuda_lstm = cuda_lstm.to('cuda:0')
cuda_lstm.eval()
torch.backends.cudnn.enabled = False
for param in cuda_lstm.parameters():
    param.requires_grad_(False)

wshds=['11', '127', '128', '15', '167', '169', '170', '19', '210', '211', '213', '215', '224', '229', '232', '238', '240', '243',\
       '244', '246', '253', '257', '258', '272', '275', '281', '282', '297', '3', '302', '308', '31', '318', '322', '325', '48', \
           '5', '50', '54', '61', '67', '8']
    
random.seed(10)   
# generate datetime series from '2023-01-01' to '2023-12-31' with daily frequency
date = pd.date_range(start='2002-01-01', end='2022-09-01', freq='D')
date=pd.DataFrame(date)
summer = date[(date[0].dt.month >= 6) & (date[0].dt.month <= 8)]
device = 'cuda:0'



idx_l=[]
for i in range(100): #20 time slots 
    idx= random.choice(list(summer.index.values))
    print(idx)   
    idx_l.append(idx)
    
eg_arr = np.zeros([365,42,len(idx_l),12])

scaler = load_scaler(run_dir)
    
for j in range(42):  #10 watersheds 
    wshd=wshds[j]
    print(wshd)
    
    # load the dataset
    dataset = get_dataset(cudalstm_config, basin=wshd, is_train=False, period='test', scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    samples = list(dataloader)
  
    for h in range(len(idx_l)):
        idx=idx_l[h]
        sample = samples[idx]
        feature_keys = list(sample['x_d'].keys())
        x = torch.cat([sample['x_d'][k] for k in feature_keys], dim=-1).to('cpu').numpy().reshape(365,12)
    
        n_iter =100 # higher number takes longer but produces better results
        alpha = torch.linspace(0, 1, n_iter, device=device).view(n_iter, 1)
        #alpha = alpha.view(n_iter, *tuple(np.ones(baseline[0].ndim, dtype='int')))
        
        for l in range(12):
            
            scaled_x = torch.cat([sample['x_d'][k] for k in feature_keys], dim=-1).to(device)
            scaled_x = scaled_x.expand(n_iter, -1, -1).clone()
            scaled_x[:,:,l] = scaled_x[:,:,l] * alpha
            scaled_x.requires_grad_(True)

            scaled_x_d = {
                k: scaled_x[:, :, m:m+1]
                for m, k in enumerate(feature_keys)
            }

            s = {
                'x_d': scaled_x_d,
                'y': sample['y'],
                'date': sample['date'],
                'x_s': sample['x_s'].to(device).expand(n_iter, -1)
            }

            part_y_hat = cuda_lstm(s)['y_hat']

            t0 = torch.autograd.grad(part_y_hat[:, -1, 0].sum(), scaled_x)
            attribution = (t0[0])[:, :, l].to('cpu').numpy().T
                
            
            integrated = attribution.sum(axis=1) / n_iter
            eg=np.multiply(integrated ,x[:,l])
            eg_arr[:,j,h,l]=eg#.reshape(365,1,1,1)
np.save('./runs/Final_20260211_1205_132934/ig.npy', eg_arr)
