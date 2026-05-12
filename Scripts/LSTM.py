# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:45:43 2023

@author: sxc6234
"""

#%%
import os
from pathlib import Path
import torch
import os
from scipy import stats
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run
import os
os.chdir("./")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    import gc
    gc.collect()
#%%
#Train the LSTM model using the configuration file
if __name__ == '__main__':
        # by default we assume that you have at least one CUDA-capable NVIDIA GPU
    if torch.cuda.is_available():
        print("yes gpu")
        torch.device('cuda:0')
        start_run(config_file=Path("./Data/Hyperparameter_{}.yml".format(18)))



#%%
#Evaluate the model on both train and test sets

run_dir = Path('./runs/Final_20260211_1205_132934/')
eval_run(run_dir=run_dir, period="train")
eval_run(run_dir=run_dir, period="test")
