import argparse
import os
import torch

from Autoformer.utils.tools import dotdict
from Autoformer.exp.exp_main import Exp_Main
import random
import numpy as np

from GridSearchTransformer import GridSearchTransformer


if __name__ == "__main__":
    model_name="Transformer"
    features= 4# số cột dữ liệu
    seq_len = 24#bước thời gian
    d_ff=[512,1024,2048]
    d_model=[256,512,1024]
    e_layers= [2,3]
    d_layers= [1,2]
    label_len = [12,24,48] 
    n_heads=[8]#default 8

    RUN_MAKE_CONFIGMODEL = True
    RUN_MAKE_CONFIGMODEL = False
    #
    fileCfgModel = "cfgmodels_"+model_name+"_ts"+str(seq_len)+"_f"+str(features)+".csv"

    mscv = GridSearchTransformer(model_name=model_name,features=features,d_ffs=d_ff,d_models=d_model,e_layers=e_layers,d_layers=d_layers, timesteps=seq_len,start_lens=label_len,n_heads=n_heads)
    if RUN_MAKE_CONFIGMODEL == True:
        mscv.saveConfigModelToFile(fileCfgModel)
    else:
        mscv.loadConfigModels(fileCfgModel)
        best_model,best_cfgmodel,mse,mae,rmse,r2 = mscv.fit()
    #
    #     best_model.summary()
    #
    #     # In ra các tham số tốt nhất
    #     # print(f"Số tổ hợp có thể thử nghiệm: {total_combinations}")
        print(f'best_cfgmodel: {best_cfgmodel}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-squared (R²): {r2}')
