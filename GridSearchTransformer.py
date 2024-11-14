import itertools
import json
import math
import random

import numpy as np
import torch
from keras import Sequential
from keras.src.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import csv

from Autoformer.exp.exp_main import Exp_Main
from Autoformer.utils.tools import dotdict
from multiprocessing import freeze_support



class GridSearchTransformer():

    def __init__(self, model_name, features,timesteps, d_ffs,d_models,e_layers,d_layers,start_lens,n_heads):
        # Tạo tổ hợp
        self.model_name = model_name
        self.timesteps = timesteps
        self.features = features
        self.list_models = []
        self.list_cfgmodels = []
        self.list_modelargs = []
        self.list_modelsettings = []
        self.combinations = []
        total_combinations = 0

        list_d_ffs = [list(combo) for combo in itertools.product(d_ffs, repeat=1)]
        list_d_models = [list(combo) for combo in itertools.product(d_models, repeat=1)]
        list_e_layers = [list(combo) for combo in itertools.product(e_layers, repeat=1)]
        list_d_layers = [list(combo) for combo in itertools.product(d_layers, repeat=1)]
        list_start_lens = [list(combo) for combo in itertools.product(start_lens, repeat=1)]
        list_n_heads = [list(combo) for combo in itertools.product(n_heads, repeat=1)]
        # In kết quả
        self.combinations.extend(list(itertools.product(list_d_ffs, list_d_models,list_e_layers,list_d_layers,list_start_lens,list_n_heads)))
        print("Tổng số model:",len(self.combinations))

    def saveConfigModelToFile(self,fileCfgModel):
        dataCfgModel = [
            ['no','timesteps','features','d_ffs', 'd_models', 'e_layers','d_layers','start_lens','n_heads','mae','mse','rmse','r2']
        ]
        # for i in range(0, len(self.layers)):
        # nlayers = self.layers[i]
        i=1
        for combo in self.combinations:
                # print(combo)
                itemcfg =[]
                itemcfg.append(i)
                itemcfg.append(self.timesteps)
                itemcfg.append(self.features)
                itemcfg.append(combo[0][0])
                itemcfg.append(combo[1][0])
                itemcfg.append(combo[2][0])
                itemcfg.append(combo[3][0])
                itemcfg.append(combo[4][0])
                itemcfg.append(combo[5][0])
                dataCfgModel.append(itemcfg)
                i = i + 1

        with open(fileCfgModel, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(dataCfgModel)
    # def genModels(self, timesteps, features, layers):
    #     for i in range(0, len(layers)):
    #         nlayers = layers[i]
    #         for combo in self.combinations:
    #             # print(combo)
    #             self.createModel(timesteps, features, nlayers, combo[0], combo[1])

    def loadConfigModels(self, fileCfgModel):
        self.file_cfg_models = fileCfgModel
        df = pd.read_csv(fileCfgModel)
        # ['no', 'timesteps', 'features', 'd_ffs', 'd_models', 'e_layers', 'd_layers', 'start_lens', 'n_heads', 'mae',
        #  'mse', 'rmse', 'r2']

        for index, row in df.iterrows():
            no = int(row['no'])
            timesteps = int(row['timesteps'])
            features = int(row['features'])
            d_ff = int(row['d_ffs'])
            d_model = int(row['d_models'])
            e_layer = int(row['e_layers'])
            d_layer = int(row['d_layers'])
            start_len = int(row['start_lens'])
            n_head = int(row['n_heads'])
            r2 = row["r2"]
            # print(row)
            if  math.isnan(r2) or r2 == '':
                self.createModel(no, timesteps, features, d_ff, d_model, e_layer,d_layer,start_len,n_head)
            else:
                print("Model No:"+str(no) +" da chay!")
    def saveResultToFile(self, fileCfgModel, rowNum, mae, mse, rmse, r2):
        df = pd.read_csv(fileCfgModel)
        index = df['no'] == rowNum
        # Bước 2: Chỉnh sửa cột trong một dòng (ở đây chỉnh sửa dòng có chỉ số là 0)
        # Ví dụ chỉnh sửa các cột 'column1' và 'column2'
        df.loc[index, 'mae'] =  mae  # Sửa giá trị cột 'column1'
        df.loc[index, 'mse'] = mse  # Sửa giá trị cột 'column2'
        df.loc[index, 'rmse'] = rmse
        df.loc[index, 'r2'] = r2

        # Bước 3: Ghi lại file CSV đã chỉnh sửa
        df.to_csv(fileCfgModel, index=False)

        print("update "+fileCfgModel+" thành công!")

    def createModel(self, no, timesteps, features, d_ff, d_model, e_layer, d_layer, start_len, n_head):

        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        args = dotdict()
        args.des = 'test'

        args.num_workers = 1
        args.gpu = 0

        args.devices = '0'
        args.use_gpu = False
        args.use_multi_gpu = False
        args.is_training = True
        args.freq = 'h'
        args.checkpoints = './checkpoints/'

        # forecasting task
        args.seq_len = timesteps  # input sequence length default=96
        args.label_len = start_len  # start token length default=48
        args.pred_len = 1  # prediction sequence length default=96

        # model define
        args.embed = 'timeF'  # time features encoding, options:[timeF, fixed, learned]
        args.enc_in = features
        args.dec_in = features
        args.c_out = 1
        args.target = "MN_KimLong"
        args.dropout = 0.05  # default=0.05
        args.bucket_size = 4  # for Reformer
        args.n_hashes = 4  # for Reformer
        args.e_layers = e_layer  # num of encoder layers,default=2
        args.d_layers = d_layer  # num of decoder layers, default=1
        args.n_heads = n_head  # num of heads, default=8
        args.factor = 1  # attn factor,default=1
        args.d_model = d_model  # dimension of model, default=512
        args.des = 'Exp'
        args.d_ff = d_ff  # dimension of fcn, default=2048
        args.moving_avg = 25  # window size of moving average,default=25
        if timesteps < 24:
            args.moving_avg = timesteps
            args.label_len = timesteps
        args.distil = True  # whether to use distilling in encoder, using this argument means not using distilling,default=True
        args.output_attention = False

        # optimization
        args.itr = 1  # experiments times
        args.patience = 3  # early stopping patience
        args.learning_rate = 0.0001  # optimizer learning rate,default=0.0001
        args.batch_size = 32
        args.activation = 'gelu'
        args.use_amp = False
        args.loss = 'mse'
        args.train_epochs = 1
        args.lradj = 'type1'  # 'adjust learning rate'

        args.root_path = 'X:\ATMEL\pythonProject'
        args.data_path = 'songhuong_mice.csv'
        args.model_id = self.model_name+str(no)
        args.model = self.model_name
        # args.model = 'Informer'
        args.data = 'custom'
        args.features = 'MS'

        print('Args in experiment:')
        # print(args)

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        print('Args in experiment:')
        print(args)

        Exp = Exp_Main
        for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)

                # exp = Exp(args)  # set experiments
                # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                # exp.train(setting)

                # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # mae, mse, rmse, r2 = exp.test(setting)
                # torch.cuda.empty_cache()


        # self.list_models.append(exp)
        self.list_modelargs.append(args)
        self.list_modelsettings.append(setting)
        model_config = {
            "no":no, "timesteps":timesteps, "features":features, "d_ff":d_ff, "d_model":d_model, "e_layer":e_layer, "d_layer":d_layer, "start_len":start_len, "n_head":n_head
        }

        # Chuyển đổi dictionary thành chuỗi JSON
        # json_str = json.dumps(model_config, indent=4)
        self.list_cfgmodels.append(model_config)
        return


    def fit(self):

            self.min_score = 100000
            self.best_model = None
            self.best_cfgmodel = None
            mse = 0
            mae=0
            rmse=0
            r2=0
            for i in range(0, len(self.list_cfgmodels)):
                try:
                    # model = self.list_models[i]
                    Exp = Exp_Main
                    model = Exp(self.list_modelargs[i])

                    cfgmodel = self.list_cfgmodels[i]
                    setting = self.list_modelsettings[i]
                    print(f'Run : {i}', cfgmodel)

                    model.train(setting)

                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    mae, mse, rmse, r2 = model.test(setting)
                    torch.cuda.empty_cache()


                    print(f'Mean Squared Error (MSE): {mse}')
                    print(f'Root Mean Squared Error (RMSE): {rmse}')
                    print(f'Mean Absolute Error (MAE): {mae}')
                    # print(f'Mean Absolute Percentage Error (MAPE): {mape}')
                    print(f'R-squared (R²): {r2}')
                    # print(f'Symmetric Mean Absolute Percentage Error (SMAPE): {smape}')
                    if mse < self.min_score:
                        self.min_score = mse
                        self.best_model = model
                        self.best_cfgmodel = cfgmodel
                        print(f'BEST Mean Squared Error (MSE): {mse}')
                        print(f'BEST cfgmodel: {cfgmodel}')
                    # In ra kết quả các chỉ số đánh giá
                    self.saveResultToFile(self.file_cfg_models, int(cfgmodel['no']), mae,mse,rmse,r2)
                    freeze_support()
                except Exception as e:
                    print(f'ExceptionType', e)
                    self.saveResultToFile(self.file_cfg_models, int(cfgmodel['no']), 0, 0, 0, 0)

            return self.best_model,self.best_cfgmodel,mse,mae,rmse,r2

