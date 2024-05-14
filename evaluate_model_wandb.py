 
import sys
import os
import glob
import numpy as np
import csv
import math

import wandb

# from keras.optimizers import Adam
# from keras.callbacks import CSVLogger
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback, TensorBoard
# from keras import backend as K
# from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D
# from keras.models import Model

# from keras.utils import plot_model
# import tensorflow as tf
# import SimpleITK as sitk
import matplotlib 
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('/home/gino/projects/4D_MRI_CNN')
# import preprocessing.preprocess as prep
import preprocessing.utils as utils

# from tqdm import tqdm
# import json

# from data_generation import DataGenerator
# from data_generation import DataGenerator2D_aug
# from data_generation import grab_samples_seq
# from net import UNET3D
from net import CNN
# from net import Baseline_Nav_3D


### ===========================
### define GPU to be used
### ===========================
# GPU = "2"
# if len(sys.argv) == 2:
#     GPU = sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU

train_data_path = '/cache/gino/Data_for_DeepLearning/'
model_main_path = '/data/gino/4D_MRI_CNN/output/models'
out_path        = '/home/gino/projects/4D_MRI_CNN/experiments'
#IDs = ['ugfqw5op', '33u6gxja', 'y3k8yg30', 'xads9fdo'] # test


if __name__ == '__main__':
    
    ### get model IDs from csv
    wandb.init()
    hps = wandb.config
    ID  =  hps.modelID
    
    result_table = {}
    ### find model with ID
    model_path, json_path, err = utils.find_model_with_ID(model_main_path, ID)
    if err != 0:
        continue
        
    hps        = utils.load_experiment_json(json_path)
    pred_cases =  hps['cases_train']
    pred_cases = "2018-06-07_MDna"
    if type(pred_cases) == str:
        pred_cases = [pred_cases]
        
    data_dir = os.path.join(train_data_path, 'train')  

    ### load model
    model = CNN.make_model(hps)    
    model.load_weights(model_path)
    
    reg  = '*_N_*.nii.gz'
    ids_temp = []
    for case in pred_cases:
        ids_temp += sorted(glob.glob(os.path.join(data_dir, case, reg)))
                    
        if len(ids_temp) == 0:
            utils.print_attantion(title='No IDs found', text='Could not find IDs for: {}'.format(case))
    
    
    if True: ### grap second half of each sequence
        ids_dict = utils.groupFileNamesBySeriesNumber_dict(ids_temp)
        keys = sorted(ids_dict.keys())
        ids = []
        for i in ids_dict:            
            ids = ids + ids_dict[i][95:]  
    else:
        ids = ids_temp
    
    batch_size  = 50
    batch_num   = int(math.floor(len(ids) / batch_size))
    generator   = DataGenerator2D_aug(cases=pred_cases, data_dir=data_dir, IDs=ids,   size_X=hps['netInput_x'],  size_Y=hps['netInput_y'], spacing=hps['resolution'], 
                                        batch_size=batch_size, shuffle=False, comment='', log_dir='.', augmentation=False, use_explicit_dist_info=hps['use_dist_info'])
    
    print('calculating test data loss for {} with {} test samples'.format(pred_cases, len(ids)))
    loss = utils.auto_evaluate(model, generator, batch_num=batch_num, step=1)
    print('validation data loss {} = {}'.format(ID, loss))
    result_table[ID] = [ID, pred_cases[0], float(hps['split_train']), loss, IDs_dict[ID]]
            
    csv_name = os.path.join(out_path, 'losses_C1-50_T5.csv')
    with open(csv_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['ID', 'case', 'split_train', 'val_loss', 'model_name'])
        for i in result_table:
            csvwriter.writerow(result_table[i])