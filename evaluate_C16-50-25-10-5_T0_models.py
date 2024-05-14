 
import sys
import os
import glob
import numpy as np
import csv
import math
import time

from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback, TensorBoard
from keras import backend as K
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D
from keras.models import Model

from keras.utils import plot_model
import tensorflow as tf
import SimpleITK as sitk
import matplotlib 
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('/home/gino/projects/4D_MRI_CNN')
import preprocessing.preprocess as prep
import preprocessing.utils as utils

from tqdm import tqdm
import json

from data_generation import DataGenerator
from data_generation import DataGenerator2D_aug
from data_generation import grab_samples_seq
from net import UNET3D
from net import CNN
from net import Baseline_Nav_3D


### ===========================
### define GPU to be used
### ===========================
# GPU = "2"
# if len(sys.argv) == 2:
#     GPU = sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU

train_data_path = '/cache/gino/Data_for_DeepLearning/'
data_dir        = os.path.join(train_data_path, 'test')  
model_main_path = '/data/gino/4D_MRI_CNN/output/models'
model_IDs_csv   = '/home/gino/projects/4D_MRI_CNN/experiments/transfer_learning_experiments/models_C16-[5-50]_T0.csv'
target_cases_csv    = '/home/gino/projects/4D_MRI_CNN/experiments/transfer_learning_experiments/all_target_cases.csv'
out_path        = '/home/gino/projects/4D_MRI_CNN/experiments/transfer_learning_experiments'
#IDs = ['ugfqw5op', '33u6gxja', 'y3k8yg30', 'xads9fdo'] # test


if __name__ == '__main__':
    start_time = time.time()
    ### get model IDs from csv
    IDs_dict = {}
    with open(model_IDs_csv, 'r', newline='') as csvfile:    
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            IDs_dict[row['ID']] = row['Name']         
    IDs = list(IDs_dict.keys())
    
    target_cases_list = []
    with open(target_cases_csv, 'r', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            target_cases_list.append(row['target_case'])         

        
    result_list = []
    for ID in IDs: 
        for target_case in target_cases_list:
            ### find model with ID
            model_path, json_path, err = utils.find_model_with_ID(model_main_path, ID)
            if err != 0:
                print("!!!!!!!!!!!!!! model ID {} not found".format(ID))
                exit()
                
            pred_cases =  target_case
            if type(pred_cases) == str:
                pred_cases = [pred_cases]
                
            ### load model
            hps   = utils.load_experiment_json(json_path)
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
                    seq_half = int(len(ids_dict[i])/2)+1
                    ids = ids + ids_dict[i][seq_half:]  ### long
                    # ids = ids + ids_dict[i][-3:]   ### short
            else:
                ids = ids_temp
            
            batch_size  = 1
            batch_num   = int(math.floor(len(ids) / batch_size))
            generator   = DataGenerator2D_aug(cases=pred_cases, data_dir=data_dir, IDs=ids,   size_X=hps['netInput_x'],  size_Y=hps['netInput_y'], spacing=hps['resolution'], 
                                            batch_size=batch_size, shuffle=False, comment='', log_dir='.', augmentation=False, use_explicit_dist_info=hps['use_dist_info'])
            
            print('Time: {}s Now calculating test data loss for {} with {} test samples'.format(time.time() - start_time, pred_cases, len(ids)))
            losses, computation_times, series_numbers, series_tps, sample_ids = utils.auto_evaluate(model, generator, batch_num=batch_num, step=1)
            print('validation data loss for model with ID: {} = {}'.format(ID, np.array(losses).mean()))
            for loss, comp_time, sn, stp, sid in zip(losses, computation_times, series_numbers, series_tps, sample_ids):
                result_list.append([ID, pred_cases[0], float(hps['split_train']), 0.0, IDs_dict[ID], loss, comp_time, sn, stp, sid])
    
    print('evaluation took {}s'.format(time.time() - start_time))
    
    csv_name = os.path.join(out_path, 'losses_C16-50-25-10-5_T0_long.csv')
    with open(csv_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['ID', 'target_case', 'source_train_size', 'target_train_size', 'model_name', 'target_loss', 'computation_time', 'series_number', 'series_timePoint', 'sample_id'])
        for row in result_list:
            csvwriter.writerow(row)