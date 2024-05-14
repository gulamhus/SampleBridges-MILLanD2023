 
import sys
import os
import glob
import numpy as np
import numpy.random as random
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

sampling        = 0.2  #20% of second half of the data
sampleSeq       = True # if False pseudo random sampling, if True sampling a continuous sequence
train_data_path = '/cache/gino/Data_for_DeepLearning/'
data_dir        = os.path.join(train_data_path, 'test')  
model_main_path = '/data/gino/4D_MRI_CNN/output/models'
model_IDs_csv   = '/home/gino/projects/4D_MRI_CNN/experiments/transfer_learning_experiments/models_C16-5_T[2-98].csv'
out_path        = '/home/gino/projects/4D_MRI_CNN/experiments/transfer_learning_experiments'
#IDs = ['ugfqw5op', '33u6gxja', 'y3k8yg30', 'xads9fdo'] # test


if __name__ == '__main__':
    start_time = time.time()
    random.seed(1337)
    
    ### get model IDs from csv
    IDs_dict = {}
    with open(model_IDs_csv, 'r', newline='') as csvfile:    
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            IDs_dict[row['ID']] = row['Name']         
    model_IDs = list(IDs_dict.keys())
    
    for m_num, model_ID in enumerate(model_IDs): 
        tf.keras.backend.clear_session()
        result_list = []
        model_start_time = time.time()
        
        ### find model with ID
        model_path, json_path, err = utils.find_model_with_ID(model_main_path, model_ID)
        if err != 0:
            continue
            
        hps        = utils.load_experiment_json(json_path)
        pred_cases =  hps['cases_train']
        if type(pred_cases) == str:
            pred_cases = [pred_cases]
            
        ### load model
        model = CNN.make_model(hps)    
        model.load_weights(model_path)
        
        reg  = '*_N_*.nii.gz'
        ids_temp = []
        for case in pred_cases:
            ids_temp += sorted(glob.glob(os.path.join(data_dir, case, reg)))
                        
            if len(ids_temp) == 0:
                utils.print_attantion(title='No IDs found', text='Could not find IDs for: {}'.format(case))
        
       
        ### sample from second half of each sequence
        ids_dict = utils.groupFileNamesBySeriesNumber_dict(ids_temp)
        keys = sorted(ids_dict.keys())
        ids = []
        for i in ids_dict:
            li       = ids_dict[i]
            seq_half = int(len(li)/2)+1
            start    = seq_half
            end      = len(li)-1
            size     = int(math.floor((end - start) * sampling))   
            if sampleSeq:  
                ids   = ids + li[start:start+size]  
            else:               
                sampling_indices = random.randint(start, end, size)
                li_np = np.array(li)            
                ids   = ids + list(li_np[sampling_indices])

        if m_num == 0:
            print('Now calculating test data loss for {}/{} modelID: {}, proband: {}, with {} test samples'.format(m_num+1, len(model_IDs), model_ID, pred_cases, len(ids)))
            
        batch_size  = 1
        batch_num   = int(math.floor(len(ids) / batch_size))
        generator   = DataGenerator2D_aug(cases=pred_cases, data_dir=data_dir, IDs=ids,   size_X=hps['netInput_x'],  size_Y=hps['netInput_y'], spacing=hps['resolution'], 
                                          batch_size=batch_size, shuffle=False, comment='', log_dir='.', augmentation=False, use_explicit_dist_info=hps['use_dist_info'])
        
        MSEs, MAEs, RMSEe, DRMSEs, MAPEs, COSINEs, MDISPs, MMDISPs, computation_times, series_numbers, series_tps, sample_ids = utils.auto_evaluate(model, generator, batch_num=batch_num, step=1, save_pref='', mask=True)
        
        
        for mse, mae, rmse, drmse, mape, cosine, mdisp, mmdisp, comp_time, sn, stp, sid in zip(MSEs, MAEs, RMSEe, DRMSEs, MAPEs, COSINEs, MDISPs, MMDISPs, computation_times, series_numbers, series_tps, sample_ids):
            result_list.append([model_ID, pred_cases[0], float(hps['split_train']), IDs_dict[model_ID], mse, mae, rmse, drmse, mape, cosine, mdisp, mmdisp, comp_time, sn, stp, sid])

        csv_name = os.path.join(out_path, 'losses_tmp', 'masked_losses_C16-5_T[2-98]_smpl-{}_prt-{}.csv'.format(int(sampling*100), utils.number_zeroPadding_3(m_num+1)))
        with open(csv_name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['ID', 'target_case', 'target_train_size', 'model_name', 'target_mse', 'target_mae', 'target_rmse', 'target_drmse', 'target_mape', 'target_cosine', 'target_mdisp', 'target_mmdisp', 'computation_time', 'series_number', 'series_timePoint', 'sample_id'])
            for row in result_list:
                csvwriter.writerow(row)
                
        print('Total Time elapsed: {}s, Time needed for last Model: {}s'.format(time.time() - start_time, time.time() - model_start_time))
        print()
        print('Now calculating test data loss for {}/{} modelID: {}, proband: {}, with {} test samples'.format(m_num+2, len(model_IDs), model_ID, pred_cases, len(ids)))
        
    print('Complete evaluation of {} models took {}min'.format(len(model_IDs), (time.time() - start_time)/60))