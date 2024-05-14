 
import sys
import os
import glob
import numpy as np
import csv

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
GPU = "3"
if len(sys.argv) == 2:
    GPU = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

train_data_path     = '/cache/gino/Data_for_DeepLearning/'
model_main_path     = '/data/gino/4D_MRI_CNN/output/models'
predict_models_csv  = '/home/gino/projects/4D_MRI_CNN/experiments/predict_models.csv'
ranges_json         = '/home/gino/projects/4D_MRI_CNN/experiments/mdna_seq_ranges_for_prediction.json'
# experiment_csv    = '/home/gino/projects/4D_MRI_CNN/evaluation/TRE/experiment.csv'
# IDs               = ['ugfqw5op'] #, 'qtkyuccx', 'zgra275j', 'blp5jgob', 'mnpnlmmm', 't0yd4k2p', '8wdmb9jx', '42v8nij2']
out_path            = '/data/gino/4D_MRI_CNN/output/debug/2D/in_out_samples/transfer_learning_experiment'
# out_path          = '/data/gino/4D_MRI_CNN/output/images_for_TRE/'
# out_path          = '/data/gino/4D_MRI_CNN/output/debug/2D/in_out_samples/for_nets_own_subject'
prediction_model_ID = '1pq4phq1'
predict_case        = '2018-06-07_MDna'

if __name__ == '__main__':
    
    
    ### get prediction sequence ranges for the target cases
    seq_ranges_for_pred = {}
    with open(ranges_json, mode='r') as f:
        seq_ranges_for_pred = json.load(f)
    
    ### get model IDs, and sequenc ranges from csv
    IDs_ranges_dict = {}
    last_model_ID = ''
    with open(predict_models_csv, 'r', newline='') as csvfile:    
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            index = row['index']
            m_ID  = row['model_ID']
            
            IDs_ranges_dict[m_ID]               = {}
            IDs_ranges_dict[m_ID]['pred_case']  = row['pred_case']
            IDs_ranges_dict[m_ID]['group']      = row['group']
            IDs_ranges_dict[m_ID]['seq_ranges'] = {}
            IDs_ranges_dict[m_ID]['dl-tl']      = row['dl-tl']
            IDs_ranges_dict[m_ID]['train_size'] = row['train_size']
            
            for seq in seq_ranges[predict_case]['seq_ranges']:             
                start_index = seq_ranges[predict_case]['seq_ranges'][seq][0]
                end_index   = seq_ranges[predict_case]['seq_ranges'][seq][1]
                IDs_ranges_dict[m_ID]['seq_ranges'][seq] = [start_index, end_index]     

    
    for ID in IDs_ranges_dict: 
        ### find model with ID
        model_path, json_path, err = utils.find_model_with_ID(model_main_path, ID)
        if err != 0:
            continue
            
        hps = utils.load_experiment_json(json_path)
        # pred_cases =  hps['cases_train']
        pred_cases = IDs_ranges_dict[ID]['case']
        if type(pred_cases) == str:
            pred_cases = [pred_cases]
            
            
        test_train      = IDs_ranges_dict[ID]['group']
        data_dir        = os.path.join(train_data_path, test_train)  
            
        if hps['netDimensions'] == '3D':
            model       = UNET3D.make_model(hps) 
            model.load_weights(model_path)  
            generator   = DataGenerator(cases=pred_cases, data_dir=data_dir, IDs=[], batch_size=10, shuffle = False, augmentation = False)
            utils.auto_predict_3D(model, generator, batch_num=3, max_vols=20, max_slices=100, out_path=os.path.join(out_path, '3D'))
        
            
        elif hps['netDimensions'] == '2D':
            model = CNN.make_model(hps)    
            model.load_weights(model_path)
            
            reg  = '*_N_*.nii.gz'
            ids = []
            for case in pred_cases:
                ids += sorted(glob.glob(os.path.join(data_dir, case, reg)))
                            
                if len(ids) == 0:
                    utils.print_attantion(title='No ref sequece found', text='Could not find ref sequence for: {}'.format(case))
                
            ids_dict = utils.groupFileNamesBySeriesNumber_dict(ids)
            # ids = grab_samples_seq(ids, IDs_ranges_dict[ID]['seq_ranges'])
            ranges = IDs_ranges_dict[ID]['seq_ranges']
            # ranges = IDs_ranges_dict['xads9fdo']['seq_ranges']
            for k in ranges:
                start = ranges[k][0]
                end   = ranges[k][1]                
                ids   = ids_dict[k][start:end]
            
                generator   = DataGenerator2D_aug(cases=pred_cases, data_dir=data_dir, IDs=ids,   size_X=hps['netInput_x'],  size_Y=hps['netInput_y'], spacing=hps['resolution'], batch_size=len(ids),  shuffle=False, comment='', log_dir='.', 
                                            augmentation=False, use_explicit_dist_info=hps['use_dist_info'])
                # utils.auto_predict_3D_from_2D(model, generator, batch_num=2, step=130, max_vols=10, max_slices=0, out_path=os.path.join(out_path, '3D_from_2D'), save_name_pref=ID)
                save_prefix = utils.concat_name([ID, IDs_ranges_dict[ID]['dl-tl'], IDs_ranges_dict[ID]['train_size']])
                utils.auto_predict_2D(model, generator, batch_num=1, step=1, out_path=out_path, save_name_pref=save_prefix, save_name_suf='_seq_{}'.format(utils.number_zeroPadding(k)))

        