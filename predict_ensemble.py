 
import sys
import os
import glob
import numpy as np
import csv
import json

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
from data_generation import grab_samples_keep_seq
from net import UNET3D
from net import CNN
from net import Baseline_Nav_3D


### ===========================
### define GPU to be used
### ===========================
GPU = "1"
if len(sys.argv) == 2:
    GPU = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

train_data_path = '/cache/gino/Data_for_DeepLearning/'
model_main_path = '/data/gino/4D_MRI_CNN/output/models'
ranges_json     = '/home/gino/projects/4D_MRI_CNN/experiments/subject1_prediction_ranges.json'
ensemble_json    = '/home/gino/projects/4D_MRI_CNN/experiments/subject1_ensemble_E15-50_T5.json'
out_path          = '/data/gino/4D_MRI_CNN/output/debug/2D/in_out_samples/ensemble_prediction'            
            
if __name__ == '__main__':
    ### get model IDs of the ensemble
    ens = {}
    with open(ensemble_json, mode='r') as f:
        ens = json.load(f)    
        
    ### get sequence ranges for the target cases
    seq_ranges = {}
    with open(ranges_json, mode='r') as f:
        seq_ranges = json.load(f)
    for target_case in ens:
        print('predicting for target case {}'.format(target_case))
        pred_cases = [target_case]
        
        ### predictions{seq1:{tp1:[p1, p2, ..., pn], 
        #                     tp2:[p1, p2, ..., pn],
        #                     ...
        #                    }, 
        #               seq2:{tp1:[p1, p2, ..., pn], 
        #                     tp2:[p1, p2, ..., pn],
        #                     ...
        #                    }, 
        #               ...
        predictions = {}
        for k in seq_ranges[target_case]['seq_ranges']:
            predictions[int(k)] = {}
            seq_start     = seq_ranges[target_case]['seq_ranges'][k][0]
            seq_end       = seq_ranges[target_case]['seq_ranges'][k][1]
            for i in range(seq_start, seq_end):
                predictions[int(k)][i] = []
                    
        for ID in ens[target_case]['ID-source_Case']: 
            
            print('  ensemble model = {}'.format(ID))
            ### find model with ID
            model_path, json_path, err = utils.find_model_with_ID(model_main_path, ID)
            if err != 0:
                continue
                
            test_train = seq_ranges[target_case]['group']
            data_dir   = os.path.join(train_data_path, test_train)  
                
            ### load hyper parameters
            hps  = utils.load_experiment_json(json_path)
            
            ### check net dimensions
            if hps['netDimensions'] != '2D':
                continue
            
            model = CNN.make_model(hps)    
            model.load_weights(model_path)
            
            ### gether samples
            reg  = '*_N_*.nii.gz'
            ids = []
            for case in pred_cases:
                ids += sorted(glob.glob(os.path.join(data_dir, case, reg)))
                            
                if len(ids) == 0:
                    utils.print_attantion(title='No ref sequence found', text='Could not find ref sequence for: {}'.format(case))
                
            ### grap sequence ranges      
            ids_dict = utils.groupFileNamesBySeriesNumber_dict(ids)
            ids_dict = grab_samples_keep_seq(ids_dict, seq_ranges[target_case]['seq_ranges'])
            
            ### predict           
            for k in ids_dict:      
                print('    seq = {}'.format(k))
                ids       = ids_dict[k]                
                generator = DataGenerator2D_aug(cases=pred_cases, data_dir=data_dir, IDs=ids, size_X=hps['netInput_x'], size_Y=hps['netInput_y'], spacing=hps['resolution'], 
                                                batch_size=len(ids),  shuffle=False, comment='', log_dir='.', augmentation=False, use_explicit_dist_info=hps['use_dist_info'])
                
                (X_val, Y_val) = generator.__getitem__(index=0, inference='single_slice')
                prediction     = model.predict(X_val, batch_size=1)
                s              = prediction.shape[0]
                for i, t in enumerate(predictions[k]):
                    p  = np.fliplr(np.flipud(prediction[i,:,:,0]))
                    predictions[k][t].append(p)
                    # name = target_case + '_' + ID + '_seq_{}_{}'.format(utils.number_zeroPadding(k), utils.number_zeroPadding(i)) + '_pred_' + '.png'
                    # matplotlib.image.imsave( os.path.join(out_path, 'single_predictions', name), p, cmap="gray")
                    
        for k in predictions:
            for i, t in enumerate(predictions[k]):
                ensemble_mean_pred = np.zeros(predictions[k][t][0].shape)
                c = 0
                for e in predictions[k][t]:
                    ensemble_mean_pred += e
                    c += 1
                ensemble_mean_pred = ensemble_mean_pred / c
                name = target_case + '_seq_{}_{}'.format(utils.number_zeroPadding(k), utils.number_zeroPadding(i)) + '_E15-50_T5_pred_' + '.png'
                matplotlib.image.imsave( os.path.join(out_path, name), ensemble_mean_pred, cmap="gray")
