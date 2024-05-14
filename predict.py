 
import sys
import os
import glob
import numpy as np

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
from net import UNET3D
from net import CNN
from net import Baseline_Nav_3D


### ===========================
### define GPU to be used
### ===========================
# GPU = "3"
# if len(sys.argv) == 2:
#     GPU = sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU

train_data_path = '/cache/gino/Data_for_DeepLearning/'
model_main_path = '/data/gino/4D_MRI_CNN/output/models'
model_path      = ''

data_dir        = os.path.join(train_data_path, 'test')  
out_path        = '/home/gino/projects/data/2D_prediction/Test_subjects/T98/pred_with_GT'



IDs = ["nw266mak", ### T98 models
        "cox6ympi",
        "cwwozqyf",
        "4p3vhv3q",]







if __name__ == '__main__':
    for ID in IDs: 
        ### find model with ID
        model_path, json_path, err = utils.find_model_with_ID(model_main_path, ID)
        if err != 0:
            continue
        hps        = utils.load_experiment_json(json_path)
        pred_cases =  hps['cases_train']
        if type(pred_cases) == str:
            pred_cases = [pred_cases]
                
        if hps['netDimensions'] == '3D':
            model       = UNET3D.make_model(hps) 
            model.load_weights(model_path)  
            generator   = DataGenerator(cases=pred_cases, data_dir=data_dir, IDs=[], batch_size=10, shuffle = False, augmentation = False)
            utils.auto_predict_3D(model, generator, batch_num=3, max_vols=20, max_slices=100, out_path=os.path.join(out_path, '3D'))    
                
        elif hps['netDimensions'] == '2D':
            model       = CNN.make_model(hps)    
            model.load_weights(model_path)
            reg  = '*_N_*.nii.gz'
            # reg  = '*_NR_*.nii.gz'
            ids_temp = []
            for case in pred_cases:
              ids_temp += sorted(glob.glob(os.path.join(data_dir, case, reg)))#[305:320]
                            
            if len(ids_temp) == 0:
             utils.print_attantion(title='No ref sequece found', text='Could not find ref sequence for: {}'.format(case))
             
            if True: ### grap second half of each sequence
                ids_dict = utils.groupFileNamesBySeriesNumber_dict(ids_temp)
                keys = sorted(ids_dict.keys())
                ids = []
                for i in ids_dict:            
                    seq_half = int(len(ids_dict[i])/2)+1
                    # ids = ids + ids_dict[i][seq_half:]  
                    ids = ids + ids_dict[i][-2:]  
            else:
                ids = ids_temp
    
            generator = DataGenerator2D_aug(cases=pred_cases, data_dir=data_dir, IDs=ids,   size_X=hps['netInput_x'],  size_Y=hps['netInput_y'], spacing=hps['resolution'], batch_size=1,  shuffle=False, comment='', log_dir='.', 
                                    augmentation=False, use_explicit_dist_info=hps['use_dist_info'])
            utils.makeDirectory(os.path.join(out_path, case))
            # utils.auto_predict_3D_from_2D(model, generator, batch_num=len(ids), step=1, max_vols=513, max_slices=513, out_path=os.path.join(out_path, case), save_name_pref=ID + '_Splt_' + str(utils.number_zeroPadding_3(hps['split_train']*100)))
            utils.auto_predict_2D(model, generator, batch_num=len(ids), step=1, out_path=os.path.join(out_path, case))

            

        