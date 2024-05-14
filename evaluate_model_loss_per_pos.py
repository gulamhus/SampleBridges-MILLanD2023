 
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

train_data_path = '/cache/gino/Data_for_DeepLearning/'
model_main_path = '/data/gino/4D_MRI_CNN/output/models'
runs_csv        = '/home/gino/projects/4D_MRI_CNN/evaluation/split_test/split_test_IDs.csv'
out_path        = '/data/gino/4D_MRI_CNN/output/debug/'

# IDs =['47f1wjrk',
#     'qtkyuccx',
#     'zgra275j',
#     '1in6zdp7',
#     'blp5jgob',
#     'zq0hubdi',
#     '79l7nxlx',
#     'mnpnlmmm',
#     't0yd4k2p',
#     '6pqx9rur',
#     'hnsfymsz',
#     'ismlgo5v',
#     '8wdmb9jx',
#     '42v8nij2',
#     'rfqx7gli',
#     'pivxe0qh',]

#IDs = ['ugfqw5op', '33u6gxja', 'y3k8yg30', 'xads9fdo'] # test




if __name__ == '__main__':
    
    ### get model IDs from csv
    IDs_dict = {}
    with open(runs_csv, 'r', newline='') as csvfile:    
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            IDs_dict[row['ID']] = 1         
    IDs = list(IDs_dict.keys())
    
    for ID in IDs: 
        model_path = ''
        ### find model with ID
        found = []
        for dir, subdirs, files in os.walk(model_main_path):
            if dir.find(ID) >= 0:
                found.append(dir)
                # model_path = os.path.join(dir, 'model_final.h5')
                model_path = os.path.join(dir, 'model_best.h5')
                # model_path = os.path.join(dir, 'model_2D_d3_s5.h5')
                
                json_path = glob.glob(os.path.join(dir, '*.json'))[0]
    
        if len(found) == 0:
            utils.print_attantion(title='ID not found', text='could not find specified ID: {}'.format(ID))
            continue
        if len(found) > 1:
            text = 'Specified ID is not unambiguous : {} \n'.format(ID) + 'Found ID in following locations:{}'.format(found)
            utils.print_attantion(title='To many IDs found', text=text)
            continue

        ### load hyper parameters
        if model_path == '':
            continue
            
        path = os.path.split(model_path)[0]
        hps = utils.load_experiment_json(path, os.path.split(json_path)[1])
        pred_cases =  hps['cases_train']
        if type(pred_cases) == str:
            pred_cases = [pred_cases]
            
        data_dir = os.path.join(train_data_path, 'train')  
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
                    utils.print_attantion(title='No IDs found', text='Could not find IDs for: {}'.format(case))
                
            ids_dict = utils.groupFileNamesBySeriesNumber_dict(ids)
            keys = sorted(ids_dict.keys())
            result_table = {}
            for i in ids_dict:            
                ids = ids_dict[i]
                ids = ids[95:]# grap second half of thit sequence
                
                generator   = DataGenerator2D_aug(cases=pred_cases, data_dir=data_dir, IDs=ids,   size_X=hps['netInput_x'],  size_Y=hps['netInput_y'], spacing=hps['resolution'], batch_size=len(ids),  shuffle=False, comment='', log_dir='.', 
                                            augmentation=False, use_explicit_dist_info=hps['use_dist_info'])
                print('calculateing val mse for slice pos {}/{}'.format(i,keys[-1]))
                result_table[i] = utils.auto_evaluate(model, generator, batch_num=1, step=1, max_vols=0, max_slices=0, out_path=os.path.join(out_path, '3D_from_2D'), save_name_pref=ID)
                print('val_mse {} = {}'.format(i,result_table[i] ))
            
            print(result_table)
            csv_name = pred_cases[0]
            with open(csv_name + '.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['case', 'seq', 'data_pos_index', 'nav_pos', 'data_pos', 'data_rel_pos', 'split_train', 'val_mse'])
                pos_index = 0
                for s in result_table:
                    ### nav pos
                    nav_name = ids_dict[s][0]
                    nav_img = sitk.ReadImage(nav_name)
                    nav_pos = nav_img.GetOrigin()[0]
                    
                    ### data pos
                    data_name = nav_name.replace('_N_', '_D_')
                    data_img = sitk.ReadImage(data_name)
                    data_pos = data_img.GetOrigin()[0]
                    
                    ### rel pos
                    data_rel_pos = data_pos - nav_pos
                    
                    csvwriter.writerow([pred_cases[0], s, pos_index, nav_pos, data_pos, data_rel_pos, float(hps['split_train']), result_table[s]])
                    pos_index = pos_index + 1