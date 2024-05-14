### ===========================
### package import
### ===========================
import os
import glob
import sys

import numpy as np
import SimpleITK as sitk

from keras import backend as K
from keras.utils import plot_model

import wandb
from wandb.keras import WandbCallback
### ===========================


### ===========================
### import own stuff
### ===========================
from data_generation import DataGenerator
from data_generation import DataGenerator2D_aug
from data_generation import split_samples_seq
from net import UNET3D
from net import CNN
import preprocessing.utils as utils
import preprocessing.preprocess as prep
### ===========================

### path to data
model_main_path = '/data/gino/4D_MRI_CNN/output/models'
data_path       = '/cache/gino/Data_for_DeepLearning/'
out_path        = '/data/gino/4D_MRI_CNN/output/models/'


### ===========================
### define GPU to be used
### ===========================
# GPU = "3"
# if len(sys.argv) == 2:
#     GPU = sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU

cases_train=['2018-05-07_JBaf',
            ]
            
cases_val = []
 

### hps for 2D replicate old findings
hps_default = dict( sfs                = 32, 
                    batchSize          = 64,
                    cnn_depht          = 3,                    
                    bn                 = False, 
                    do                 = 0.15,
                    cases_train        = cases_train, 
                    cases_val          = cases_val,                    
                    regEx              = '*_N_*.nii.gz', 
                    
                    augmentation       = True,
                    aug_rot            = 3,   ### augmentation rotation range in degree 
                    aug_w_shift        = 10,  ### augmentation width shift range in voxel                
                    aug_h_shift        = 10,  ### augmentation height shift range in voxel
                    aug_zoom_min       = 0.8, ### augmentation min zoom     
                    aug_zoom_max       = 1.2,  
                           
                    shuffle            = True,
                    use_dist_info      = False,
                    
                    split_train        = 0.1, 
                    split_val          = 0.02,           ### percentage of samples to be used for training and validation respectivly. 
            
                    fineTuneFrom       = '', #'21mcw7gu', #'pia9npio', #'/data/gino/4D_MRI_CNN/output/models/2D/2021_02_10_09_50_28_None_z63il8jn/model_2D_d3_s6.h5', #'/data/gino/4D_MRI_CNN/output/models/2D/2021_02_06_18_06_42_None_3skzjv2x/model_2D_d3_s6.h5', #'/data/gino/4D_MRI_CNN/output/models/3D/2021_01_25_17_48_09_None_2vo66uah/model_3D_d2_s3.h5', #'/cache/gino/Data_for_DeepLearning/output/models/2D/useCase2/PreTrain/model_uC2_preTrain_P2_d2_s5.h5',      ### start from this pre trained model
                    modelName          = "T98",
                    netDimensions      = '2D',
                    netInput_x         = 128,
                    netInput_y         = 128,
                    resolution         = [1.818, 1.818, 1.818],
                    LR                 = 0.000413,
                    
                    L2_WEIGHT          = 0.0,                    
                    ACTIVATION         = 'lrelu',
                    LAST_L_ACTIVATION  = 'lrelu', 
                    LOS_FUNC           = 'mse',
                    NORMILIZATION      = 'standard', #'double_mean', 'max'
                    
                    LEARN_DIFF         = False,
                    
                    ERLSTOP_MIN_DELTA  = 0.0001, 
                    ERLSTOP_PATIENCE   = 100,
                    LR_DECAY_FACTOR    = 0.8, 
                    LR_DECAY_PATIENCE  = 50, 
                    LR_DECAY_MIN_LR    = 1e-8, 
                    LR_DECAY_MIN_DELTA = 0.001,
                        )

### set relative working directory for different user
os.chdir(os.path.expanduser('~/projects/4D_MRI_CNN/'))

if __name__ == '__main__':
    
    ### init wandb
    wandb.init(project="4D_MRI_CNN", config=hps_default, dir='/data/gino/4D_MRI_CNN/output')
    hps = wandb.config
    
    ### make dir to save model, doc.json
    timeStamp = utils.get_timestamp()
    out_dir   = timeStamp + '_' + str(wandb.run.name) + '_' + str(wandb.run.id)
    out_path  = os.path.join(out_path, hps.netDimensions, out_dir)
    utils.makeDirectory(out_path)

    ### save info about this run
    MODEL_FILENAME  = os.path.join(out_path, hps.modelName +  '_best.h5')  
    utils.save_experiment_json(hps._items, save_path=out_path, save_name=MODEL_FILENAME[0:-3] + '.json')
    
    
           
    print("============================================")
    print("got {} arguments".format(len(sys.argv)))
    for i, arg in enumerate(sys.argv):
        print("{}: {}".format(i, arg))        
    print("============================================")
    
    ### gather samples 
    print("============================================")
    print('here comes my cases_train arg')
    cases_train = hps.cases_train
    if type(cases_train) == str:
        print('got case as string: {}'.format(cases_train))
        if '[' in cases_train: ### cases_train = "['case1', 'case2', ...]"
            
            print('cases seem to be in a list.'.format(cases_train))
            
            var_type_befor = type(cases_train)
            cases_train = eval(cases_train) ### should be evaluated as a list of str
            print('evaluated the string as a list for you: from {} to {}'.format(var_type_befor, type(cases_train)))

        else:
            cases_train = [cases_train]
            print('it was only a single case. I put it in a list for you: {}'.format(cases_train))
    else:
        print('got case as list: {}'.format(cases_train))      
         
    print('checking the soundness of each element in cases_train:')
    for c in cases_train:
        print(c, type(c)) 
    
    cases_val = hps.cases_val
    if type(cases_val)   == str:
        cases_val = [cases_val]
        
    train_samples = []
    val_samples   = []
    val_data_path = os.path.join(data_path, 'val')
    # train_data_path = os.path.join(data_path, 'train')
    train_data_path = os.path.join(data_path, 'test')
    
    if len(cases_val) > 0:
        for case in cases_train:
            samples               = sorted(glob.glob(os.path.join(train_data_path, case, hps.regEx)))
            samples               = utils.groupFileNamesBySeriesNumber(samples) 
            train_samples_temp, _ = split_samples_seq(samples, [hps.split_train, hps.split_val])
            train_samples         = train_samples + samples

        for case in cases_val:
            samples             = sorted(glob.glob(os.path.join(val_data_path, case, hps.regEx)))
            samples             = utils.groupFileNamesBySeriesNumber(samples) 
            _, val_samples_temp = split_samples_seq(samples, [hps.split_train, hps.split_val])
            val_samples         = val_samples + samples

    else:   ### split samples in train and validation
        val_data_path = train_data_path
        for case in cases_train:  
            samples                = sorted(glob.glob(os.path.join(train_data_path, case, hps.regEx)))
            samples                = utils.groupFileNamesBySeriesNumber(samples) 
            train_split, val_split = split_samples_seq(samples, [hps.split_train, hps.split_val])
            train_samples          = train_samples + train_split
            val_samples            = val_samples   + val_split

    ### sanity check the split
    prep.sanity_check_training_split(train_samples, val_samples)
    
    K.clear_session()
    if hps.netDimensions == '3D':
        ### make model
        model = UNET3D.make_model(hps._items)
        
        ### make data generator for validation data
        val_gen  = DataGenerator(cases='', data_dir=val_data_path, IDs=val_samples, batch_size=hps.batchSize,    shuffle = False, log_dir='.', 
                                 augmentation=hps.augmentation, aug_rot=hps.aug_rot, aug_w_shift=hps.aug_w_shift, aug_h_shift=hps.aug_h_shift, aug_zoom_min=hps.aug_zoom_min, aug_zoom_max=hps.aug_zoom_max)

        ### make data generator for training data
        data_gen = DataGenerator(cases='', data_dir=train_data_path, IDs=train_samples, batch_size= hps.batchSize, shuffle=hps.shuffle, log_dir='.', 
                                 augmentation=hps.augmentation, aug_rot=hps.aug_rot, aug_w_shift=hps.aug_w_shift, aug_h_shift=hps.aug_h_shift, aug_zoom_min=hps.aug_zoom_min, aug_zoom_max=hps.aug_zoom_max)
    
    elif hps.netDimensions == '2D':
        s_x = hps.netInput_x
        s_y = hps.netInput_y
        spacing = hps.resolution 
        # s_x = 64
        # s_y = 64
        # spacing = [4.0, 4.0, 4.0]
        
        
        ### make model
        model = CNN.make_model(hps._items)
        
        ### make data generator for validation data 
        val_gen  = DataGenerator2D_aug(cases='', data_dir=val_data_path, IDs=val_samples,   size_X=s_x, size_Y=s_y, spacing=spacing, batch_size=hps.batchSize,  shuffle=False,       comment='', log_dir='.', 
                                       augmentation=False, aug_rot=hps.aug_rot, aug_w_shift=hps.aug_w_shift, aug_h_shift=hps.aug_h_shift, aug_zoom_min=hps.aug_zoom_min, aug_zoom_max=hps.aug_zoom_max, use_explicit_dist_info=hps.use_dist_info)

        ### make data generator for training data
        data_gen = DataGenerator2D_aug(cases='', data_dir=train_data_path, IDs=train_samples, size_X=s_x, size_Y=s_y, spacing=spacing, batch_size= hps.batchSize, shuffle=hps.shuffle, comment='', log_dir='.', 
                                       augmentation=hps.augmentation, aug_rot=hps.aug_rot, aug_w_shift=hps.aug_w_shift, aug_h_shift=hps.aug_h_shift, aug_zoom_min=hps.aug_zoom_min, aug_zoom_max=hps.aug_zoom_max, use_explicit_dist_info=hps.use_dist_info)

    ### save model graph
    model.summary()  ## print model architecture
    plot_model(model, to_file=os.path.join(out_path, 'model.png'), show_shapes=True)

    ### if finetuning:
    if hps.fineTuneFrom !='':     
        ### ToDo check consistency between hps 
        model_path, json_path, err = utils.find_model_with_ID(model_main_path, hps.fineTuneFrom)   
        if err != 0:
            exit()     
        model.load_weights(model_path)

    ### make callbacks   
    CSV_FILENAME    = os.path.join(out_path, "log.csv")
    callbacks       = utils.make_callbacks(hps, csv_filename=CSV_FILENAME, model_filename=MODEL_FILENAME)
    callbacks.append(WandbCallback())
    
    ### train and save model
    history = model.fit_generator(generator=data_gen, validation_data=val_gen, callbacks=callbacks, use_multiprocessing=False, epochs=200, workers=8)
    model.save(os.path.join(out_path, hps.modelName + '_final.h5'))
  
    
    ### auto predict after training run
    out_path = os.path.join('/data/gino/4D_MRI_CNN/output/auto_predict', out_dir)
    utils.makeDirectory(out_path)

    # if hps.netDimensions == '3D':
    #     utils.auto_predict_3D(model, val_gen, batch_num=3, max_vols=20, max_slices=30, out_path=out_path, save_name_pref=str(wandb.run.id))
    # elif hps.netDimensions == '2D':
    #     utils.auto_predict_3D_from_2D(model, val_gen, out_path=out_path, batch_num=30, save_name_pref=str(wandb.run.id))
wandb.finish