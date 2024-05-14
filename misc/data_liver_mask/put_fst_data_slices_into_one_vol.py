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

train_data_path = '/cache/gino/Data_for_DeepLearning/'
data_dir        = os.path.join(train_data_path, 'test')  
out_path        = '/home/gino/projects/4D_MRI_CNN/experiments/transfer_learning_experiments'
cases = ['subjet1', 'subject2'] #list of folder names containing subject data

for case in cases:
    print('===========', case)
    reg  = '*_N_*.nii.gz'
    ids_temp = []
    ids_temp += sorted(glob.glob(os.path.join(data_dir, case, reg)))
                
    if len(ids_temp) == 0:
        utils.print_attantion(title='No IDs found', text='Could not find IDs for: {}'.format(case))


    ids_dict = utils.groupFileNamesBySeriesNumber_dict(ids_temp)
    keys = sorted(ids_dict.keys())
    ids = []
    for i in ids_dict:         
        ids = ids + ids_dict[i][0:1]  ### grap first sample of each sequence


    data_itks = []
    for i, nav_filename in enumerate(ids):
        data_filename = nav_filename.replace('_N_', '_D_')    
        data_itks.append(sitk.ReadImage(data_filename, sitk.sitkFloat32))
        
    vol_path = glob.glob(os.path.join(data_dir, case, '*_Volume_1.nii.gz'))[0]
    # vol_path = glob.glob(os.path.join(data_dir, case, '*_liver_mask.nrrd'))[0]
    vol_itk  = sitk.ReadImage(vol_path)


    ### pad and crop vol to fit the data slices
    last_size = data_itks[0].GetSize()
    for d in data_itks[1:]:
        if d.GetSize() != last_size:
            print('data inconsistancy in case {}'.format(case))
        last_size = d.GetSize()
        

    d_origins_px = []
    d_ends_px    = []
    d_origins_mm = []
    d_ends_mm    = []
    # print('old pixel origins')
    for d in data_itks:
        do_mm = d.GetOrigin()
        de_mm = d.TransformContinuousIndexToPhysicalPoint(d.GetSize())
        do_px = vol_itk.TransformPhysicalPointToContinuousIndex(do_mm)
        de_px = vol_itk.TransformPhysicalPointToContinuousIndex(de_mm)
        # print('do_px {}'.format(do_px))
        
        d_origins_px.append(do_px)
        d_ends_px.append(de_px)    
        
        d_origins_mm.append(do_mm)
        d_ends_mm.append(de_mm)
    
    min_x_px = 10000
    min_y_px = 10000
    min_z_px = 10000
    max_x_px = -10000
    max_y_px = -10000
    max_z_px = -10000
    
    for o in d_origins_px:
        min_x_px = min(o[0], min_x_px)
        min_y_px = min(o[1], min_y_px)
        min_z_px = min(o[2], min_z_px)
        max_x_px = max(o[0], max_x_px)
        max_y_px = max(o[1], max_y_px)
        max_z_px = max(o[2], max_z_px)
        
    for e in d_ends_px:
        min_x_px = min(e[0], min_x_px)
        min_y_px = min(e[1], min_y_px)
        min_z_px = min(e[2], min_z_px)
        max_x_px = max(e[0], max_x_px)
        max_y_px = max(e[1], max_y_px)
        max_z_px = max(e[2], max_z_px)
        
    print('Slice count = {}'.format(len(data_itks)))
    print('Slice size  = {}'.format(last_size))
    print('Slice spacing  = {}'.format(data_itks[0].GetSpacing()))
    print('volume size = {}'.format(vol_itk.GetSize()))
    print('volume spacing = {}'.format(vol_itk.GetSpacing()))
    print('volume origin  = {}'.format(vol_itk.GetOrigin()))
    
    print('min_x_px', min_x_px)
    print('min_y_px', min_y_px)
    print('min_z_px', min_z_px)
    print('max_x_px', max_x_px)
    print('max_y_px', max_y_px)
    print('max_z_px', max_z_px)
    
    
    
    ### resample vol and set new origin, copp and padd in one step
    new_vo_px = (min_x_px, min_y_px, min_z_px)
    new_ve_px = (max_x_px, max_y_px, max_z_px)
    new_vo_mm = vol_itk.TransformContinuousIndexToPhysicalPoint(new_vo_px)
    new_ve_mm = vol_itk.TransformContinuousIndexToPhysicalPoint(new_ve_px)
    print("new_vo_px {}, new_ve_px {}".format(new_vo_px, new_ve_px))
    
    new_vol_size_px = (new_ve_px[0] - new_vo_px[0], new_ve_px[1] - new_vo_px[1], new_ve_px[2] - new_vo_px[2])
    vz              = new_vol_size_px
    print('new vol size px lr={}(x), ap={}(y), is={}(z)'.format(vz[0], vz[1], vz[2]))
    d_sp         = data_itks[0].GetSpacing()
    new_spacing  = [4.0, d_sp[0], d_sp[1]]
    # vol_itk = prep.resampleImage(vol_itk, new_spacing, sitk.sitkLinear, 0, new_vo_mm, new_vol_size_px)
    vol_itk = prep.resampleImage(vol_itk, new_spacing, sitk.sitkNearestNeighbor, 0, new_vo_mm, new_vol_size_px)
    sitk.WriteImage(vol_itk, os.path.join(data_dir, case, case + '_Volume_1_resampled.nii.gz'))    
    # sitk.WriteImage(vol_itk, os.path.join(data_dir, case + '_liver_mask_resampled.nii'))
    
    vz = vol_itk.GetSize()
    print('new vol size after resampling lr={}(x), ap={}(y), is={}(z)'.format(vz[0], vz[1], vz[2]))


    
    ### resample data slices to resampled vol and 
    ### add voxel intensities of slice into one vol
    d_res_vol = prep.resampleToReference(data_itks[0], vol_itk, sitk.sitkNearestNeighbor, 0)
    # sitk.WriteImage(d_res_vol, os.path.join(data_dir, case + '_mask_ref_vol_{}.nii.gz'.format(0)))
    # print('new pixel origins')
    # last_do_px = vol_itk.TransformPhysicalPointToContinuousIndex(data_itks[0].GetOrigin())
    # print('do_x_px {}, delta {}'.format(last_do_px[0], 0))
    for i, d in enumerate(data_itks[1:]):
        # do_px = vol_itk.TransformPhysicalPointToContinuousIndex(d.GetOrigin())
        # print('do_x_px {}, delta {}'.format(do_px[0], last_do_px[0] - do_px[0]))
        # last_do_px = do_px
        d_res_vol = d_res_vol + prep.resampleToReference(d, vol_itk, sitk.sitkNearestNeighbor, 0)
        # sitk.WriteImage(d_res_vol, os.path.join(data_dir, case + '_mask_ref_vol_{}.nii.gz'.format(i+1)))
        
    ### save vol 
    sitk.WriteImage(d_res_vol, os.path.join(data_dir, case + '_mask_ref_vol.nii'))
    sitk.WriteImage(d_res_vol, os.path.join(data_dir, case + '_mask_ref_vol.nii.gz'))
    sitk.WriteImage(d_res_vol, os.path.join(data_dir, case, case + '_mask_ref_vol.nii'))
    sitk.WriteImage(d_res_vol, os.path.join(data_dir, case, case + '_mask_ref_vol.nii.gz'))


    