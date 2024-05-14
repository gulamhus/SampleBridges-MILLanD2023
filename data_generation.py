import keras
import glob
import numpy as np
#import dataAugmentation
import SimpleITK as sitk
import os
import preprocessing.utils as utils
import time
import math
import preprocessing.preprocess as prep
import matplotlib
import matplotlib.pyplot as plt
import threading
import random
import json

import tensorflow as tf

SIZE_X = 64 #112
SIZE_Y = 64 #112
SIZE_Z = 80

class DataGenerator(keras.utils.Sequence):
    def __init__(self, cases='', IDs=[], data_dir='', batch_size=2,
                 shuffle=True, 
                 augmentation = True, 
                 aug_rot      = 10,  ### augmentation rotation range in degree 
                 aug_w_shift  = 10,  ### augmentation width shift range in voxel                
                 aug_h_shift  = 10,  ### augmentation height shift range in voxel
                 aug_zoom_min = 0.8, ### augmentation min zoom     
                 aug_zoom_max = 1.3, ### augmentation max zoom   
                 sampe_weights = [], 
                 log_dir = '.'):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.aug_args = dict(rotation_range=aug_rot, 
                             width_shift_range=aug_w_shift, 
                             height_shift_range=aug_h_shift, 
                             zoom_range=[aug_zoom_min, aug_zoom_max], 
                             data_format="channels_last", 
                             dtype=np.float32)
        
        self.data_dir = data_dir

        self.IDs = []
        self.vols = {}
        self.slice_norm_params = {}
        self.vol_norm_params   = {}

        if len(IDs) > 0:
            self.init_data_IDs(IDs, data_dir)
        elif cases != '' and data_dir != '':
            self.init_data_case(cases, data_dir)
        else:
            print('no data paths are given')


        self.on_epoch_end()
        self.sample_weights= sampe_weights
        self.examples = 1
        self.logger = utils.function_logger(os.path.join(log_dir, 'data_generation'))

    def init_data_case(self,cases, data_dir):
        reg  = '*_N_*.nii.gz'
        for case in cases:
            proband_samples = sorted(glob.glob(os.path.join(data_dir, case, reg)))
            self.init_data_IDs(proband_samples, data_dir)

    def init_data_IDs(self, IDs, data_dir):
        self.IDs = self.IDs + IDs
        for i in IDs:
            ### extract case name
            split_filename = i.split('/')
            case = split_filename[-2]

            if not case in self.slice_norm_params:
                with open(os.path.join(data_dir, case, 'normalization_factors.json'), 'r') as fp:
                    data = json.load(fp)
                    self.slice_norm_params[case] = {'mean': data['slice_mean'], 'std': data['slice_adjusted_std']}
                    self.vol_norm_params[case]   = {'mean': data['vol_mean'],   'std': data['vol_adjusted_std']}
            
            if not case in self.vols:
                vol_filename = os.path.join(data_dir, case, case +'_Volume_1.nii.gz')
                vol = prep.load_and_preprocess_vol(vol_filename, newSpacing=[4.0, 4.0, 4.0], newSize=[160, 160, 80])

                ### crop volume to 64 x 64 x 80
                with open(os.path.join(data_dir, 'vol_origins_phy.json'), mode='r') as f:
                    vol_origins = json.load(f)
                    index = vol.TransformPhysicalPointToIndex(vol_origins[case])
                                             
                     ### crop to 64 x 64 x 64
                    from_bottom = [index[0]-64, index[1]-64, 16]     
                    from_Top    = [160 - index[0]-16, 160 - index[1], 0]
                    vol = sitk.Crop(vol, from_bottom, from_Top)
                
                self.vols[case] = vol

        self.indices = np.arange(len(self.IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, IDs_batch, return_RefVol=False, ref_vols=[]):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty([self.batch_size, SIZE_Y,SIZE_X, SIZE_Z, 2], dtype=np.float32)
        Y = np.empty([self.batch_size, SIZE_Y,SIZE_X, SIZE_Z, 2], dtype=np.float32)

        # Generate data
        for i, ID in enumerate(IDs_batch):
            # load sample
            nav_filename = ID
            data_filename = nav_filename.replace('_N_', '_D_')
            split_filename = data_filename.split('/')
            case = split_filename[-2]
            
            slice_n = prep.load_and_preprocess_slice(nav_filename, newSpacing=[4.0, 4.0, 4.0])
            slice_d = prep.load_and_preprocess_slice(data_filename, newSpacing=[4.0, 4.0, 4.0])
  

            # slice_n = sitk.Multiply(slice_n, 1/self.slice_norm_factors[case])
            # slice_d = sitk.Multiply(slice_d, 1/self.slice_norm_factors[case])

            # resample slices to volume
            nav_pad  = prep.resampleToReference(slice_n, self.vols[case], sitk.sitkNearestNeighbor, 0)
            data_pad = prep.resampleToReference(slice_d, self.vols[case], sitk.sitkNearestNeighbor, -1)  # -1 is for masking in loss function


            ### image to array
            vol_cropped_arr = sitk.GetArrayFromImage(self.vols[case])
            nav_pad_arr     = sitk.GetArrayFromImage(nav_pad)
            data_pad_arr    = sitk.GetArrayFromImage(data_pad)       
            mask_pad_arr    = np.where(data_pad_arr >= 0, 1.0, 0.0) 


            # nav             = np.amax(sitk.GetArrayFromImage(nav_pad), axis=2).reshape(SIZE_Y, SIZE_X, 1)
            # nav_pad_arr     = np.zeros((SIZE_Y,SIZE_X, SIZE_Z), dtype=np.float32)
            # nav_pad_arr[:,:,:] = nav

            # nav = sitk.GetImageFromArray(nav_pad_arr)
            # nav.CopyInformation(nav_pad)
            # sitk.WriteImage(nav, os.path.join('/cache/gino/Data_for_DeepLearning/output/debug', 'nav_pad.nrrd'))
            
            ### normalize
            slice_means = self.slice_norm_params[case]['mean']
            slice_stds  = self.slice_norm_params[case]['std']
            vol_means   = self.vol_norm_params[case]['mean']
            vol_stds    = self.vol_norm_params[case]['std']

            vol_cropped_arr = (vol_cropped_arr  - vol_means)    / vol_stds 
            nav_pad_arr     = (nav_pad_arr      - slice_means)  / slice_stds
            data_pad_arr    = (data_pad_arr     - slice_means)  / slice_stds
    

            # store augmented sample 
            X[i, :, :, :, 0] = vol_cropped_arr
            X[i, :, :, :, 1] = nav_pad_arr
            Y[i, :, :, :, 0] = data_pad_arr
            Y[i, :, :, :, 1] = mask_pad_arr
            if return_RefVol:
                ref_vols.append(self.vols[case])
            
            
        

        # TODO augment sample
        if self.augmentation:
            #X, Y = self.augment(X, Y)
            (X,Y) = self.augment_(X, Y)
        return X, Y

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print( 'List_ID', type(self.list_IDs))
        #print('batch', type(self.batch_size))
        return int(np.floor(len(self.IDs) / self.batch_size))

    def __getitem__(self, index, return_RefVol=False, ref_vols=[]):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        IDs_batch = [self.IDs[k] for k in indexes]

        # select weights if they exist, otherwise give list of ones for equal sampling
        if not self.sample_weights == []:
            s_weights_temp = [self.sample_weights[j] for j in indexes]
        else:
            s_weights_temp_arr = np.ones(self.batch_size)
            s_weights_temp = s_weights_temp_arr.tolist()

        # Generate data
        X, Y  = self.__data_generation(IDs_batch, return_RefVol=return_RefVol, ref_vols=ref_vols)
        return X, Y,  np.array(s_weights_temp)

    def augment(self, X, Y):
        data = np.concatenate((X[:, :, :, :, 0] , X[:, :, :, :, 1], Y[:, :, :, :, 0]), axis=3)
        aug = tf.keras.preprocessing.image.ImageDataGenerator(**self.aug_args)
        
        for data_batch, y_batch in aug.flow(data, Y, batch_size=self.batch_size, shuffle=False):
            X_train = np.empty([self.batch_size, SIZE_Y, SIZE_X, SIZE_Z, 2], dtype=np.float32)
            Y_train = np.empty([self.batch_size, SIZE_Y, SIZE_X, SIZE_Z, 2], dtype=np.float32)
            X_train[:, :, :, :, 0] = data_batch[:,:,:,:80]
            X_train[:, :, :, :, 1] = data_batch[:,:,:,80:160]
            Y_train[:, :, :, :, 0] = data_batch[:,:,:,160:]
            Y_train[:, :, :, :, 1] = Y[:,:,:,:,1]
            return (X_train, Y_train)

    def augment_(self, X, Y):
        aug_args = dict(rotation_range=10, 
                            width_shift_range=10, 
                            height_shift_range=10, 
                            zoom_range=[0.8, 1.3], 
                            data_format="channels_last", 
                            dtype=np.float32)

        data = np.concatenate((X[:, :, :, :, 0] , X[:, :, :, :, 1], Y[:, :, :, :, 0]), axis=3)
        aug = tf.keras.preprocessing.image.ImageDataGenerator(**aug_args)
                
        data_batch = [aug.random_transform(x) for x in data]
        # data_batch = aug.random_transform(data)
        del aug
        X_train = np.empty([self.batch_size, SIZE_Y, SIZE_X, SIZE_Z, 2], dtype=np.float32)
        Y_train = np.empty([self.batch_size, SIZE_Y, SIZE_X, SIZE_Z, 2], dtype=np.float32)
        for i, d in enumerate(data_batch):
            X_train[i, :, :, :, 0] = d[:, :, :80]
            X_train[i, :, :, :, 1] = d[:, :, 80:160]
            Y_train[i, :, :, :, 0] = d[:, :, 160:]
            Y_train[i, :, :, :, 1] = Y[i, :, :, :, 1]
        return (X_train, Y_train)

    def augment__(self, X, Y):
        aug_args = dict(rotation_range=10, 
                            width_shift_range=10, 
                            height_shift_range=10, 
                            zoom_range=[0.8, 1.3], 
                            data_format="channels_last", 
                            dtype=np.float32)

        data = np.concatenate((X[:, :, :, :, 0] , X[:, :, :, :, 1], Y[:, :, :, :, 0]), axis=3)
        aug = tf.keras.preprocessing.image.ImageDataGenerator(**aug_args)
                
        data_batch = tf.map_fn(aug.random_transform, tf.convert_to_tensor(data, dtype=tf.float32))
        # data_batch = aug.random_transform(data)
        del aug
        X_train = np.empty([self.batch_size, SIZE_Y, SIZE_X, SIZE_Z, 2], dtype=np.float32)
        Y_train = np.empty([self.batch_size, SIZE_Y, SIZE_X, SIZE_Z, 2], dtype=np.float32)
        X_train[:, :, :, :, 0] = data_batch[:,:,:,:80]
        X_train[:, :, :, :, 1] = data_batch[:,:,:,80:160]
        Y_train[:, :, :, :, 0] = data_batch[:,:,:,160:]
        Y_train[:, :, :, :, 1] = Y[:,:,:,:,1]
        return (X_train, Y_train)

class DataGenerator2D(keras.utils.Sequence):
    def __init__(self, list_IDs, data_dir, slice_pos, slice_stdandardization_params, vol_stdandardization_params, slice_norm_factors, vol_norm_factors, vol_origins, batch_size=2,
                 shuffle=True, augmentation = False, augmentation_parameters = {}, comment='_', log_dir='.'):
        'Initialization'
        self.sample_count = 0
        self.batches_generated = 0
        self.epochs_lived = 0
        self.actual_bs = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_channels = 1
        self.list_IDs = list_IDs       
        self.augmentation = augmentation
        self.augmentation_parameters = augmentation_parameters
        self.data_dir = data_dir
        self.on_epoch_end()
        self.slice_norm_factors = slice_norm_factors
        self.vol_norm_factors = vol_norm_factors
        self.slice_stdandardization_params = slice_stdandardization_params
        self.vol_stdandardization_params = vol_stdandardization_params
        self.vol_origins = vol_origins
        self.slice_pos = slice_pos
        self.vols_cropped, self.vol_slices, self.vol_slices_arr = self.load_proband_volume_slices()
        self.vols  = self.load_proband_volumes()
        self.comment = comment
        self.logger = utils.function_logger(os.path.join(log_dir, 'data_generation'))
        self.mean_slices = self.loadMeanSlices()
        #prep.sanity_check_SampleNumber(list_IDs, comment, slice_pos, self.logger)
    
    def load_proband_volume_slices(self):
        vols = {}
        vol_slices = {}
        vol_slices_arr = {}
        for case in self.vol_norm_factors:
            vol_filename = os.path.join(self.data_dir,case, case +'_Volume_1.nii.gz')
            vol = prep.load_and_preprocess_vol(vol_filename, newSpacing=[4.0, 4.0, 4.0], newSize=[160, 160, 80])
            
            # intensity normalization
            vol = sitk.Multiply(vol, 1/self.vol_norm_factors[case])
            #sitk.WriteImage(vol, 'vol_normalized.nrrd')

            from_bottom = [self.vol_origins[case][0]-64, self.vol_origins[case][1]-64, 16]     
            from_Top    = [160 - self.vol_origins[case][0], 160 - self.vol_origins[case][1], 0]
            vol_cropped = sitk.Crop(vol, from_bottom, from_Top)
            #sitk.WriteImage(vol_cropped, 'vol_cropped.nrrd')
            vols[case] = vol_cropped

            from_bottom = [int(self.slice_pos), 0, 0]     
            from_Top    = [63 - int(self.slice_pos), 0, 0]
            vol_slice = sitk.Crop(vol_cropped, from_bottom, from_Top)
            #sitk.WriteImage(vol_slice, 'vol_slice.nrrd')
            vol_slices[case] = vol_slice
            vol_slices_arr[case] = sitk.GetArrayFromImage(vol_slice)
        return vols, vol_slices, vol_slices_arr


    def load_proband_volumes(self):
        vols = {}
        for case in self.vol_norm_factors:
            vol_filename = os.path.join(self.data_dir,case, case +'_Volume_1.nii.gz')
            vol = prep.load_and_preprocess_vol(vol_filename, newSpacing=[4.0, 4.0, 4.0], newSize=[160, 160, 80])
            
            # intensity normalization
            vol = sitk.Multiply(vol, 1/self.vol_norm_factors[case])
            vols[case] = vol
        return vols

    def crop_vol(self, vol, origin):
        from_bottom = [origin[0]-64, origin[1]-64, origin[2]-64]     
        from_Top    = [160 - origin[0], 160 - origin[1], 80 - origin[2]]
        #print('{} --- vol crop: bottom {}, top {}'.format(threading.currentThread().getName(), from_bottom, from_Top))
        vol_cropped = sitk.Crop(vol, from_bottom, from_Top)
        return vol_cropped

    def augmentation_translation(self, point):        
        # x = int(random.uniform(-25,25))
        # y = int(random.uniform(-25,25))
        # z = int(random.uniform(-25,25))   
        translation = self.augmentation_parameters['tra']
        x0 = translation['x'][0] 
        x1 = translation['x'][1]
        x = int(random.uniform(x0, x1))

        y0 = translation['y'][0] 
        y1 = translation['y'][1]        
        y = int(random.uniform(y0, y1))

        z0 = translation['z'][0] 
        z1 = translation['z'][1]
        z = int(random.uniform(z0, z1))
        translated_point = [point[0] + x, point[1] + y, point[2] + z]
        return translated_point

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.epochs_lived += 1

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        ### Initialization
        X = np.empty([self.batch_size, SIZE_Y,SIZE_X, 4], dtype=np.float32)
        Y = np.empty([self.batch_size, SIZE_Y,SIZE_X, 3], dtype=np.float32)

        # X = np.empty([self.batch_size, SIZE_Y,SIZE_X, 1], dtype=np.float32)
        # Y = np.empty([self.batch_size, SIZE_Y,SIZE_X, 1], dtype=np.float32)


        ### Generate data
        for i, ID in enumerate(list_IDs_temp):
            ### ID is data frame file name
            data_filename  = ID
            nav_filename   = data_filename.replace('_D_', '_N_')
            split_filename = data_filename.split('/')

            ### extract case name
            case = split_filename[-2]
            prep.sanity_check_Proband(nav_filename, data_filename, ID, self.logger)
            
            ### augmentation
            # if self.augmentation:
            #     origin_augmented = self.augmentation_translation(self.vol_origins[case])
            # else:
            origin_augmented = self.vol_origins[case]

            ### crop volume
            vol_cropped = self.crop_vol(self.vols[case], origin_augmented)

            ### load and resample
            nav_slice  = prep.load_and_preprocess_slice(nav_filename,  newSpacing=[4.0, 4.0, 4.0])
            data_slice = prep.load_and_preprocess_slice(data_filename, newSpacing=[4.0, 4.0, 4.0])

            ### intensity normalization
            # nav_slice  = sitk.Multiply(nav_slice,  1/self.slice_norm_factors[case])
            # data_slice = sitk.Multiply(data_slice, 1/self.slice_norm_factors[case])

            ### check if data frame was acquired after navigator farme
            prep.sanity_check_chronology(nav_filename, data_filename, self.logger)

            ### resample navigator and dataslice to vol voxel dimensions
            nav_slice_res  = prep.resampleSliceToVolme(nav_slice,  vol_cropped, sitk.sitkNearestNeighbor, 0)
            data_slice_res = prep.resampleSliceToVolme(data_slice, vol_cropped, sitk.sitkNearestNeighbor, 0)

            ### crop a slice from the volume at nav position
            nav_index        = vol_cropped.TransformPhysicalPointToIndex(nav_slice.GetOrigin())
            vol_slice_navpos = prep.cropSliceFromVolume_X(vol_cropped, nav_index[0])

            ### crop a slice from the volume at data position
            data_index        = vol_cropped.TransformPhysicalPointToIndex(data_slice.GetOrigin())
            vol_slice_datapos = prep.cropSliceFromVolume_X(vol_cropped, data_index[0])

            ### get image array data
            vol_slice_datapos_arr = sitk.GetArrayFromImage(vol_slice_datapos)        
            vol_slice_navpos_arr  = sitk.GetArrayFromImage(vol_slice_navpos)    
            nav_slice_res_arr     = sitk.GetArrayFromImage(nav_slice_res)
            data_slice_res_arr    = sitk.GetArrayFromImage(data_slice_res)

            ### intensity standartization
            # vol_slice_datapos_arr = prep.per_image_standardization_numpy(vol_slice_datapos_arr)
            # vol_slice_navpos_arr  = prep.per_image_standardization_numpy(vol_slice_navpos_arr)
            # #print('BEFOR standartization nav_mean:{}, data_mean:{}, nav_std:{}, data_std:{}'.format(np.mean(nav_slice_res_arr), np.mean(data_slice_res_arr), np.std(nav_slice_res_arr), np.std(data_slice_res_arr)))
            # nav_slice_res_arr     = prep.per_image_standardization_numpy(nav_slice_res_arr)
            # data_slice_res_arr    = prep.per_image_standardization_numpy(data_slice_res_arr)
            # #print('AFTER standartization nav_mean:{}, data_mean:{}, nav_std:{}, data_std:{}'.format(np.mean(nav_slice_res_arr), np.mean(data_slice_res_arr), np.std(nav_slice_res_arr), np.std(data_slice_res_arr)))
            
            vol_mean    = self.vol_stdandardization_params[0][case]
            vol_adj_std = self.vol_stdandardization_params[1][case]
            vol_slice_datapos_arr = (vol_slice_datapos_arr - vol_mean) / vol_adj_std
            vol_slice_navpos_arr  = (vol_slice_navpos_arr  - vol_mean) / vol_adj_std

            #print('BEFOR standartization nav_mean:{}, data_mean:{}, nav_std:{}, data_std:{}'.format(np.mean(nav_slice_res_arr), np.mean(data_slice_res_arr), np.std(nav_slice_res_arr), np.std(data_slice_res_arr)))
            slice_mean    = self.slice_stdandardization_params[0][case]
            slice_adj_std = self.slice_stdandardization_params[1][case]
            nav_slice_res_arr     = (nav_slice_res_arr  - slice_mean) / slice_adj_std
            data_slice_res_arr    = (data_slice_res_arr - slice_mean) / slice_adj_std
            #print('AFTER standartization nav_mean:{}, data_mean:{}, nav_std:{}, data_std:{}'.format(np.mean(nav_slice_res_arr), np.mean(data_slice_res_arr), np.std(nav_slice_res_arr), np.std(data_slice_res_arr)))

            prep.sanity_check_slicePos(nav_slice_res, data_slice_res, vol_slice_navpos, vol_slice_datapos, vol_cropped, ID, self.logger)
            
            ### store augmented sample
            X[i, :, :, 0] = vol_slice_datapos_arr[:,:,0]
            X[i, :, :, 1] = vol_slice_navpos_arr[:,:,0]
            X[i, :, :, 2] = nav_slice_res_arr[:,:,0]
            X[i, :, :, 3] = self.mean_slices[int(self.slice_pos)][:,:,0]
            Y[i, :, :, 0] = data_slice_res_arr[:,:,0]
            Y[i, :, :, 1] = data_slice_res_arr[:,:,0]
            Y[i, :, :, 2] = data_slice_res_arr[:,:,0]

            # X[i, :, :, 0] = nav_slice_res_arr[:,:,0]
            # Y[i, :, :, 0] = data_slice_res_arr[:,:,0]

            self.sample_count += 1
        return X, Y

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print( 'List_ID', type(self.list_IDs))
        #print('batch', type(self.batch_size))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes for the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y  = self.__data_generation(list_IDs_temp)
        self.batches_generated += 1
        self.actual_bs.append(len(list_IDs_temp))

        return X, Y
    
    def loadMeanSlices(self):
        fileFolder = r'/cache/gino/Data_for_DeepLearning'
        regEx = 'mean_slice_*.nrrd'
        filenames = sorted(glob.glob(os.path.join(fileFolder, regEx)))
        mean_slices_temp = {}
        for fn in filenames:
            s_pos = int(fn.split('_')[-1].split('.')[0])
            img = sitk.ReadImage(fn)
            mean_slices_temp[s_pos] = sitk.GetArrayFromImage(img)
        return mean_slices_temp

class DataGenerator2D_aug(keras.utils.Sequence):
    def __init__(self, cases='', data_dir='', IDs = [], size_X=64, size_Y=64, batch_size=32, spacing=[4.0, 4.0, 4.0], shuffle=True, comment='_', log_dir='.', 
                 augmentation=True, 
                 aug_rot      = 10,  ### augmentation rotation range in degree 
                 aug_w_shift  = 10,  ### augmentation width shift range in voxel                
                 aug_h_shift  = 10,  ### augmentation height shift range in voxel
                 aug_zoom_min = 0.8, ### augmentation min zoom     
                 aug_zoom_max = 1.3, ### augmentation max zoom                    
                 use_explicit_dist_info=False):

        self.set_batch_size(batch_size)
        self.shuffle = shuffle
        self.size_X = size_X
        self.size_Y = size_Y
        self.spacing = spacing


        if len(IDs) > 0:
            self.init_data_IDs(IDs, data_dir)
        elif cases != '' and data_dir != '':
            self.init_data_case(cases, data_dir)
        else:
            print('no data paths are given')
        
        self.comment = comment
        self.logger = utils.function_logger(os.path.join(log_dir, 'data_generation'))
        self.augmentation = augmentation
        self.aug_args = dict(rotation_range=aug_rot, 
                             width_shift_range=aug_w_shift, 
                             height_shift_range=aug_h_shift, 
                             zoom_range=[aug_zoom_min, aug_zoom_max], 
                             data_format="channels_last", 
                             dtype=np.float32)
        self.use_explicit_dist_info = use_explicit_dist_info
    
    def set_batch_size(self, batch_size):        
        self.batch_size = batch_size
    
    def init_data_case(self,cases, data_dir):
        reg  = '*_N_*.nii.gz'
        for case in cases:
            proband_samples = sorted(glob.glob(os.path.join(data_dir, case, reg)))
            self.init_data_IDs(proband_samples, data_dir)


    def init_data_IDs(self, IDs, data_dir):
        self.IDs  = []
        self.vols = {}
        self.slice_norm_params = {}
        self.vol_norm_params   = {}
        
        self.add_data_IDs(IDs, data_dir)
        
            
    def add_data_IDs(self, IDs, data_dir):
        self.IDs = self.IDs + IDs
        for i in IDs:
            ### extract case name
            split_filename = i.split('/')
            case = split_filename[-2]

            if not case in self.slice_norm_params:
                with open(os.path.join(data_dir, case, 'normalization_factors.json'), 'r') as fp:
                    data = json.load(fp)
                    self.slice_norm_params[case] = {'mean': data['slice_mean'], 'std': data['slice_adjusted_std']}
                    self.vol_norm_params[case]   = {'mean': data['vol_mean'],   'std': data['vol_adjusted_std']}
            
            if not case in self.vols:
                vol_filename = os.path.join(data_dir, case, case +'_Volume_1.nii.gz')
                ### oh my god, die voxel anzahl muss immer im code geändert werden
                # vol = prep.load_and_preprocess_vol(vol_filename, newSpacing= self.spacing, newSize=[160, 160, 80])
                vol = prep.load_and_preprocess_vol(vol_filename, newSpacing = self.spacing, newSize=[209, 209, 160])

                ### crop volume to 160 x 64 x 64
                with open(os.path.join(data_dir, 'vol_origins_phy.json'), mode='r') as f:
                    vol_origins = json.load(f)
                    index = vol.TransformPhysicalPointToIndex(vol_origins[case])
                                        
                    # from_bottom = [30, index[1]-self.size_X, vol.GetSize()[2]-64]     
                    # from_Top    = [30, vol.GetSize()[1] - index[1], 0]
                    from_bottom = [0, index[1]-self.size_X, vol.GetSize()[2]-128] #     
                    from_Top    = [0, vol.GetSize()[1] - index[1], 0]
                    vol = sitk.Crop(vol, from_bottom, from_Top)
                    sitk.WriteImage(vol,  os.path.join('/data/gino/4D_MRI_CNN/output/debug/', 'corpped_vol.nrrd'))
                
                self.vols[case] = vol

        self.indices = np.arange(len(self.IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    ### inference: 'no', 'single_slice', '3D_from_2D'
    def __data_generation(self, batch_IDs, inference='no', return_RefVol=False, ref_vols=[], caseIDs=[], slice_pos_px=[]):
        
        if inference == '3D_from_2D':
            keys = list(self.vols.keys())
            tensor_size = self.batch_size * self.vols[keys[0]].GetSize()[0]
        else:            
            tensor_size = self.batch_size
            
        X = np.empty([tensor_size, self.size_Y, self.size_X, 3], dtype=np.float32)
        Y = np.empty([tensor_size, self.size_Y, self.size_X, 1], dtype=np.float32)


        ### Generate data
        for i, nav_filename in enumerate(batch_IDs):
            data_filename = nav_filename.replace('_N_', '_D_')

            ### determine case of sampel
            case = nav_filename.split('/')[-2]

            ### load files
            nav_slice  = prep.load_and_preprocess_slice(nav_filename,  newSpacing=self.spacing)
            data_slice = prep.load_and_preprocess_slice(data_filename, newSpacing=self.spacing)
           
            vol = self.vols[case]
            vol_size = vol.GetSize()
            
            ### resample navigator and dataslice to vol voxel dimensions
            nav_slice_res  = prep.resampleSliceToVolme(nav_slice,  vol, sitk.sitkNearestNeighbor, 0)
            data_slice_res = prep.resampleSliceToVolme(data_slice, vol, sitk.sitkNearestNeighbor, 0)

            ### crop a slice from the volume at nav position
            nav_index        = vol.TransformPhysicalPointToIndex(nav_slice_res.GetOrigin())
            vol_slice_navpos = prep.cropSliceFromVolume_X(vol, nav_index[0])

            ### normalize images
            nav_slice         = self.normalize_slice(     sitk.GetArrayFromImage(nav_slice_res    )[:,:,0], case)
            vol_slice_navpos  = self.normalize_vol_slice( sitk.GetArrayFromImage(vol_slice_navpos )[:,:,0], case)
            data_slice        = self.normalize_slice(     sitk.GetArrayFromImage(data_slice_res   )[:,:,0], case) 
            
            ### volume slice indices for which a prediction is wanted
            if inference == '3D_from_2D':
                vol_slice_datapos_indices = range(vol_size[0])
            else:
                data_index = vol.TransformPhysicalPointToIndex(data_slice_res.GetOrigin())
                vol_slice_datapos_indices = [data_index[0]]
                
            ### prepare dist info
            dist_normed = self.generate_dist(nav_slice_res, data_slice_res)  
            
            ### crop a slice from the volume at data position(s)
            for j, pos in enumerate(vol_slice_datapos_indices):
                temp = prep.cropSliceFromVolume_X(vol, pos)
                vol_slice_datapos = self.normalize_vol_slice( sitk.GetArrayFromImage(temp)[:,:,0], case)

                ind = i * len(vol_slice_datapos_indices) + j
                
                X[ind, :, :, 0] = nav_slice
                X[ind, :, :, 1] = vol_slice_navpos
                X[ind, :, :, 2] = vol_slice_datapos
                # X[ind, :, :, 3] = dist_normed
                Y[ind, :, :, 0] = data_slice
                slice_pos_px.append(data_index[0])
            
            if return_RefVol:
                ref_vols.append(self.vols[case])
            caseIDs.append(case)
                
        if self.augmentation:
            (X, Y) =  self.augment(X, Y)
            
        return (X, Y)
    
    
    def generate_dist(self, nav_slice_res, data_slice_res):
        dist_normed = np.full(sitk.GetArrayFromImage(nav_slice_res).shape, 0, dtype=np.float32)[:,:,0]  
        if self.use_explicit_dist_info:
            nav_data_dist            = nav_slice_res.GetOrigin()[0] - data_slice_res.GetOrigin()[0]
            nav_data_dist_normed     = nav_data_dist / 176.0 ### 176 mm is the negative max distance in our data
            dist_normed              = np.full(sitk.GetArrayFromImage(nav_slice_res).shape, nav_data_dist_normed, dtype=np.float32)[:,:,0]  
            line_pos                 = math.floor(nav_data_dist_normed * 30) + 32 ### scale it to img size
            dist_normed[:, line_pos] = 1.5
        return dist_normed
            
    def normalize_slice(self, slice_arr, case):
        return (slice_arr - self.slice_norm_params[case]['mean']) / self.slice_norm_params[case]['std']
        
                
    def normalize_vol_slice(self, vol_slice_arr, case):
        return (vol_slice_arr - self.vol_norm_params[case]['mean']) / self.vol_norm_params[case]['std']
       
    
    def augment(self, X, Y):
        dist = X[:, :, :, 3:4] 
        data = np.concatenate((X[:, :, :, :3] , Y), axis=3)
        aug = tf.keras.preprocessing.image.ImageDataGenerator(**self.aug_args)
        
        for data_batch, y_batch in aug.flow(data, Y, batch_size=self.batch_size, shuffle=False):
            X_train = data_batch[:,:,:,:3]
            X_train = np.concatenate((X_train, dist), axis=3)
            Y_train = data_batch[:,:,:,3:]
            return(X_train, Y_train)
        
    ### inference: 'no', 'single_slice', '3D_from_2D'
    def __getitem__(self, index, inference='no', return_RefVol=False, ref_vols=[], caseIDs=[], batch_IDs=[], slice_pos_px=[]):
        'Generate one batch of data'
        # Generate indices for the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]


        # Find list of IDs
        batch_IDs_temp = [self.IDs[i] for i in batch_indices]
        
        #copy batch ids to output parameter
        for bid in batch_IDs_temp:
            batch_IDs.append(bid)
            
        # Generate data
        X, Y  = self.__data_generation(batch_IDs_temp, inference=inference, return_RefVol=return_RefVol, ref_vols=ref_vols, caseIDs=caseIDs, slice_pos_px=slice_pos_px)
        return X, Y
    
    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.IDs) / self.batch_size))




def split_samples(samples=[], split=[0.6]):
    if len(split) == 1:
        split_train = split[0]
        split_val   = 1.0 - split_train
    if len(split) == 2:
        split_train = split[0]
        split_val   = split[1]

    if split_train + split_val > 1.0:
        print('splits add up to more than 1')
        exit()

    rng     = np.random.default_rng()
    indices = np.arange(len(samples))
    indices = rng.permutation(indices)

    train_size = math.floor(len(samples) * split_train)
    val_size   = math.floor(len(samples) * split_val)

    train_sampling = indices[:train_size]
    val_sampling   = indices[train_size:train_size+val_size]

    samples       = np.array(samples)
    train_samples = samples[train_sampling].tolist()
    val_samples   = samples[val_sampling].tolist()
    return train_samples, val_samples


def split_samples_seq(samples=[], split=[0.6]):
    if len(split) == 1:
        split_train = split[0]
        split_val   = 1.0 - split_train
    if len(split) == 2:
        split_train = split[0]
        split_val   = split[1]

    if split_train + split_val > 1.0:
        print('splits add up to more than 1')
        exit()
        
    train_samples = []
    val_samples   = []
    for s in samples:
        seq    = np.array(s) ### oh it's not even necessary anymore to use an np array, but never change a running system
        train_size = math.floor(len(seq) * split_train)
        val_size   = math.floor(len(seq) * split_val)

        train_samples = train_samples + seq[:train_size].tolist()
        val_samples   = val_samples   + seq[train_size:train_size+val_size].tolist()

    return train_samples, val_samples

def grab_samples_seq(series={}, ranges={}):
    result = []
    for k in ranges:
        start  = ranges[k][0]
        end    = ranges[k][1]
        result = result + series[k][start:end]
    return result

def grab_samples_keep_seq(series={}, ranges={}):
    ids = {}
    for k in ranges:
        start  = ranges[k][0]
        end    = ranges[k][1]        
        ids[int(k)] = series[int(k)][start:end]
    return ids
