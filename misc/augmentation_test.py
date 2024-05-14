 
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
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('/home/gino/projects/4D_MRI_CNN')
import preprocessing.preprocess as prep
import SimpleITK as sitk
import matplotlib 

from tqdm import tqdm
import json
import wandb
from wandb.keras import WandbCallback

from net import VGG_encoder
from data_generation import DataGenerator2D_aug


train_data_path = '/cache/gino/Data_for_DeepLearning/'

def visualize_augmentation(X, Y, skip=0, name_prefix='', save_path='/data/gino/4D_MRI_CNN/output/debug'):
  s = X.shape[0] ### number of samples
  x = X.shape[1]
  y = X.shape[2]
  for i in range(0,s,skip + 1):
    result = np.zeros((x,y*5), dtype=np.float32)
    result[:,:y]      = np.fliplr(np.flipud(     X[i,:,:,0]))
    result[:,y:y*2]   = np.fliplr(np.flipud(     X[i,:,:,1]))
    result[:,y*2:y*3] = np.fliplr(np.flipud(     X[i,:,:,2]))
    # result[:,y*3:y*4] = np.fliplr(np.flipud(     X[i,:,:,3]))
    result[:,y*4:y*5] = np.fliplr(np.flipud(     Y[i,:,:,0]))
    
    name = name_prefix + 'aug_' + str(i) + '.png'
    matplotlib.image.imsave( os.path.join(save_path, name), result, cmap="gray")

### get val data
cases = ['subjet1', 'subject2'] #list of folder names containing subject data
val_gen = DataGenerator2D_aug(cases=cases,  data_dir=os.path.join(train_data_path, 'train'),IDs = [], size_X=128, size_Y=128, spacing=[1.818, 1.818, 1.818], batch_size=20,  shuffle=False, comment='', log_dir='.', augmentation=True,              
                 aug_rot      = 3,  ### augmentation rotation range in degree 
                 aug_w_shift  = 10,  ### augmentation width shift range in voxel                
                 aug_h_shift  = 10,  ### augmentation height shift range in voxel
                 aug_zoom_min = 0.8, ### augmentation min zoom     
                 aug_zoom_max = 1.2,
                 use_explicit_dist_info=True)
for i in range(2):
  (X_val, Y_val) = val_gen.__getitem__(i)
  visualize_augmentation(X_val, Y_val, skip=0, name_prefix=str(i) + '_px_', save_path='/data/gino/4D_MRI_CNN/output/debug/2D_augmentation')