import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import json
import preprocess as prep

train_data_path = '/cache/gino/Data_for_DeepLearning/'
train_files_dir = '/cache/gino/Data_for_DeepLearning/train'

if __name__ == '__main__':
    os.chdir(os.path.expanduser('~/projects/4D_MRI_CNN/'))
    f = open('Folds/Fold1_train_all.txt')
    probands = f.readlines()


    print('Checking training data intensity distributen ...')    

    ### loading lookup
    with open(os.path.join(train_data_path, 'lookup.json'), 'r') as f:
        lookup = json.load(f)

    vol_origins = {}
    with open(os.path.join(train_data_path, 'vol_origins.json'), mode='r') as f:
        vol_origins = json.load(f)
        

    for s in lookup:
        if lookup[s]: ### if dictionary contains entries
            cases_for_slice = list(lookup[s].keys())
            samples_all = []
            for c in cases_for_slice:
                regEx = os.path.join(train_files_dir, c, '*S_{}_D_*.nii.gz'.format(lookup[s][c]))
                samples = glob.glob(regEx)
                samples_all += samples
            if len(samples_all) > 0:
               print('todo')
    