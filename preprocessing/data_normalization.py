import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import json
import preprocess as prep
import math

train_data_path = '/cache/gino/Data_for_DeepLearning/'
train_files_dir = '/cache/gino/Data_for_DeepLearning/test'

### calculate mean and adjusted std
def mean_and_adjusted_std(fileNames):
    img = sitk.GetArrayFromImage(sitk.ReadImage(fileNames[0]))
    img_shape = img.shape
    imgs = np.zeros((len(fileNames), img_shape[1], img_shape[2]))

    i = 0
    for fn in tqdm(fileNames):
        img = sitk.GetArrayFromImage(sitk.ReadImage(fn))
        imgs[i, :, :] = img 
        i += 1
    mean   = np.mean(imgs)
    stddev = np.std(imgs)
    adjusted_stddev = max(stddev, 1.0/math.sqrt(imgs.size))
    return (mean, adjusted_stddev)

### calculate mean and adjusted std for vols
def mean_and_adjusted_std_vol(fileName):
    img = sitk.GetArrayFromImage(sitk.ReadImage(fileName))
    mean   = np.mean(img)
    stddev = np.std(img)
    adjusted_stddev = max(stddev, 1.0/math.sqrt(img.size))
    return (mean, adjusted_stddev)


    max_intensity = 0
    for fn in tqdm(fileNames):
        img = sitk.ReadImage(fn)
        img_arr = sitk.GetArrayFromImage(img)
        max_temp = np.amax(img_arr)
        max_intensity = max(max_temp, max_intensity)
    
    return max_intensity

if __name__ == '__main__':
    # os.chdir(os.path.expanduser('~/projects/4D_MRI_CNN/'))
    # f = open('Folds/Fold1_train_all.txt')
    # probands = f.readlines()

    probands = os.listdir(train_files_dir)

    for p in probands:
        if not os.path.isdir(os.path.join(train_files_dir,p)):
            continue
        
        print('Calculating standardization factors for ', p)
        p = p.replace('\n', '')
        norm_factors = {}
        fileName = os.path.join(train_files_dir, p, 'normalization_factors.json')
        if os.path.isfile(fileName):
            print('skip {}'.format(fileName))
            continue

        cases = glob.glob(os.path.join(train_files_dir, p,'*_D_*.nii.gz'))
        (mean, adj_std) = mean_and_adjusted_std(cases)
        norm_factors['slice_mean']         = mean
        norm_factors['slice_adjusted_std'] = adj_std

        vol = glob.glob(os.path.join(train_files_dir, p,'*_Volume_*.nii.gz'))[0]
        (mean, adj_std) = mean_and_adjusted_std_vol(vol)
        norm_factors['vol_mean']           = mean
        norm_factors['vol_adjusted_std']   = adj_std

        with open(fileName, 'w') as fp:
            json.dump(norm_factors, fp, indent=4)
