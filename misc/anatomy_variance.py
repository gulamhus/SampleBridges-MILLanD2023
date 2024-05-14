import glob
import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib 
sys.path.append('/home/gino/projects/4D_MRI_CNN')
import preprocessing.preprocess as prep

cases = ['subjet1', 'subject2'] #list of folder names containing subject data
reg  = '*_N_*.nii.gz'
for case in cases:
  ID = sorted(glob.glob(os.path.join('/cache/gino/Data_for_DeepLearning/train', case, reg)))[10]
  nav_slice  = prep.load_and_preprocess_slice(ID,  newSpacing=[4.0, 4.0, 4.0])
  img = np.flipud(sitk.GetArrayFromImage(nav_slice)[0,:,:])
  file_name = case + '_nav.png'
  matplotlib.image.imsave(os.path.join('/cache/gino/Data_for_DeepLearning/output/debug/anatomy_variance', file_name), img, cmap="gray")
