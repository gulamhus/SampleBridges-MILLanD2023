import numpy as np
import nibabel as nib
import matplotlib.image
import glob
import os

train_files_dir = ''

### check volume alignment
with open('Folds/Fold1_train.txt') as f:
  train_cases = f.readlines()

  for case in train_cases:
          case = case.replace('\n', '')
          sample_vols = sorted(glob.glob(os.path.join(train_files_dir, case,'*_Volume_*.nii.gz')))
          for sample in sample_vols:
            vol_nii = nib.load(sample)
            vol_arr = np.float32(vol_nii.get_fdata(caching='unchanged'))
            ax = vol_arr[:, :, 36]
            name = case + '_align_img.png'
            matplotlib.image.imsave( os.path.join('/cache/anneke/4D_MRI_data/output/debug', name), ax)
            print(sample)

            
### check nav alignment in registered volumes

