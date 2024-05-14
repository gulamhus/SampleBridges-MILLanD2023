import numpy as np
import nibabel as nib
import matplotlib.image
import matplotlib.pyplot as plt
import glob
import os
import dicom
import re
from tqdm import tqdm
from preprocess import ms_from_DicomTag_AcquisitionTime

### ===================================
### shows that acquisition time in descrip tag of nifti is corresponding to the one extracted from dicom earlier
### ===================================
# file_dir = r'D:\med_Bilddaten\4D_Leber\DeepLearning\Data_for_DeepLearning\subject1'
# file_names = sorted(glob.glob(os.path.join(file_dir, '*_S_*.nii.gz')))

# parts = os.path.basename(file_names[0]).split('.')[0].split('_')
# t_n_last = float(parts[3])

# nii = nib.load(file_names[0])
# descrip = str(nii.header['descrip'])
# t_d_last = ms_from_DicomTag_AcquisitionTime(descrip.split(';')[1].split('=')[1])

# last_i = 0
# last_Tn = 0
# last_Td = 0
# for i, file_name in enumerate(file_names):           
#   parts = os.path.basename(file_name).split('.')[0].split('_')
#   t_n = float(parts[3])### time extracted from name

#   nii = nib.load(file_name)
#   descrip = str(nii.header['descrip'])

#   t_d = ms_from_DicomTag_AcquisitionTime(descrip.split(';')[1].split('=')[1]) ### time extracted from description
#   if abs(abs(t_n - t_n_last) - 170) < 15 and abs(abs(t_d - t_d_last) - 170) > 15: 
#     print('=========================')
#     print(file_name)
#     print(t_n - t_n_last, t_n)
#     print(t_d - t_d_last, t_d) 
#     print('dT_n =', t_n - last_Tn, ' dT_d =', t_d - last_Td, ' i =', i, ' di =', i-last_i)
#     last_i = i
#     last_Tn = t_n
#     last_Td = t_d

#   t_n_last = t_n
#   t_d_last = t_d

### ===================================
### plotting both time stamps next to each other showing that they are identical
### ===================================
# file_dir = r'D:\med_Bilddaten\4D_Leber\DeepLearning\Data_for_DeepLearning\subject1'
# file_names = sorted(glob.glob(os.path.join(file_dir, '*_S_*.nii.gz')))

# t_n_list = []
# t_d_list = []
# for file_name in file_names:           
#   parts = os.path.basename(file_name).split('.')[0].split('_')
#   t_n = float(parts[3]) ### time extracted from name
#   t_n_list.append(t_n)

#   nii = nib.load(file_name)
#   descrip = str(nii.header['descrip'])
#   t_d = ms_from_DicomTag_AcquisitionTime(descrip.split(';')[1].split('=')[1]) ### time extracted from description
#   t_d_list.append(t_d)
  

# t_n_arr = np.array(t_n_list)    
# t_d_arr = np.array(t_d_list)
# plt.plot(t_n_arr, 'r--')
# plt.plot(t_d_arr, 'b--')
# plt.show()


### ===================================
### checks pairs, if navigator frame is acquired befor the data frame 
### ===================================
def checkChronology(data_file_names):
  dt_arr =[]
  for data_file_name in tqdm(data_file_names):  
    data_nii = nib.load(data_file_name)

    nav_file_name = data_file_name.replace('_D_', '_N_')
    nav_nii = nib.load(nav_file_name) 

    data_descrip = str(data_nii.header['descrip'])
    t_d = ms_from_DicomTag_AcquisitionTime(data_descrip.split(';')[1].split('=')[1]) ### data time extracted from description

    nav_descrip = str(nav_nii.header['descrip'])
    t_n = ms_from_DicomTag_AcquisitionTime(nav_descrip.split(';')[1].split('=')[1]) ### nav time extracted from description

    dt = t_d - t_n
    dt_arr.append(dt)
    if abs(abs(dt) - 170) < 15:
      a = 1
      #print(t_d - t_n, t_d)
    else:
      print('=========================================================')
      print('=== Here is some thing strange!!!') 
      print('===', data_file_name)
      print('===', t_d - t_n, t_d)
      print('=========================================================')
  # dt_arr = np.array(dt_arr)    
  # plt.plot(dt_arr)
  # plt.show()

cases = ['subject1','subject2']
file_dir = r'D:\med_Bilddaten\4D_Leber\DeepLearning\Data_for_DeepLearning'
for case in tqdm(cases):
  data_file_names = sorted(glob.glob(os.path.join(file_dir, case, '*_D_*.nii.gz')))
  print('')
  print('checking', case)
  checkChronology(data_file_names)

### ===================================
### making motion readily visible to see if it's continuous or discontinuous
### ===================================
# def check_motion_continuidy(folder):
#   file_names = sorted(glob.glob(os.path.join(folder, '*.nii.gz')))
#   for file_name in tqdm(file_names):
#     nii = nib.load(file_name)
#     arr = np.float32(nii.get_fdata(caching='unchanged'))[:,:,0]
#     arr = np.flipud(np.transpose(arr))
#     name = re.sub(r"nii\.gz", "png", file_name)
#     matplotlib.image.imsave( os.path.join(folder, name), arr, cmap='gray')
            

# check_motion_continuidy(r'subject1')