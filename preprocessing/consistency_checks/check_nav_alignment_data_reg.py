import numpy as np
import nibabel as nib
import matplotlib.image
import glob
import os
import csv
import cv2
import SimpleITK as sitk
import json
import preprocess as prep

out_folder = 'some_path'


train_files_dir = 'some_path'
train_cases = os.listdir(train_files_dir) 
nav_xs_reg = []
out_reg_blend = np.full((64,64),600, dtype='float32')
out_reg_blend_vol = np.full((64,64,64),600, dtype='float32')
cropped_vol_origins = {}
for case in train_cases:
  case = case.replace('\n', '')
  vol_filename   = sorted(glob.glob(os.path.join(train_files_dir, case,'*_Volume_1.nii.gz')))[0]
  data_filename  = sorted(glob.glob(os.path.join(train_files_dir, case,'*_D_0001.nii.gz')))[0]
  nav_filename   = sorted(glob.glob(os.path.join(train_files_dir, case,'*_N_0001.nii.gz')))[0]
  vol            = prep.load_and_preprocess_vol(vol_filename,    newSpacing=[4.0, 4.0, 4.0], newSize=[160, 160, 80])
  data           = prep.load_and_preprocess_slice(data_filename, newSpacing=[4.0, 4.0, 4.0])
  nav            = prep.load_and_preprocess_slice(nav_filename,  newSpacing=[4.0, 4.0, 4.0])

  c = vol.TransformPhysicalPointToIndex(data.GetOrigin())
  cropped_vol_origins[case] = c
  from_bottom = [c[0]-63, c[1]-63, 16]
  from_Top    = [159 - c[0], 159 - c[1], 0]
  vol_cropped = sitk.Crop(vol, from_bottom, from_Top)
  vol_cropped_array = sitk.GetArrayFromImage(vol_cropped)

  s_pt = vol.TransformIndexToPhysicalPoint([0,70,0])
  s_c = vol_cropped.TransformPhysicalPointToIndex(s_pt)
  out = vol_cropped_array[:,s_c[1],:]

  nav_xs_reg.append(vol_cropped.TransformPhysicalPointToIndex(nav.GetOrigin())[0])
  out[:, nav_xs_reg[-1]] = 1
  out = np.fliplr(np.flipud(out))
  out_reg_blend = out_reg_blend + out

  out_vol = vol_cropped_array
  out_vol[:, :, nav_xs_reg[-1]] = 1
  out_reg_blend_vol = out_reg_blend_vol + out_vol

  print("{} Nav x coord {}px".format(case, nav_xs_reg[-1]))


name = os.path.join('some_path', 'vol_origins.json')
with open(name, 'w') as f:
  json.dump(cropped_vol_origins, f, indent=4)

out_reg_blend_vol = sitk.GetImageFromArray(out_reg_blend_vol)
out_reg_blend_vol.CopyInformation(vol_cropped)
name = os.path.join(out_folder, 'out_reg_blend_vol.nrrd')
sitk.WriteImage(out_reg_blend_vol, name)

name = os.path.join(out_folder, case + '_liver_reg_blend.png')
matplotlib.image.imsave(name, out_reg_blend / len(train_cases))

nav_xs_reg = np.array(nav_xs_reg)
left_deviation = round((np.amin(nav_xs_reg) - np.mean(nav_xs_reg)), 1)
right_deviation = round((np.amax(nav_xs_reg) - np.mean(nav_xs_reg)), 1)
std = round(np.std(nav_xs_reg), 1)
print("Nav x coord srd = {}px, max_left deviation = {}px, max_right deviation = {}px".format(std, left_deviation, right_deviation))
print("Nav x coord srd = {}mm, max_left deviation = {}mm, max_right deviation = {}mm".format(std * 2.4, left_deviation * 2.4, right_deviation * 2.4))


### check if data slice is in cropped vol
slice_norm = 'slice_double_mean'
vol_norm   = 'vol_double_mean'
slice_norm_factor = 1.0
vol_norm_factor = 1.0
with open(os.path.join(train_files_dir, case, 'normalization_factors.json'), 'r') as fp:
    data = json.load(fp)
    slice_norm_factor = data[slice_norm]
    vol_norm_factor   = data[vol_norm]

vol_origins = cropped_vol_origins
for case in train_cases:
  case = case.replace('\n', '')
  vol_filename   = sorted(glob.glob(os.path.join(train_files_dir, case,'*_Volume_1.nii.gz')))[0]
  nav_filename   = sorted(glob.glob(os.path.join(train_files_dir, case,'*_N_0001.nii.gz')))[0]
  data_filename  = sorted(glob.glob(os.path.join(train_files_dir, case,'*_D_0001.nii.gz')))[0]

  # preprocess and create volumes
  vol = prep.load_and_preprocess_vol(vol_filename, newSpacing=[4.0, 4.0, 4.0], newSize=[160, 160, 80])
  vol = sitk.Multiply(vol, 1.0/vol_norm_factor)
  
  ### crop to 64 x 64 x 64
  from_bottom = [vol_origins[case][0]-63, vol_origins[case][1]-63, 16]     
  from_Top    = [159 - vol_origins[case][0], 159 - vol_origins[case][1], 0]
  vol_cropped = sitk.Crop(vol, from_bottom, from_Top)

  slice_n = prep.load_and_preprocess_slice(nav_filename, newSpacing=[4.0, 4.0, 4.0])
  slice_d = prep.load_and_preprocess_slice(data_filename, newSpacing=[4.0, 4.0, 4.0])

  slice_n = sitk.Multiply(slice_n, 1.0/slice_norm_factor)
  slice_d = sitk.Multiply(slice_d, 1.0/slice_norm_factor)

  # resample slices to volume
  vol_n_cropped = prep.resampleToReference(slice_n, vol_cropped, sitk.sitkNearestNeighbor, 0)
  vol_d_cropped = prep.resampleToReference(slice_d, vol_cropped, sitk.sitkNearestNeighbor, -1)  # -1 is for masking in loss function

  result_name = case + '_d.nrrd'
  sitk.WriteImage(vol_d_cropped, os.path.join('some_path/debug', result_name))