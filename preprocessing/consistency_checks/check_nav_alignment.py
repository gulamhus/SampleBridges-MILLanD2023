import numpy as np
import nibabel as nib
import matplotlib.image
import glob
import os
import csv
import cv2
import dataLoading
import SimpleITK as sitk

out_folder = 'some_path'

spine_centers = {}
with open('some_path/vol_alignment.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    spine_centers = {rows[0]:[int(rows[1]), int(rows[2]), int(rows[3])] for rows in reader}


train_files_dir = 'some_path/4D_MRI_data/data_init/train'
train_cases = os.listdir(train_files_dir) 
nav_xs = []
nav_xs_reg = []
out_blend = np.full((80,160),600, dtype='float32')
out_reg_blend = np.full((64,96),600, dtype='float32')
for case in train_cases:
    case = case.replace('\n', '')
    sample_vol   = sorted(glob.glob(os.path.join(train_files_dir, case,'*_Volume_1.nii.gz')))
    data_loader = dataLoading.dataLoader()
    vol = data_loader.load_and_preprocess_vol(sample_vol[0], 1)
    

    nav_filename = sorted(glob.glob(os.path.join(train_files_dir, case,'*_N_0001.nii.gz')))[0]
    slice_n = sitk.ReadImage(nav_filename)
    
    slice_n.SetSpacing([slice_n.GetSpacing()[0], slice_n.GetSpacing()[1], 4.0])
    #print("Nav coord {} in vol {}".format(vol.TransformPhysicalPointToIndex(slice_n.GetOrigin()), vol.GetSize()))
    #print("Vol size = {}, spacing = {}".format(vol.GetSize(),vol.GetSpacing()))
    
    s_c = [0,60,0]
    s_pt = vol.TransformIndexToPhysicalPoint(s_c)
    vol_array = sitk.GetArrayFromImage(vol)
    out = vol_array[:,s_c[1],:]

    nav_xs.append(vol.TransformPhysicalPointToIndex(slice_n.GetOrigin())[0])
    out[:, nav_xs[-1]] = 1
    out = np.fliplr(np.flipud(out))
    out_blend = out_blend + out
    print("{} Nav x coord {}px".format(case, nav_xs[-1]))
    #name = os.path.join(out_folder, case + '.png')
    #matplotlib.image.imsave(name, out)

    c_x = int(spine_centers[case][0] / 2)
    c_y = int(spine_centers[case][1] / 2)
    from_bottom = [c_x-30, c_y-25, 16]
    from_Top    = [160-(c_x+66), 160-(c_y+71), 0]
    vol_cropped = sitk.Crop(vol, from_bottom, from_Top)
    vol_cropped_array = sitk.GetArrayFromImage(vol_cropped)
    s_c = vol_cropped.TransformPhysicalPointToIndex(s_pt)
    out = vol_cropped_array[:,s_c[1],:]

    nav_xs_reg.append(vol_cropped.TransformPhysicalPointToIndex(slice_n.GetOrigin())[0])
    out[:, nav_xs_reg[-1]] = 1
    out = np.fliplr(np.flipud(out))
    out_reg_blend = out_reg_blend + out
    #name = os.path.join(out_folder, case + '_reg.png')
    #matplotlib.image.imsave(name, out)
    #print("{} Nav x coord {}px".format(case, nav_xs_reg[-1]))


# name = os.path.join(out_folder, case + '_blend.png')
# matplotlib.image.imsave(name, out_blend / len(train_cases))

# name = os.path.join(out_folder, case + '_reg_blend.png')
# matplotlib.image.imsave(name, out_reg_blend / len(train_cases))

nav_xs = np.array(nav_xs)
left_deviation = round((np.amin(nav_xs) - np.mean(nav_xs)), 1)
right_deviation = round((np.amax(nav_xs) - np.mean(nav_xs)), 1)
std = round(np.std(nav_xs), 1)
print("Nav x coord srd = {}px, max_left deviation = {}px, max_right deviation = {}px".format(std, left_deviation, right_deviation))
print("Nav x coord srd = {}mm, max_left deviation = {}mm, max_right deviation = {}mm".format(std * 2.4, left_deviation * 2.4, right_deviation * 2.4))

# nav_xs_reg = np.array(nav_xs_reg)
# left_deviation = round((np.amin(nav_xs_reg) - np.mean(nav_xs_reg)), 1)
# right_deviation = round((np.amax(nav_xs_reg) - np.mean(nav_xs_reg)), 1)
# std = round(np.std(nav_xs_reg), 1)
# print("Nav x coord srd = {}px, max_left deviation = {}px, max_right deviation = {}px".format(std, left_deviation, right_deviation))
# print("Nav x coord srd = {}mm, max_left deviation = {}mm, max_right deviation = {}mm".format(std * 2.4, left_deviation * 2.4, right_deviation * 2.4))