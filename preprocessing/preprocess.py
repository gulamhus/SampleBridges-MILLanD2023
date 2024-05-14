import numpy as np
import matplotlib.image
import glob
import os
import csv
import SimpleITK as sitk
from tqdm import tqdm
import json
import math
import threading

def resampleImage_hotFixe(inputImage, newSpacing, interpolator, defaultValue, new_origin=None, new_size=None):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    if new_size == None:
        wanted_size_in_old_spacing = inputImage.GetSize() ### voxel count
    else:
        wanted_size_in_old_spacing = new_size
        
    oldSpacing = inputImage.GetSpacing()
    newSize_x  = oldSpacing[0] / newSpacing[0] * wanted_size_in_old_spacing[0]
    newSize_y  = oldSpacing[1] / newSpacing[1] * wanted_size_in_old_spacing[1]
    newSize_z  = oldSpacing[2] / newSpacing[2] * wanted_size_in_old_spacing[2]
    new_size    = [int(math.floor(newSize_x)), int(math.ceil(newSize_y)), int(math.ceil(newSize_z))]
    # new_vol_size = (int(math.ceil(new_ve_px[0] - new_vo_px[0])), int(math.ceil(new_ve_px[1] - new_vo_px[1])), int(math.ceil(new_ve_px[2] - new_vo_px[2])))

    if new_origin == None:
        new_origin = inputImage.GetOrigin()
    else:
        print(new_origin)
        new_origin = [int(math.ceil(new_origin[0])), int(math.floor(new_origin[1]))-1, int(math.ceil(new_origin[2]))] ### -1 is a hot fix, workaround
        print(new_origin)
        
    filter = sitk.ResampleImageFilter()
    filter.SetOutputSpacing(newSpacing)
    filter.SetInterpolator(interpolator)
    filter.SetOutputOrigin(new_origin)
    filter.SetOutputDirection(inputImage.GetDirection())
    filter.SetSize(new_size)
    filter.SetDefaultPixelValue(defaultValue)
    outImage = filter.Execute(inputImage)

    return outImage

def resampleImage(inputImage, newSpacing, interpolator, defaultValue, new_origin=None, new_size=None):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImage = castImageFilter.Execute(inputImage)

    if new_size == None:
        wanted_size_in_old_spacing = inputImage.GetSize() ### voxel count
    else:
        wanted_size_in_old_spacing = new_size
        
    oldSpacing = inputImage.GetSpacing()
    newSize_x  = oldSpacing[0] / newSpacing[0] * wanted_size_in_old_spacing[0]
    newSize_y  = oldSpacing[1] / newSpacing[1] * wanted_size_in_old_spacing[1]
    newSize_z  = oldSpacing[2] / newSpacing[2] * wanted_size_in_old_spacing[2]
    new_size    = [int(newSize_x), int(newSize_y), int(newSize_z)]
    
    if new_origin == None:
        new_origin = inputImage.GetOrigin()
    else:
        new_origin = [int(new_origin[0]), int(new_origin[1]), int(new_origin[2])]
        
    filter = sitk.ResampleImageFilter()
    filter.SetOutputSpacing(newSpacing)
    filter.SetInterpolator(interpolator)
    filter.SetOutputOrigin(new_origin)
    filter.SetOutputDirection(inputImage.GetDirection())
    filter.SetSize(new_size)
    filter.SetDefaultPixelValue(defaultValue)
    outImage = filter.Execute(inputImage)

    return outImage

def resampleToReference(inputImg, referenceImg, interpolator, defaultValue):

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    inputImg = castImageFilter.Execute(inputImg)

    #sitk.WriteImage(inputImg,'input.nrrd')

    filter = sitk.ResampleImageFilter()
    filter.SetReferenceImage(referenceImg)
    filter.SetDefaultPixelValue(float(defaultValue)) ## -1
    # float('nan')
    filter.SetInterpolator(interpolator)

    outImage = filter.Execute(inputImg)

    return outImage


def resampleSliceToVolme(inputImg, referenceVol, interpolator=sitk.sitkNearestNeighbor, defaultValue=0):            
    res_vol     = resampleToReference(inputImg, referenceVol, interpolator, defaultValue)  
    slice_index = referenceVol.TransformPhysicalPointToIndex(inputImg.GetOrigin())   
    # print('{} --- slice_index: {}'.format(threading.currentThread().getName(), slice_index[0]))  
    return cropSliceFromVolume_X(res_vol, slicePos=slice_index[0])     

def cropSliceFromVolume_X(inputVolume, slicePos):           
    vol_size    = inputVolume.GetSize()
    from_bottom = [slicePos, 0, 0]     
    from_Top    = [vol_size[0] - 1 - slicePos, 0, 0]
    # print('{} --- slicePos: {}'.format(threading.currentThread().getName(), slicePos))  
    # print('{} --- Slice: bottom {}, top {}'.format(threading.currentThread().getName(),from_bottom, from_Top))
    return sitk.Crop(inputVolume, from_bottom, from_Top)

def loadNormalizationFactors(hp, jsonFilePath):
    slice_norm = ''
    vol_norm = ''
    if hp['NORMILIZATION'] == 'max':
        slice_norm = 'slice_max'
        vol_norm   = 'vol_max'
    elif hp['NORMILIZATION'] == 'double_mean':
        slice_norm = 'slice_double_mean'
        vol_norm   = 'vol_double_mean'

    slice_norm_factor = 1.0
    vol_norm_factor   = 1.0
    with open(jsonFilePath, 'r') as fp:
        data = json.load(fp)
        slice_norm_factor = data[slice_norm]
        vol_norm_factor   = data[vol_norm]
    return (slice_norm_factor, vol_norm_factor)

def loadStandardizationParameter(jsonFilePath):
    vol_stdandardization_params = {}
    with open(jsonFilePath, 'r') as fp:
        data = json.load(fp)
        vol_stdandardization_params['slice_mean']         = data['slice_mean']
        vol_stdandardization_params['slice_adjusted_std'] = data['slice_adjusted_std']
        vol_stdandardization_params['vol_mean']           = data['vol_mean']
        vol_stdandardization_params['vol_adjusted_std']   = data['vol_adjusted_std']
    return vol_stdandardization_params

def size_correction(image, ref_size):
    pad_x = (float(ref_size[0]) - float(image.GetSize()[0]))/2
    pad_y = (float(ref_size[1]) - float(image.GetSize()[1]))/2
    pad_z = (float(ref_size[2]) - float(image.GetSize()[2]))/2

    if pad_x < 0 or pad_y < 0 or pad_z < 0:
        print('could not pad image, new size {} is smaller than original {}'.format(ref_size, image.GetSize()))
        return image
    
    filter = sitk.ConstantPadImageFilter()
    filter.SetConstant(0)
    filter.SetPadUpperBound([int(math.floor(pad_x)),int(math.floor(pad_y)),int(math.floor(pad_z))])
    filter.SetPadLowerBound([int(math.ceil(pad_x)), int(math.ceil(pad_y)), int(math.ceil(pad_z))])

    return filter.Execute(image)

def load_and_preprocess_vol(filename, newSpacing, newSize=[]):
    vol = sitk.ReadImage(filename)    
    #print(vol.GetSize())
    # sitk.WriteImage(vol,  os.path.join('/data/gino/4D_MRI_CNN/output/debug/', 'load_vol.nrrd'))
    vol = resampleImage(vol, newSpacing, sitk.sitkLinear, 0)
    #print(vol.GetSize())
    # sitk.WriteImage(vol,  os.path.join('/data/gino/4D_MRI_CNN/output/debug/', 'resampled_vol.nrrd'))
    if newSize == []:
        return vol
    
    vol = size_correction(vol, newSize)
    #print(vol.GetSize())
    # sitk.WriteImage(vol,  os.path.join('/data/gino/4D_MRI_CNN/output/debug/', 'padded_vol.nrrd'))
    return vol

def load_and_preprocess_slice(filename, newSpacing):
    s = sitk.ReadImage(filename)
    s = resampleImage(s, newSpacing, sitk.sitkLinear, 0)
    return s

def resample(img, newSpacing):
    img = resampleImage(img, newSpacing, sitk.sitkLinear, 0)
    return img

def ms_from_DicomTag_AcquisitionTime(t_dcm):
    [hms_s, ms_s] = t_dcm.split(".")
    h = float(hms_s[0:2])
    m = float(hms_s[2:4])
    s = float(hms_s[4:6])
    ms = float(ms_s[0:3])
    ms = ms + 1000*s + 60*1000*m + 60*60*1000*h
    return ms

def normalize_numpy(img):
    min_val = np.min(img)
    max_vav = np.max(img)
    img = ((img - min_val) / (max_vav - min_val))
    return img

def per_image_standardization_numpy(img):
    mean   = np.mean(img)
    stddev = np.std(img)
    adjusted_stddev = max(stddev, 1.0/math.sqrt(img.size))
    return (img - mean) /  adjusted_stddev

def sanity_check_training_split(train_samples, val_samples, logger=-1):
    print('train val intersection', len(np.intersect1d(np.array(train_samples), np.array(val_samples))))
    print('split_train len={}, min={}'.format(len(train_samples), len(train_samples)*166/1000/60)) 
    print('split_val len={}, min={}'.format(len(val_samples), len(val_samples)*166/1000/60)) 
    print('split', len(train_samples) / (len(train_samples) + len(val_samples))) 

def sanity_check_chronology(nav_filename, data_filename, logger=-1):
    ### load nav file and get acquisition time
    reader = sitk.ImageFileReader()
    reader.SetFileName( nav_filename )
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    tinfo = reader.GetMetaData("ITK_FileNotes")
    t_n = ms_from_DicomTag_AcquisitionTime(tinfo.split(';')[1].split('=')[-1])  
    
    ### load data file and get acquisition time      
    reader = sitk.ImageFileReader()
    reader.SetFileName( data_filename )
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    tinfo = reader.GetMetaData("ITK_FileNotes")
    t_d = ms_from_DicomTag_AcquisitionTime(tinfo.split(';')[1].split('=')[-1])

    diff = t_d - t_n
    ### check if dt < 0 -> data slice acquired befor nav
    ### check if dt diverges greatly from 170ms
    if diff < 0 or abs(diff - 167) > 10:
        if logger:
            logger.info('Chronology......t_d - t_n = ' + str(diff) + ' ID = ' + nav_filename)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Warning: Data before Nav slice. Diff = {}, ID = {}'.format(diff, nav_filename))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

def sanity_check_slicePos(nav_slice, data_slice, vol_slice_atNavPos, vol_slice_atDataPos, vol, ID, logger):
    nav_index_1 = vol.TransformPhysicalPointToIndex(nav_slice.GetOrigin())
    nav_index_2 = vol.TransformPhysicalPointToIndex(vol_slice_atNavPos.GetOrigin())

    data_index_1 = vol.TransformPhysicalPointToIndex(data_slice.GetOrigin())
    data_index_2 = vol.TransformPhysicalPointToIndex(vol_slice_atDataPos.GetOrigin())

    if nav_index_1 != nav_index_2 or data_index_1 != data_index_2:
        if logger:
            logger.info('Sliceposition..... Nav =  {} -> {}; Data = {} -> {};  ID = {}'.format(nav_index_1, nav_index_2, data_index_1, data_index_2,ID))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Warning: Slice pos not matching. Nav =  {} -> {}; Data = {} -> {};  ID = {}'.format(nav_index_1, nav_index_2, data_index_1, data_index_2, ID))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

def sanity_check_Proband(nav_filename, data_filename, ID, logger=-1):
    ### extract case name    
    case1 = nav_filename.split('/')[-2]
    case2 = data_filename.split('/')[-2]
    if case1 != case2:
        if logger:
            logger.info('Porband..... Nav =  {} ; Data = {};  ID = {}'.format(case1, case2, ID))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Warning: Porband..... Nav =  {} ; Data = {};  ID = {}'.format(case1, case2, ID))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

def sanity_check_SampleNumber(ID_list, generatorName, slice_pos, logger=-1):
    case_histogram = {}
    for id in ID_list:
        case = id.split('/')[-2]
        #print(case)
        if not case in case_histogram:
            case_histogram[case] = 1
        else: 
            case_histogram[case] += 1
    
    if logger:
        logger.info('SampleNumber..... Slice Pos = {}; Generator = {}; Total Sample count = {}, Samples = {}'.format(slice_pos, generatorName, len(ID_list), case_histogram))


# def sanity_check_normalization(nav_slice, data_slice, vol, ID, logger=-1):
#     nav_mean                 = np.mean(nav_slice)
#     nav_std                  = np.std(nav_slice)
#     data_mean                = np.mean(data_slice)
#     data_std                 = np.std(data_slice)

#     vol = sitk.GetImageFromArray(vol)
#     vol_slice_atNavPos_std   = np.std(vol_slice_atNavPos)
#     vol_slice_atDataPos_mean = np.mean(vol_slice_atDataPos)
#     vol_slice_atDataPos_std  = np.std(vol_slice_atDataPos)

#     if abs(nav_mean) > 1.0:
#         logger.info('Normalization..... nav_mean = {}; nav_std = {}; data_mean = {}, data_std = {}, '.format(slice_pos, generatorName, len(ID_list), case_histogram))
