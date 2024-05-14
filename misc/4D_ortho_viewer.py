import SimpleITK as itk
import numpy as np
from scipy import ndimage, misc
import math
import sys
# from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from volume_renderer import volume_renderer
sys.path.append('/home/gino/projects/4D_MRI_CNN')
# sys.path.append(r'D:\med_Bilddaten\4D_Leber\DeepLearning\4D_MRI_CNN')

import preprocessing.utils as utils

"""
usage:
save_planes(
    volumes_path='volumes',
    x=30, y=50, z=50, flip_mode='control'
)
"""

flip_modes = {
    'control':
        [
            [   ### sag
                lambda x: np.swapaxes(x, axis1=0, axis2=2),
                # lambda x: np.flip(x, axis=2)
                # lambda x: np.flip(x, axis=1),
                # lambda x: np.swapaxes(x, 0, 1),
                # lambda x: np.flip(x, axis=0)
            ],
            [   ### ax
                lambda x: np.swapaxes(x, axis1=0, axis2=1),
                lambda x: np.flip(x, axis=2)
            ],
            [   ### cor
                lambda x: np.flip(x, axis=0),
                lambda x: np.flip(x, axis=2)
            ]
        ],
    'test':
        [
            [   ### sag
                lambda x: np.swapaxes(x, axis1=1, axis2=2),
                lambda x: np.swapaxes(x, axis1=0, axis2=1),
                lambda x: np.flip(x, axis=1), 
                lambda x: np.flip(x, axis=2)
            ],
            [   ### cor
                lambda x: np.swapaxes(x, axis1=0, axis2=1),
                lambda x: np.flip(x, axis=1), 
                lambda x: np.flip(x, axis=2)
            ],
            [   ### ax
                lambda x: np.flip(x, axis=1),   
                lambda x: np.flip(x, axis=2)
                
            ]
        ]
}


# adapted for 2D images from https://stackoverflow.com/questions/65496246/
# simpleitk-coronal-sagittal-views-problems-with-size
def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0)):
    """
    Resample itk_image to new out_spacing
    :param itk_image: the input image
    :param out_spacing: the desired spacing
    :return: the resampled image
    """
    # get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    # calculate new size
    out_size = [
        int(np.round(osz * (osp / ousp)))
        for osp, osz, ousp in zip(original_spacing, original_size, out_spacing)
    ]
    # instantiate resample filter with properties and execute it
    resample = itk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(itk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(itk.sitkNearestNeighbor)
    return resample.Execute(itk_image)


def rescale(volume):
    # Cut off values which are too large
    # clamp_filter = itk.ClampImageFilter()
    # clamp_filter.SetLowerBound(-1)
    # clamp_filter.SetUpperBound(255)
    # clamp_filter.SetOutputPixelType(itk.sitkFloat32)
    # volume = clamp_filter.Execute(volume)
    # rescale between 0 and 255
    rescale_filter = itk.RescaleIntensityImageFilter()
    rescale_filter.SetOutputMaximum(255)
    rescale_filter.SetOutputMinimum(0)
    return rescale_filter.Execute(volume)


def rotate_image(image, flip_functions):
    for flip_function in flip_functions:
        image = flip_function(image)
    return image


def get_planes(volume, x, y, z, MIP_Thickness=[1,1,1]):
    pos_x = x 
    pos_y = y
    pos_z = z
    
    if pos_x is None: pos_x = 0
    if pos_y is None: pos_y = 0
    if pos_z is None: pos_z = 0
    
    start_x = pos_x - math.ceil(MIP_Thickness[0]/2.0) 
    start_y = pos_y - math.ceil(MIP_Thickness[1]/2.0)
    start_z = pos_z - math.ceil(MIP_Thickness[2]/2.0)
    
    end_x = pos_x + math.floor(MIP_Thickness[0]/2.0) 
    end_y = pos_y + math.floor(MIP_Thickness[1]/2.0)
    end_z = pos_z + math.floor(MIP_Thickness[2]/2.0)
    
    sagittal  = volume[start_x:end_x, :, :]
    coronal   = volume[:, start_y:end_y, :]
    transvers = volume[:, :, start_z:end_z]
    return sagittal, coronal, transvers


def slice_planes_from_vol(volume, x=None, y=None, z=None, MIP_Thickness=[2,2,2], flip_mode='control'):
    result = {}
    planes = get_planes(volume, x, y, z, MIP_Thickness=MIP_Thickness)
    directions = {'control':['sag', 'ax', 'cor'], 'test':['sag', 'cor', 'ax']}
    for direction, plane, flip_functions, pos in zip(directions[flip_mode], planes, flip_modes[flip_mode], [x, y, z]):
        if pos is None:
            continue
        plane = resample_image(plane)
        plane = rescale(plane)
        plane = itk.GetArrayFromImage(plane)
        plane = rotate_image(plane, flip_functions)    
        plane = np.max(plane, axis=0)
        result[direction] = plane
    return result

def get_ortho_view(planes):
    shape_x  = max(planes['ax'].shape[0], planes['sag'].shape[0], planes['cor'].shape[0]) * 2
    shape_y  = max(planes['ax'].shape[1], planes['sag'].shape[1], planes['cor'].shape[1]) * 2
    result   = np.zeros((shape_x, shape_y))
    
    for img, pos in zip([planes['ax'], planes['sag'], planes['cor']], [(0,0), (1,0), (1,1)]):
        result = put_img_in_quadrant(img, result, pos)
    return result
        
def put_img_in_quadrant(src_img, dst_img, quadrant=(0,0)):
    shape_x = dst_img.shape[0]
    shape_y = dst_img.shape[1]
    m_x               = int(((shape_x / 2) - src_img.shape[0]) / 2)  ### margin_x
    m_y               = int(((shape_y / 2) - src_img.shape[1]) / 2)  ### margin_y
    quadrand_orig_x   = int(quadrant[1] * shape_x / 2)                ### quadrand_origin_x
    quadrand_orig_y   = int(quadrant[0] * shape_y / 2)                ### quadrand_origin_y
    dst_img[quadrand_orig_x+m_x:quadrand_orig_x+m_x+src_img.shape[0], quadrand_orig_y+m_y:quadrand_orig_y+m_y+src_img.shape[1]] = src_img
    return dst_img


### CNN predictions
# ### test subjects
# path_in         = r'/data/gino/4D_MRI_CNN/output/4D_prediction/Test_subjects/T5/pred_ref_sequence'
# path_in         = r'/data/gino/4D_MRI_CNN/output/4D_prediction/Test_subjects/T98/pred_ref_sequence'
# path_in         = r'/data/gino/4D_MRI_CNN/output/4D_prediction/Test_subjects/C16-5_T5/pred_ref_sequence'
path_in         = r'/data/gino/4D_MRI_CNN/output/4D_prediction/Test_subjects/C16-5_T98/pred_ref_sequence'
path_out        = path_in # os.path.join(path_in, 'ortho_view')
sub_folders     = ['subjet1', 'subject2'] #list of folder names containing subject data
pos             = {'subjet1':[153,50,60], 'subject2':[150,59,50]}



i = 0
for sf in sub_folders:
    utils.makeDirectory(os.path.join(path_out, sf))
    volume_paths    =  sorted(glob.glob(os.path.join(path_in, sf, '*.nii.gz')))
    for p in tqdm(volume_paths):        
        
        ### load volume
        volume = itk.ReadImage(p)     
        
        ### create ortho view
        [x,y,z]    = pos[sf] 
        # slices     = slice_planes_from_vol(volume, x=x, y=y, z=z, MIP_Thickness=[14,30,30], flip_mode='control')
        slices     = slice_planes_from_vol(volume, x=x, y=y, z=z, MIP_Thickness=[30,30,30], flip_mode='test')
        ortho_view = get_ortho_view(slices)
        
        ### save img
        filename = os.path.split(p)[1].split('.')[0] + '_ortho_view.png'
        plt.imsave(os.path.join(path_out,sf,filename), ortho_view, cmap="gray")
        i += 1
        continue

