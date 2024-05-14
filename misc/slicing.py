import SimpleITK as itk
import numpy as np
from PIL import Image
import pathlib


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
            [
                lambda x: np.flip(x, axis=1),
                lambda x: np.swapaxes(x, 0, 1),
                lambda x: np.flip(x, axis=0)
            ],
            [
                lambda x: np.flip(x, axis=1)
            ],
            [
                lambda x: np.flip(x, axis=1)
            ]
        ],
    'test':
        [
            [
                lambda x: np.flip(x, axis=1),
                lambda x: np.flip(x, axis=0)
            ],
            [
                lambda x: np.flip(x, axis=1),
                lambda x: np.flip(x, axis=0)
            ],
            [
                lambda x: np.flip(x, axis=1),
                lambda x: np.flip(x, axis=0)
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
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
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
    image = itk.GetArrayFromImage(image)
    for flip_function in flip_functions:
        image = flip_function(image)
    return image


def get_planes(volume, x, y, z):
    pos_x = x 
    pos_y = y
    pos_z = z
    
    if pos_x is None: pos_x = 0
    if pos_y is None: pos_y = 0
    if pos_z is None: pos_z = 0
    
    pos_x -= 1 
    pos_y -= 1
    pos_z -= 1
    
    sagittal  = volume[pos_x, :, :]
    coronal   = volume[:, pos_y, :]
    transvers = volume[:, :, pos_z]
    return sagittal, coronal, transvers


def save_planes(volumes_path, x=None, y=None, z=None, flip_mode='control', save_path=None, suffix=None):
    volumes_path = pathlib.Path(volumes_path)
    volume_paths = [
        volume_path for volume_path in volumes_path.iterdir()
        if set(volume_path.suffixes) == set(['.nii', '.gz'])
    ]
    for volume_path in volume_paths:
        volume = itk.ReadImage(str(volume_path))
        planes = get_planes(volume, x, y, z)
        directions = {'control':['sag', 'ax', 'cor'], 'test':['sag', 'cor', 'ax']}
        for direction, plane, flip_functions, pos in zip(directions[flip_mode], planes, flip_modes[flip_mode], [x, y, z]):
            if pos is None:
                continue
            plane     = resample_image(plane)
            plane     = rescale(plane)
            plane_np  = rotate_image(plane, flip_functions)
            save_name = volume_path.name.split('.')[0]
            if save_path is None:
                filename = (
                    f"{volume_path.parent}/{save_name}_{direction}{pos}.png"
                )
            else:
                filename = f"{save_path}/{save_name}_{direction}{pos}.png"
            img_pil = Image.fromarray(plane_np.astype(np.uint8))
            img_pil.save(filename)


volume_path = r''
# save_planes(volumes_path=volume_path, x=32, y=76, z=80, flip_mode='control')
save_planes(volumes_path=volume_path, x=38, flip_mode='control')
save_planes(volumes_path=volume_path, x=28, flip_mode='control')
save_planes(volumes_path=volume_path, z=80, flip_mode='control')


volume_path = r''
# save_planes(volumes_path=volume_path, x=137, y=51, z=74, flip_mode='test')
save_planes(volumes_path=volume_path, x=150, flip_mode='test')
save_planes(volumes_path=volume_path, x=128, flip_mode='test')
save_planes(volumes_path=volume_path, y=50, flip_mode='test')

