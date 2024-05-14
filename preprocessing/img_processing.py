import SimpleITK as sitk
import numpy as np
import math


def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")
    
def registration_2D(fixed_img_2D, moving_img_2D, grid_size=[10,10]):
  fixed  = fixed_img_2D
  moving = moving_img_2D
  
  ### init the trarsformation mesh
  transformDomainMeshSize = grid_size
  tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

  ### make registration itk object
  R = sitk.ImageRegistrationMethod()
  R.SetMetricAsMeanSquares()
  R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
  R.SetInitialTransform(tx, True)
  R.SetInterpolator(sitk.sitkLinear)
  # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

  return R.Execute(fixed, moving)

def transform_to_displacementField(ref_img, transformation):
  ## convert the transform to a displacement field
  return sitk.TransformToDisplacementField(transformation, 
                                  sitk.sitkVectorFloat64,
                                  ref_img.GetSize(),
                                  ref_img.GetOrigin(),
                                  ref_img.GetSpacing(),
                                  ref_img.GetDirection())

def deform_img(fixed, moving, transformation):
  ### move moving_img
  resampler = sitk.ResampleImageFilter()
  resampler.SetReferenceImage(fixed)
  resampler.SetInterpolator(sitk.sitkLinear)
  resampler.SetDefaultPixelValue(0)
  resampler.SetTransform(transformation)
  return resampler.Execute(moving)
      

