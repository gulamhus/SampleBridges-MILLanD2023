"""
@author: Gino Gulamhussene

@description: The script generates a nifti from the source dicom volume images with the right meta data in the header
"""

""" ================== 
script imports
=================== """
import os
import sys
import configparser
from probandMetaData import *
from scipy import ndimage as nd
import nibabel as nib
import numpy as np
from globalHelper import *



""" ================== 
program parameters
=================== """
probandFolders = ['subjet1', 'subject2'] #list of folder names containing subject data



print('===============================================')
print('dicomVolume_2_Nii.py')
print('===============================================')

""" ================== 
global variables
=================== """
studyFolder  = r'D:\med_Bilddaten\4D_Leber\Dicom_MR_Data'



""" ================== 
script
=================== """
### create a nifti with the correct header info
for p in probandFolders:
  volumeFolders = []
  volumeNames = []
  for i in range(3):
    folderName = os.path.join(studyFolder, p, 'MR_Data', 'StarVibe_' + str(i))
    if os.path.isdir(folderName):
      volumeFolders.append(folderName)
      volumeNames.append(p + '_Volume_' + str(i))

  for i in range(len(volumeFolders)):
    srcFolder = volumeFolders[i]
    outFolder = os.path.join(r'D:\med_Bilddaten\4D_Leber\DeepLearning\Data_for_DeepLearning', p)
    if not os.path.isdir(outFolder):
      os.mkdir(outFolder)
    os.system(r'D:\med_Bilddaten\4D_Leber\MRIcroGL_windows\MRIcroGL\Resources\dcm2niix.exe -z y -f ' + volumeNames[i] + r' -o ' + outFolder + r' ' + srcFolder)
    
