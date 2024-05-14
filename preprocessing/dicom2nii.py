"""
@author: Gino Gulamhussene

@description: The script generates niftis from the source dicom series 
"""

""" ================== 
script imports
=================== """
import os
import sys
import numpy as np
import glob
import dicom
from tqdm import tqdm



""" ================== 
program parameters
=================== """


print('===============================================')
print('dicom_2_Nii.py')
print('===============================================')

""" ================== 
global variables
=================== """
main_dir = r'D:\med_Bilddaten\4D_Leber\Dicom_MR_Data'
out_main_dir = r'D:\med_Bilddaten\4D_Leber\DeepLearning\Data_for_DeepLearning'
dcm2niix_dir = r'D:\med_Bilddaten\4D_Leber\MRIcroGL_windows\MRIcroGL\Resources'
cases = ['subjet1', 'subject2'] #list of folder names containing subject data

def ms_from_DicomTag_AcquisitionTime(t_dcm):
    [hms_s, ms_s] = t_dcm.split(".")
    h = float(hms_s[0:2])
    m = float(hms_s[2:4])
    s = float(hms_s[4:6])
    ms = float(ms_s[0:3])
    ms = ms + 1000*s + 60*1000*m + 60*60*1000*h
    return ms

### get the r location of dicom file
def getRLocation(dicomFile):
  tag2 = getattr(dicomFile, 'ImagePositionPatient', None)
  tag3 = getattr(dicomFile, 'ImageOrientationPatient', None)
  xdir = tag3[:3]
  ydir = tag3[3:]
  zdir = np.cross(xdir, ydir)
  r_location = np.dot(tag2, zdir)
  return r_location


# ===================================================
# go through dicom files in dicom_fnames, 
# groupes the file names by series number
# ===================================================
def sortDataFilesBySeriesNumber(dicom_fnames):  
  fileCount = len(dicom_fnames)
  result_dicom_fNames = [] ### will be a list of lists
  result_dicom_fNames.append([])
  ### init progress bar und start stop watch

  setNumber_last = dicom_fnames[0].split('.')[3]
  for i in range(fileCount):

    ### meta data
    setNumber = dicom_fnames[i].split('.')[3]

    if setNumber == setNumber_last:
      result_dicom_fNames[-1].append(dicom_fnames[i])
    else:
      result_dicom_fNames.append([]) 
      result_dicom_fNames[-1].append(dicom_fnames[i])

    setNumber_last = setNumber
  return result_dicom_fNames



""" ================== 
script
=================== """
for case in cases:
  print(case)
  # src_dir = os.path.join(main_dir, case, 'MR_Data', 'Nav_+_Data')
  # out_dir = os.path.join(out_main_dir, case)
  
  ### for nav pur (ref)
  src_dir = os.path.join(main_dir, case, 'MR_Data', 'Nav_Pur_1')  
  out_dir = os.path.join(out_main_dir, case + '_refSeq')
  
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)

  regEx = '*.IMA'
  proband_samples = sorted(glob.glob(os.path.join(src_dir, regEx)))
  ### sort images by acquisition time
  r_s = []
  t_s = []
  f_s = []
  for sample in tqdm(proband_samples):
    dfile = dicom.read_file(sample)    
    r = getRLocation(dfile)
    r_s.append(int(r))

    t_dcm = dfile.AcquisitionTime  
    t = ms_from_DicomTag_AcquisitionTime(t_dcm)
    t_s.append(int(t))

    filename = sample.split('\\')[-1]
    f_s.append(filename)

  t_s = np.array(t_s)
  indices = np.argsort(t_s)

  os.chdir(src_dir)
  refNr = 1
  for i in tqdm(indices):
    filename = f_s[i]
    r = r_s[i]
    t = t_s[i]    

    padding = ''
    if refNr < 1000:
      padding = '0'
    if refNr < 100:
      padding = '00'
    if refNr < 10:
      padding = '000'
      
    exe = os.path.join(dcm2niix_dir, 'dcm2niix.exe')
    # command = exe + ' -z y -s y -m n -v 0 -f ' + case + r'_T_' + str(t) + r'_S_%4s_%4r_R_' + str(r) + r' -o ' + out_dir + r' ' + filename
    
    ### for nav pur
    command = exe + ' -z y -s y -m n -v 0 -f ' + case + r'_S_%4s' + '_NR_' +  padding + str(refNr) + r' -o ' + out_dir + r' ' + filename
    os.system(command)
    
    refNr += 1
     


