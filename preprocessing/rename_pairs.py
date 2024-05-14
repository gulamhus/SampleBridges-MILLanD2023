import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import dicom


main_dir = r'D:\med_Bilddaten\4D_Leber'
files_dir = r'D:\med_Bilddaten\4D_Leber\\DeepLearning\Data_for_DeepLearning'
cases = ['subjet1', 'subject2'] #list of folder names containing subject data


# ===================================================
# go through files in fnames, 
# groupes the file names by series number
# ===================================================
def sortDataFilesBySeriesNumber(fnames):  
  result_fnames = [] ### will be a list of lists
  result_fnames.append([])
  
         
  parts = os.path.basename(fnames[0]).split('.')[0].split('_')
  setNumber_last = parts[5]
  for filename in fnames:

    ### meta data
    parts = os.path.basename(filename).split('.')[0].split('_')
    setNumber = parts[5]

    if setNumber == setNumber_last:
      result_fnames[-1].append(filename)
    else:
      result_fnames.append([]) 
      result_fnames[-1].append(filename)

    setNumber_last = setNumber
  return result_fnames

### get the r location of dicom file
def getRLocation(dicomFile):
  tag2 = getattr(dicomFile, 'ImagePositionPatient', None)
  tag3 = getattr(dicomFile, 'ImageOrientationPatient', None)
  xdir = tag3[:3]
  ydir = tag3[3:]
  zdir = np.cross(xdir, ydir)
  r_location = np.dot(tag2, zdir)
  return r_location

for case in tqdm(cases):
  print("processing: ", case)
  case = case.replace('\n', '')

  ### get navigator R location
  nav_dir = os.path.join(main_dir, 'Dicom_MR_Data', case, 'MR_Data', 'Nav_Pur_1')
  regEx = '*.IMA'
  nav_sample = sorted(glob.glob(os.path.join(nav_dir, regEx)))[0]
  dfile = dicom.read_file(nav_sample)    
  nav_r = int(getRLocation(dfile))

  regEx = '*_T_*.nii.gz'
  samples = sorted(glob.glob(os.path.join(files_dir, case, regEx)))
  samples = sortDataFilesBySeriesNumber(samples)

  for i in range(len(samples)): 
    r = int(samples[i][0].split('.')[0].split('_R_')[-1])

    if r != nav_r: ### first is data slice
      samples[i] = samples[i][1:-1] ### first data file removed because it has no leading navigator slice also last file is nav has no following data file and is removed

  os.chdir(os.path.join(files_dir, case))
  for sequence in samples:
    pairNr = 1
    for i, filename in enumerate(sequence):            
      parts = os.path.basename(filename).split('.')[0].split('_')
      t_n = parts[3]
      r_n = parts[8]
      s_n = parts[5]

      padding = ''
      if pairNr < 1000:
        padding = '0'
      if pairNr < 100:
        padding = '00'
      if pairNr < 10:
        padding = '000'

      #new_filename = case + '_T_' + t_n + '_S_' + s_n
      new_filename = case + '_S_' + s_n
      if i%2 == 0:
        new_filename = new_filename + '_N_' + padding + str(pairNr) + '.nii.gz'
      else:
        new_filename = new_filename + '_D_' + padding + str(pairNr) + '.nii.gz'
        pairNr += 1
      new_filename = os.path.join(files_dir, case, new_filename)
      os.rename(filename, new_filename)
      
