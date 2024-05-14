import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np


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

train_files_dir = 'some_path/4D_MRI_data/data_init/train'
train_files_dir = 'D:\\med_Bilddaten\\4D_Leber\\DeepLearning\\Data_for_DeepLearning'
cases = ['subjet1', 'subject2'] #list of folder names containing subject data

print(cases)
id_list = []
for case in cases:
  case = case.replace('\n', '')
  regEx = '*_[DN]_*.nii.gz'
  proband_samples = sorted(glob.glob(os.path.join(train_files_dir, case, regEx)))

  t_last = int(os.path.basename(proband_samples[0]).split('.')[0].split('_')[3])
  dt_s = []
  for ID in tqdm(proband_samples[1:]):
    filename = ID

    # f = sitk.ReadImage(filename)    
    # tinfo = f.GetMetaData('ITK_FileNotes')
    # t = ms_from_DicomTag_AcquisitionTime(tinfo.split(';')[1].split('=')[-1])
    t = int(os.path.basename(filename).split('.')[0].split('_')[3])
    dt = t - t_last
    t_last = t
    #print('{}, dt = {}'.format(filename, dt))
    
    dt_s.append(dt)
    #print('{} dt = {}'.format(nav_filename, diff))
  dt_s = np.array(dt_s)

  #vals, counts = np.unique(dt_s, return_counts=True)
  print(case, 'normal: {}, big: {}'.format(np.count_nonzero(dt_s < 200), np.count_nonzero(dt_s > 200)))
