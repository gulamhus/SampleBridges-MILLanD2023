import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import dicom


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


files_dir = 'D:\\med_Bilddaten\\4D_Leber'
cases = ['subjet1', 'subject2'] #list of folder names containing subject data

for case in cases:
  regEx = '*.IMA'
  proband_samples = sorted(glob.glob(os.path.join(files_dir, case, 'MR_Data', 'Nav_+_Data', regEx)))
  t_last = 0
  for sample in proband_samples:
    dfile = dicom.read_file(sample)    
    r = getRLocation(dfile)
    t_dcm = dfile.AcquisitionTime  
    t = ms_from_DicomTag_AcquisitionTime(t_dcm)
    dt =  t - t_last
    t_last = t

    filename = sample.split('\\')[-1]
    series_Nr = filename.split('.')[3]
    order_Nr = filename.split('.')[4]
    print('{} ########### S={}, O={}, r={}, t={}, dt={}'.format(filename, series_Nr, order_Nr, r, t, dt))
