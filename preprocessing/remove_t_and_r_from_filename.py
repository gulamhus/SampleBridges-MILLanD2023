import os
import glob
from tqdm import tqdm
import re

files_dir = r'/cache/anneke/4D_MRI_data/data_init/train'
files_dir = r'D:\med_Bilddaten\4D_Leber\DeepLearning\Data_for_DeepLearning'

cases = ['subjet1', 'subject2'] #list of folder names containing subject data

for case in cases:
  print("processing: ", case)
  case = case.replace('\n', '')
  
  regEx = '*.nii.gz'
  filenames = sorted(glob.glob(os.path.join(files_dir, case, regEx)))
  for f in filenames:
    
    #new_filename = re.sub(r"_R_-?\d+\.", ".", f)
    #new_filename = re.sub(r"_R_R", "", f)
    new_filename = re.sub(r"_T_\d+_", "_", f)
    #print(new_filename)
    os.rename(f, new_filename)
