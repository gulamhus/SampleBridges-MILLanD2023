
import glob
import os
import json
import preprocess as prep


data_path = '/cache/gino/Data_for_DeepLearning/test'

  

cropped_vol_origins = {}
physicalPointOrigins = {}
name = os.path.join('/cache/gino/Data_for_DeepLearning/', 'vol_origins.json')
name_phy = os.path.join('/cache/gino/Data_for_DeepLearning/', 'vol_origins_phy.json')

with open(name, 'r') as f:
    cropped_vol_origins = json.load(f)
    
with open(name_phy, 'r') as f:
    physicalPointOrigins = json.load(f)
    
cases = os.listdir(data_path) 
for case in cases:
  case = case.replace('\n', '')
  vol_filename   = sorted(glob.glob(os.path.join(data_path, case, '*_Volume_1.nii.gz')))[0]
  data_filename  = sorted(glob.glob(os.path.join(data_path, case, '*_D_0001.nii.gz')))[0]
  # nav_filename   = sorted(glob.glob(os.path.join(data_path, case, '*_N_0001.nii.gz')))[0]
  vol            = prep.load_and_preprocess_vol(vol_filename,    newSpacing=[4.0, 4.0, 4.0], newSize=[160, 160, 80])
  data           = prep.load_and_preprocess_slice(data_filename, newSpacing=[4.0, 4.0, 4.0])
  # nav            = prep.load_and_preprocess_slice(nav_filename,  newSpacing=[4.0, 4.0, 4.0])


  physicalPointOrigins[case] = data.GetOrigin()
  c = vol.TransformPhysicalPointToIndex(data.GetOrigin())
  cropped_vol_origins[case] = c

with open(name, 'w') as f:
  json.dump(cropped_vol_origins, f, indent=4)
  
with open(name_phy, 'w') as f:
  json.dump(physicalPointOrigins, f, indent=4)