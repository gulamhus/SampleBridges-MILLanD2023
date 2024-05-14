import glob
import os
import math
import csv
import sys
sys.path.append('/home/gino/projects/4D_MRI_CNN')
import preprocessing.utils as utils

with open('/home/gino/projects/4D_MRI_CNN/doc/data_summary.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['subject', 'slice_count', 'samples_per_pos', 'total_sample_count', 'acq_time in minutes'])
    # Set the directory you want to start from
    
    subject_count = 0
    rootDirs = ['/cache/gino/Data_for_DeepLearning/train', '/cache/gino/Data_for_DeepLearning/val', '/cache/gino/Data_for_DeepLearning/test']
    for rootDir in rootDirs:
      for dirName, subdirList, fileList in os.walk(rootDir):
          subject_count += 1
          fnames = sorted(glob.glob(os.path.join(dirName, '*_N_*.nii.gz')))
          if len(fnames) == 0:
            continue
          
          fnames_dict         = utils.groupFileNamesBySeriesNumber(fnames)
          total_sample_count  = 0
          for samples in fnames_dict:
            total_sample_count += len(samples)
            
          slice_count  = len(fnames_dict)
          samples_per_pos = math.floor(total_sample_count/slice_count)
          writer.writerow([os.path.split(dirName)[1], slice_count, samples_per_pos, total_sample_count, slice_count])
          