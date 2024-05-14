import os
import glob
from moviepy.editor import *

# main_dir = r'/data/gino/4D_MRI_CNN/output/4D_prediction/Test_subjects'
main_dir = r'D:\med_Bilddaten\4D_Leber\DeepLearning\output\4D_ortho_view_videos\test_subjects'

# method_dir = r'C16-5_T98'
# method_dir = r'C16-5_T5'
# method_dir = r'T50'
# method_dir = r'T98'
method_dir = r'E10-C16-5_T5'


source_dir = os.path.join(main_dir, method_dir)
subject_dirs = os.listdir(source_dir)
for sub_dir in subject_dirs:
  if os.path.isdir(os.path.join(source_dir, sub_dir, 'rotating_MIP')):
    files = sorted(glob.glob(os.path.join(source_dir, sub_dir, '*.png')))

  
    clip = ImageSequenceClip(files, fps = 6) 
    clip.write_videofile(os.path.join(source_dir, "{}_ref_seq_{}.mp4".format(sub_dir, method_dir)), fps = 30)
    # clip.write_gif(os.path.join(r'/home/gino/projects/data/4D_prediction/Test_subjects/ortho_view', "2018-05-07_JPaf_T98.gif"), fps = 30)