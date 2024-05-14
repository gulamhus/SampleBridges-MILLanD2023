import csv
import glob
import os
from tqdm import tqdm

main_path     = '/home/gino/projects/4D_MRI_CNN/experiments/transfer_learning_experiments'
# csv_main_name = r'masked_losses_T[[]2-98[]]_smpl-20'
csv_main_name = r'masked_losses_C16-5_T[[]2-98[]]_smpl-20'

### get csv parts
reg = csv_main_name + '_prt-*.csv'
csv_names = sorted(glob.glob(os.path.join(main_path,'losses_tmp', reg)))

all_rows = []
header = None
for csv_name in tqdm(csv_names):
  with open(os.path.join(main_path, csv_name), 'r', newline='') as csvfile:    
    csv_reader = csv.reader(csvfile)
    for i, row in enumerate(csv_reader):
      if i == 0 and header == None: 
        header = row
      elif i > 0:   
        all_rows.append(row)
      

save_name = csv_main_name.replace('[[]', '[').replace('[]]', ']')
with open(os.path.join(main_path, save_name + '.csv'), 'w', newline='') as csvfile:    
  csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  csvwriter.writerow(header)
  for row in all_rows:
    csvwriter.writerow(row)
