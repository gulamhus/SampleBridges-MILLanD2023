import numpy as np
import matplotlib 
import csv
sys.path.append('/home/gino/projects/4D_MRI_CNN')
import preprocessing.preprocess as prep
import preprocessing.utils as utils

losses_csv   = '/home/gino/projects/4D_MRI_CNN/experiments/transfer_learning_experiments/losses_C16-5_T[2-98]_medium.csv'

### get model IDs from csv
comp_times = []
with open(losses_csv, 'r', newline='') as csvfile:    
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        comp_times.append(row['computation_time'])         

plt.plot(range(0,len(comp_times), comp_times))
plt.show()