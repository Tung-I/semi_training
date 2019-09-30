import csv
import glob
import numpy as np
from pathlib import Path


data_dir = Path('/home/tony/Documents/cell_data/DSB2018plus/stage_1_train')
# data_dir = Path('/home/tony/Documents/cell_data/HE')
paths = [path for path in data_dir.iterdir() if path.is_dir()]
num_paths = len(paths)

random_seed = 0
np.random.seed(int(random_seed))
np.random.shuffle(paths)

with open('DSB_train_val_split.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for idx, p in enumerate(paths):
        if idx < int(num_paths * 0.9):
            writer.writerow([str(p), 'train'])
        else:
            writer.writerow([str(p), 'validation'])

