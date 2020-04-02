# region Imports
import numpy as np
import time
from pathlib import Path
import getpass
import json
import pandas as pd
import os

from utils import *
from OurDataset import *
import torch

# endregion

##
# Parameters:
##
random.seed(42)
NUM_WORKERS = 4
GPU_ID = 0 if 'nir' in os.getcwd() else 1
device = torch.device('cuda: ' + str(GPU_ID) if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

ensemble_name = 'en4'
root_folder = Path(os.getcwd())
submission_folder = root_folder / 'submissions_files' / ensemble_name

# get the sample submission file for loading pairs, and create the new submission file.
sample_submission_path = root_folder / 'data' / 'faces' / 'sample_submission.csv'
dst_submission_path = root_folder / 'submissions_files' / ensemble_name / str(ensemble_name + '.csv')
copyfile(sample_submission_path, dst_submission_path)
df_submit = pd.read_csv(str(dst_submission_path))

submissions = [s for s in os.listdir(submission_folder) if s.endswith('.csv') and s != ensemble_name + '.csv']
submissions_paths = [root_folder / 'submissions_files' / ensemble_name / s for s in submissions]
print('Creating ensemble submission out of {} submissions!'.format(len(submissions)))

ensemble_csvs = [pd.read_csv(s) for s in submissions_paths]
all_sigmoids = np.vstack([np.array(csv.is_related) for csv in ensemble_csvs])

res1 = np.mean(all_sigmoids, axis=0)
# a = np.abs(res -.5)
# b = []
# for i in range(a.shape[1]):
#     best_idx_in_line = np.argmax(a[:, i])
#     b.append(res[best_idx_in_line][i])
# res = np.array(b)
df_submit.is_related = res1

# write submission
res1 = df_submit.to_csv(dst_submission_path, index=False)
# submit file
os.environ["KAGGLE_CONFIG_DIR"] = str(root_folder / '..')
os.system('kaggle competitions submit -c recognizing-faces-in-the-wild -f ' +
          str(dst_submission_path) + ' -m ' + ensemble_name + '.csv')
# show submissions
os.system('kaggle competitions submissions recognizing-faces-in-the-wild')
