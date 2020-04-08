# region Imports
import numpy as np
import time
from pathlib import Path
import getpass
import json
import pandas as pd
import os
import urllib
import wget
import requests

from utils import *
from OurDataset import *
import torch
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy.matlib
# endregion


# auto create ensemble

root_folder = Path(os.getcwd()).parent

# set kaggle folder
os.environ["KAGGLE_CONFIG_DIR"] = str(root_folder / '..' )

api = KaggleApi()
api.authenticate()
fields, submissions = api.competition_submissions_cli(competition='recognizing-faces-in-the-wild',
                                    competition_opt=None,
                                    csv_display=True,
                                    quiet=True)

publicScoreIdx = fields.index('publicScore')
privateScoreIdx = fields.index('privateScore')
filenameIdx = fields.index('fileName')
urlIdx = fields.index('url')

# filter previous auto ensembles
submissions = list(filter(lambda s: not s[filenameIdx].startswith('auto_en_'), submissions))

publicScore = np.array([x[publicScoreIdx] for x in submissions], dtype=float)
privateScore = np.array([x[privateScoreIdx] for x in submissions], dtype=float)

method = 'top'  # 'top' or 'threshold'
score = 'public'  # 'public' or 'private'
func = 'l2'  # 'mean', 'wmean', 'l1', 'l2'

score_vec = publicScore if score == 'public' else privateScore
K = 10
threshold = 0.85

if method == 'top':
    topIdx = np.argsort(score_vec)[-K:]
elif method == 'threshold':
    topIdx = np.argwhere(score_vec > threshold).squeeze()

selected_submissions = [submissions[i] for i in topIdx]

submissions_paths = []
submissions_names = []
selected_scores = []

for si, s in enumerate(selected_submissions):
    # check if submission file exist
    filePath = root_folder / 'submissions_files' / s[filenameIdx]
    fileExists = os.path.exists(filePath)
    if not fileExists:
        continue
    submissions_paths.append(str(filePath))
    submissions_names.append(str(s[filenameIdx]))
    selected_scores.append(s[publicScoreIdx] if score == 'public' else s[privateScoreIdx])

selected_scores = np.array(selected_scores, dtype=float)
weights = selected_scores / np.sum(selected_scores)

ensemble_csvs = [pd.read_csv(s) for s in submissions_paths]
all_sigmoids = np.vstack([np.array(csv.is_related) for csv in ensemble_csvs])

if func == 'mean':
    res1 = np.mean(all_sigmoids, axis=0)
elif func == 'wmean':
    res1 = np.average(all_sigmoids, axis=0, weights=weights)
elif func == 'l1':
    weights = np.abs(selected_scores - 0.5)
    weights_mat = numpy.matlib.repmat(weights, all_sigmoids.shape[1], 1).T
    res1 = np.sum(np.multiply(all_sigmoids, weights_mat), axis=0) / np.sum(weights)
elif func == 'l2':
    weights = (selected_scores - 0.5)**2
    weights_mat = numpy.matlib.repmat(weights, all_sigmoids.shape[1], 1).T
    res1 = np.sum(np.multiply(all_sigmoids, weights_mat), axis=0) / np.sum(weights)

# get the sample submission file for loading pairs, and create the new submission file.
ensemble_name = 'auto_en_' + time.strftime('%d.%m.%H.%M.%S')
sample_submission_path = root_folder / 'data' / 'faces' / 'sample_submission.csv'
dst_submission_path = root_folder / 'submissions_files' / str(ensemble_name + '.csv')
copyfile(sample_submission_path, dst_submission_path)
df_submit = pd.read_csv(str(dst_submission_path))

df_submit.is_related = res1

# write submission
res1 = df_submit.to_csv(dst_submission_path, index=False)
with open(str(dst_submission_path).replace('.csv', '.txt'), "wt") as f:
    f.write(str(submissions_names) + '\n')
    f.write('method: ' + method + '\n')
    f.write('score: ' + score + '\n')
    f.write('func: ' + func + '\n')
    f.write('K: ' + str(K) + '\n')
    f.write('threshold: ' + str(threshold))

# submit file
os.system('kaggle competitions submit -c recognizing-faces-in-the-wild -f ' +
          str(dst_submission_path) + ' -m ' + ensemble_name + '.csv')
# show submissions

fields, submissions = api.competition_submissions_cli(competition='recognizing-faces-in-the-wild',
                                    competition_opt=None,
                                    csv_display=True,
                                    quiet=True)

while submissions[0][publicScoreIdx] == 'None':
    fields, submissions = api.competition_submissions_cli(competition='recognizing-faces-in-the-wild',
                                                          competition_opt=None,
                                                          csv_display=True,
                                                          quiet=True)
    print('Waiting for result...\n')
    time.sleep(5)

print(fields)
print(submissions[0])
