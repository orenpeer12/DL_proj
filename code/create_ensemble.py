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
GPU_ID = 0
# device = torch.device('cuda: ' + str(GPU_ID) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

ensemble_name = 'en2'
root_folder = Path(os.getcwd())
submission_folder = root_folder / 'submissions_files' / ensemble_name
BinaryEnsemble = False

# get the sample submission file for loading pairs, and create the new submission file.
sample_submission_path = root_folder / 'data' / 'faces' / 'sample_submission.csv'
dst_submission_path = root_folder / 'submissions_files' / ensemble_name / str(ensemble_name + '.csv')
copyfile(sample_submission_path, dst_submission_path)
df_submit = pd.read_csv(str(dst_submission_path))

submissions = [s for s in os.listdir(submission_folder) if s.endswith('.csv') and s != ensemble_name + '.csv']
submissions_paths = [root_folder / 'submissions_files' / ensemble_name / s for s in submissions]
print('Creating ensemble submission out of {} submissions!'.format(len(submissions)))

if BinaryEnsemble:
    ensemble_csvs = [pd.read_csv(s) for s in submissions_paths]
    # for (lines) in zip(ensemble_csvs):
    #     lines +=1
    zero_one_count = np.zeros((ensemble_csvs[0].count(axis=0)[0],2))

    for csv_file in ensemble_csvs:
        for line_num, is_related in enumerate(csv_file.is_related):
            zero_one_count[line_num][is_related] += 1

    res = np.argmax(zero_one_count, axis=1)
    ties = np.where(zero_one_count[:, 0] == zero_one_count[:, 1])[0]
    res[ties] = np.random.randint(0, 1, len(ties))

    df_submit.is_related = res

    # write submission
    df_submit['is_related'].value_counts()
    res = df_submit.to_csv(dst_submission_path, index=False)

else:
    models_paths = [root_folder / 'models' / m.split('_')[0] / m.replace('.csv', '.pt') for m in submissions]
    hyper_parameters_paths = [root_folder / 'models' / m.split('_')[0] / 'Hyper_Params.json' for m in submissions]
    hyper_parameters = []
    for hp in hyper_parameters_paths:
        with open(hp) as json_file:
            hyper_parameters.append(json.load(json_file))
    for model_path, hp in zip(models_paths, hyper_parameters):
        classifier_str = hp['classifier']
        feature_extractor = hp['feature_extractor']




# os.system('kaggle competitions submit -c recognizing-faces-in-the-wild -f ' + \
#                   str(dst_submission_path) + ' -m ' + ensemble_name + '.csv')
# # show submissions
# os.system('kaggle competitions submissions recognizing-faces-in-the-wild')
