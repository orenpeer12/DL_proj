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

ensemble_name = 'en3'
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
    mean = [91.4953, 103.8827, 131.0912]
    std = [1, 1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        scale_tensor_255,
        rgb2bgr,
        transforms.Normalize(mean=mean,
                             std=std)
    ]),
    # load test-set
    testset = TestDataset(df=df_submit, root_dir=root_folder / 'data' / 'faces' / 'test', transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8)
    # load models-paths and loacate the best validation-accuracy-models
    models_names = [m.split('_')[0] for m in submissions]
    best_models_names = [get_best_model(root_folder / 'models' / m) for m in models_names]
    outputs = []
    for mn, bmn in zip(models_names, best_models_names):
        # load model
        net = torch.load(root_folder / 'models' / mn / bmn + '')
        net.load_state_dict(torch.load(root_folder / 'models' / model_time / model_name.replace('.pt', '_state.pt')))

        # pass testset through model:
        net.eval()
        net.to(device)
        for i, data in enumerate(test_loader, 0):
            row, img0, img1 = data
            row, img0, img1 = row.to(device), img0.to(device), img1.to(device)

            output = net(img0, img1)
            # predicted = torch.round(output.data).long().view(-1)
            predicted = output.data.view(-1)



# os.system('kaggle competitions submit -c recognizing-faces-in-the-wild -f ' + \
#                   str(dst_submission_path) + ' -m ' + ensemble_name + '.csv')
# # show submissions
# os.system('kaggle competitions submissions recognizing-faces-in-the-wild')
