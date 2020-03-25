from utils import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import getpass
import sys

run_id = 1585128107

if getpass.getuser() == 'nirgreshler':
    root_folder = Path('E:\\DL_Course\\FacesInTheWild') if sys.platform.startswith('win') \
        else Path('/home/oren/nir/DL_proj')
else:
    root_folder = Path('C:\\Users\\Oren Peer\\Documents\\technion\\OneDrive - Technion\\Master\\DL_proj') if \
        sys.platform.startswith('win') \
        else Path('/home/oren/PycharmProjects/DL_proj')

history_file_path = root_folder / 'code' / 'curves' / (str(run_id) + '.npy')

show_plot(history_file_path)