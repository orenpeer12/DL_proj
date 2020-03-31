from utils import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import getpass
import sys
import PySimpleGUI as sg

root_folder = Path(os.getcwd()).parent

files = os.listdir(root_folder / 'curves')

sg.theme('DarkAmber')  # Add a touch of color
# All the stuff inside your window.
layout = [[sg.Listbox(files, size=(30, 30))],
          [sg.Button('Ok'), sg.Button('Cancel')]]

# Create the Window
window = sg.Window('Select run to plot', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):  # if user closes window or clicks cancel
        exit()
    elif event is 'Ok' and len(values[0]) > 0:
        print('plotting', values[0][0])
        break

window.close()

run_id = values[0][0]
# run_id = 1585487497

# if getpass.getuser() == 'nirgreshler':
#     root_folder = Path('E:\\DL_Course\\FacesInTheWild') if sys.platform.startswith('win') \
#         else Path('/home/oren/nir/DL_proj')
# else:
#     root_folder = Path('C:\\Users\\Oren Peer\\Documents\\technion\\OneDrive - Technion\\Master\\DL_proj') if \
#         sys.platform.startswith('win') \
#         else Path('/home/oren/PycharmProjects/DL_proj')

history_file_path = root_folder / 'curves' / run_id

show_plot(history_file_path)

# root_path = Path('/home/oren/PycharmProjects/DL_proj')
# model_name = "1585131238_e56_vl0.6654_va63.18.pt"
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])])
# create_submission(root_folder=root_path, model_name=model_name, transform=transform)