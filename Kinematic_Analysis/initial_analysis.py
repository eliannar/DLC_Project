import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mat4py import loadmat

#DVIR_PATH = r'C:\Users\dvir.bens\Documents\DLC\dataAnaly\analysis_lib\3D_New\11-03-21\nana-try2-trail1_.avi_DLC_Dvir_3D.csv'   #path to the 3D csv file from DLC analysis
ELIANNA_DATA_PATH = r'Z:\elianna.rosenschein\n270122\DLC_Analysis\nana-trial1_DLC_3D_SIDE.csv'
ELIANNA_ED_PATH = r'Z:\elianna.rosenschein\n270122\EDfiles\n6901ee.1.mat'
ELIANNA_INFO_FILE = r'Z:\elianna.rosenschein\n270122\Info\n270122_param.mat'
BODY_PART = 'finger'  #body part that we want to analyze
FRAME_RATE = 120      #number of frame rate belonging record videos software, it help us know the time difference between 2 coordinates data into excel.
THRESHOLD = 0.025
MOVEMENT_SENSITIVITY = 20

# dictionary that links the body parts from the analysis to the colomns names
body_cols_side = {'finger': ['DLC_3D_SIDE', 'DLC_3D_SIDE.1', 'DLC_3D_SIDE.2'],
                      'wrist': ['DLC_3D_SIDE.3', 'DLC_3D_SIDE.4', 'DLC_3D_SIDE.5'],
                      'elbow': ['DLC_3D_SIDE.6', 'DLC_3D_SIDE.7', 'DLC_3D_SIDE.8']}                                     #dictionary that linking the body parts from the analysis to the colomns names}
body_cols_top = {'finger': ['DLC_3D_TOP', 'DLC_3D_TOP.1', 'DLC_3D_TOP.2'],
                 'wrist': ['DLC_3D_TOP.3', 'DLC_3D_TOP.4', 'DLC_3D_TOP.5'],
                 'elbow': ['DLC_3D_TOP.6', 'DLC_3D_TOP.7', 'DLC_3D_TOP.8']}


prep_angle = {'SIDE': body_cols_side, 'TOP': body_cols_top}


#PREPROCESSING
def preprocessing(data_path, angle):
    info_file = loadmat(ELIANNA_INFO_FILE)

    hfs_subsess, burst_subsess = load_info(ELIANNA_INFO_FILE)
    ed_file = loadmat(ELIANNA_ED_PATH)
    TrialTimes = ed_file['TrialTimes']

    partial_data = process_data(ELIANNA_DATA_PATH, angle)
    indices = loadmat(r'Z:\elianna.rosenschein\alignment_indices_n270122.mat')
    videoInfo = loadmat(r'Z:\elianna.rosenschein\vidInfo_n270122.mat')
    plot_velocity(partial_data)


def load_info(info_file_path):
    info_file = loadmat(info_file_path)
    hfs_subsess = info_file['SESSparam']['fileConfig']['HFS']
    burst_subsess = info_file['SESSparam']['fileConfig']['BURST']
    return hfs_subsess, burst_subsess


def process_data(data_path, angle):
    data = pd.read_csv(data_path, header=0, usecols=prep_angle[angle][BODY_PART])
    # rename headers:
    headers_names = ['{0}_x'.format(BODY_PART), '{0}_y'.format(BODY_PART), '{0}_z'.format(BODY_PART)]
    body_part_cols = [prep_angle[angle][BODY_PART][0], prep_angle[angle][BODY_PART][1], prep_angle[angle][BODY_PART][2]]
    d = dict(zip(body_part_cols, headers_names))
    partial_data = data.rename(columns=d, inplace=False)
    partial_data = partial_data.drop([0, 1])

    # convert argument to a float type
    partial_data['{0}_x'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_x'.format(BODY_PART)], downcast="float")
    partial_data['{0}_y'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_y'.format(BODY_PART)], downcast="float")
    partial_data['{0}_z'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_z'.format(BODY_PART)], downcast="float")

    return partial_data


def plot_velocity(data):
    diff = data.diff()
    diff_x = diff['{0}_x'.format(BODY_PART)].plot()

    plt.show()
    diff_y = diff['{0}_y'.format(BODY_PART)].plot()
    plt.show()
    diff_z = diff['{0}_z'.format(BODY_PART)].plot()
    # plt.plot(data['{0}_x'.format(BODY_PART)], data['{0}_y'.format(BODY_PART)])
    plt.show()


preprocessing(ELIANNA_DATA_PATH, 'SIDE')







