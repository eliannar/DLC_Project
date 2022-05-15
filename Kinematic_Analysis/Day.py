import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mat4py import loadmat

DATA_PATH = r'Z:\elianna.rosenschein\n270122\DLC_Analysis\nana-trial{0}_DLC_3D_SIDE.csv '
ED_PATH = r'Z:\elianna.rosenschein\n270122\EDfiles\n{0}ee.{1}.mat'
INFO_FILE = r'Z:\elianna.rosenschein\n270122\Info\n270122_param.mat'
INDEX_FILE = r'Z:\elianna.rosenschein\alignment_indices_n270122.mat'
VIDEO_INFO_FILE = r'Z:\elianna.rosenschein\vidInfo_n270122'

# dictionary that links the body parts from the analysis to the colomns names
BODY_PART = 'finger'  #body part that we want to analyze
body_cols_side = {'finger': ['DLC_3D_SIDE', 'DLC_3D_SIDE.1', 'DLC_3D_SIDE.2'],
                      'wrist': ['DLC_3D_SIDE.3', 'DLC_3D_SIDE.4', 'DLC_3D_SIDE.5'],
                      'elbow': ['DLC_3D_SIDE.6', 'DLC_3D_SIDE.7', 'DLC_3D_SIDE.8']}
body_cols_top = {'finger': ['DLC_3D_TOP', 'DLC_3D_TOP.1', 'DLC_3D_TOP.2'],
                 'wrist': ['DLC_3D_TOP.3', 'DLC_3D_TOP.4', 'DLC_3D_TOP.5'],
                 'elbow': ['DLC_3D_TOP.6', 'DLC_3D_TOP.7', 'DLC_3D_TOP.8']}


prep_angle = {'SIDE': body_cols_side, 'TOP': body_cols_top}

class Day:
    id = 0
    num_of_subsessions = 0
    # I need a table with day, subsession,
    subsess_data = {}

    def __init__(self):
        self.trial_data = pd.DataFrame({'TrialNum': [], 'csvNum': [], 'valid': [], 'subSess': [], 'target': [], 'update': [], 'HFS': [], 'Burst': []})


    def load_info_file(self, info_path):
        info_file = loadmat(info_path)
        self.id = info_file['DDFparam']['ID']
        self.num_of_subsessions = len(info_file['SESSparam']['SubSess']['Files'])
        self.subsess_files = info_file['SESSparam']['SubSess']['Files']
        self.hfs_subsess = info_file['SESSparam']['fileConfig']['HFS']
        self.burst_subsess = info_file['SESSparam']['fileConfig']['BURST']
        index_file = loadmat(INDEX_FILE)
        self.csv_indices = index_file
        self.load_ed_files()

    def load_ed_files(self):
        running_count = 0
        for i in range(self.num_of_subsessions):
            self.subsess_data[i] = []
            files_start = self.subsess_files[i][0]
            files_end = self.subsess_files[i][1]
            for j in range(files_end + 1 - files_start):
                path = ED_PATH.format(str(self.id) + '0' + str(i + 1), j + 1)
                ed_file = loadmat(path)
                for t in range(len(ed_file['trials'])):
                    running_count += 1
                    if [running_count] in self.csv_indices['I']:
                        general_ind = self.csv_indices['I'].index([running_count])
                        csv_ind = self.csv_indices['J'][general_ind][0]
                    else:
                        csv_ind = None
                    j_offset = self.subsess_files[i][0] + j - 1
                    #TODO different kinds of burst: 25 = low (assign 2), empty/130 = high (assign 1), from electrode.1.stim.amp
                    #TODO trial times
                    temp = {'TrialNum': running_count, 'csvNum': csv_ind, 'subSess': i + 1, 'valid': ed_file['trials'][t][2], 'target': ed_file['trials'][t][4], 'update': ed_file['trials'][t][5], 'HFS': self.hfs_subsess[j_offset], 'Burst': self.burst_subsess[j_offset]} #TODO change j index
                    self.trial_data = self.trial_data.append(temp, ignore_index=True)


    def process_data(data_path, angle):
        data = pd.read_csv(data_path, header=0, usecols=prep_angle[angle][BODY_PART])
        # rename headers:
        headers_names = ['{0}_x'.format(BODY_PART), '{0}_y'.format(BODY_PART), '{0}_z'.format(BODY_PART)]
        body_part_cols = [prep_angle[angle][BODY_PART][0], prep_angle[angle][BODY_PART][1],
                          prep_angle[angle][BODY_PART][2]]
        d = dict(zip(body_part_cols, headers_names))
        partial_data = data.rename(columns=d, inplace=False)
        partial_data = partial_data.drop([0, 1])

        # convert argument to a float type
        partial_data['{0}_x'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_x'.format(BODY_PART)],
                                                                downcast="float")
        partial_data['{0}_y'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_y'.format(BODY_PART)],
                                                                downcast="float")
        partial_data['{0}_z'.format(BODY_PART)] = pd.to_numeric(partial_data['{0}_z'.format(BODY_PART)],
                                                                downcast="float")
        return partial_data



day = Day()
day.load_info_file(INFO_FILE)