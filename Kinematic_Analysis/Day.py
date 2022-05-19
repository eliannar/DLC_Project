import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mat4py import loadmat

# TODO format date
DATA_PATH = r'Z:\elianna.rosenschein\n270122\DLC_Analysis\nana-trial{0}_DLC_3D_SIDE.csv '
ED_PATH = r'Z:\elianna.rosenschein\n270122\EDfiles\n{0}ee.{1}.mat'
INFO_FILE = r'Z:\elianna.rosenschein\n270122\Info\n270122_param.mat'
INDEX_FILE = r'Z:\elianna.rosenschein\alignment_indices_n270122.mat'
VIDEO_INFO_FILE = r'Z:\elianna.rosenschein\vidInfo_n270122'

# dictionary that links the body parts from the analysis to the colomns names
BODY_PART = 'finger'  # body part that we want to analyze
ANGLE = 'SIDE'
body_cols_side = {'finger': ['DLC_3D_SIDE', 'DLC_3D_SIDE.1', 'DLC_3D_SIDE.2'],
                  'wrist': ['DLC_3D_SIDE.3', 'DLC_3D_SIDE.4', 'DLC_3D_SIDE.5'],
                  'elbow': ['DLC_3D_SIDE.6', 'DLC_3D_SIDE.7', 'DLC_3D_SIDE.8']}
body_cols_top = {'finger': ['DLC_3D_TOP', 'DLC_3D_TOP.1', 'DLC_3D_TOP.2'],
                 'wrist': ['DLC_3D_TOP.3', 'DLC_3D_TOP.4', 'DLC_3D_TOP.5'],
                 'elbow': ['DLC_3D_TOP.6', 'DLC_3D_TOP.7', 'DLC_3D_TOP.8']}

camera_angle = {'SIDE': body_cols_side, 'TOP': body_cols_top}


class Day:
    id = 0
    num_of_subsessions = 0
    subsess_files = None
    hfs_subsess = None
    burst_subsess = None
    csv_indices = None
    trial_data = None
    burst_settings = None

    def __init__(self):
        self.trial_data = pd.DataFrame(
            {'TrialNum': [], 'csvNum': [], 'valid': [], 'subSess': [], 'target': [], 'update': [], 'HFS': [],
             'Burst': [], 'TrialTimes': []})

    def load_info_file(self, info_path):
        """
        Loads info from Info File for this day
        :param info_path:
        :return:
        """
        info_file = loadmat(info_path)
        self.id = info_file['DDFparam']['ID']
        self.num_of_subsessions = len(info_file['SESSparam']['SubSess']['Files'])
        self.subsess_files = info_file['SESSparam']['SubSess']['Files']
        self.hfs_subsess = info_file['SESSparam']['fileConfig']['HFS']
        self.burst_subsess = info_file['SESSparam']['fileConfig']['BURST']
        index_file = loadmat(INDEX_FILE)
        self.csv_indices = index_file
        self.burst_settings = info_file['SESSparam']['SubSess']['Electrode']

    def load_ed_files(self):
        """
        Fills Dataframe with each row representing a trial, containing all the relevant information from the ED files
        :return:
        """
        running_count = 0  # counter for ALL trials from day
        for subsess in range(self.num_of_subsessions):
            files_start, files_end = self.subsess_files[subsess]
            for subsess_file in range(
                    files_end + 1 - files_start):  # subsess_file is file index in files from ONE subsession
                path = ED_PATH.format(str(self.id) + '0' + str(subsess + 1), subsess_file + 1)
                ed_file = loadmat(path)
                invalid_counter = 0
                for trial in range(len(ed_file['trials'])):  # trial is trial index from one file from one subsession
                    running_count += 1
                    file_offset = self.subsess_files[subsess][0] + subsess_file - 1  # file index in all files from day
                    csv_ind = self.find_csv_index(running_count)
                    invalid_counter, trial_times_lst = self.find_trialtimes_index(ed_file, invalid_counter, trial)
                    burst_type = self.find_burst_type(subsess, file_offset)
                    temp = {'TrialNum': running_count, 'csvNum': csv_ind, 'subSess': subsess + 1,
                            'valid': ed_file['trials'][trial][2], 'target': ed_file['trials'][trial][4],
                            'update': ed_file['trials'][trial][5], 'HFS': self.hfs_subsess[file_offset],
                            'Burst': burst_type,
                            'TrialTimes': trial_times_lst}
                    self.trial_data = self.trial_data.append(temp, ignore_index=True)

    def find_burst_type(self, subsess, file_num):
        # different kinds of burst: 25 = low (assign 2), empty/130 = high (assign 1), from electrode.1.stim.amp
        is_burst = self.burst_subsess[file_num]
        if not is_burst:
            return 0
        elif self.burst_settings[subsess]['Stim'][0]['Freq'] == 25:
            return 2
        else:
            return 1

    def find_csv_index(self, running_count):
        if [running_count] in self.csv_indices['I']:
            general_ind = self.csv_indices['I'].index([running_count])
            return self.csv_indices['J'][general_ind][0]
        else:
            return None

    def find_trialtimes_index(self, edFile, invalid_trial_counter, trial_index):
        """
        returns an updated invalid_trial_counter (adds 1 if this trial is invalid), and returns a shifted counter if
        for TrialTimes (if this trial is valid)
        :param edFile:
        :param invalid_trial_counter:
        :param trial_index:
        :return:
        """
        ret1 = invalid_trial_counter
        ret2 = None
        if not (edFile['trials'][trial_index][2]):
            ret1 += 1
        if edFile['trials'][trial_index][2]:
            trial_times_ind = trial_index - invalid_trial_counter
            ret2 = edFile['TrialTimes'][trial_times_ind]
        return ret1, ret2

    def process_data(self, trial_index, data_path):
        data = pd.read_csv(data_path.format(trial_index), header=0, usecols=camera_angle[ANGLE][BODY_PART])
        # rename headers:
        headers_names = ['{0}_x'.format(BODY_PART), '{0}_y'.format(BODY_PART), '{0}_z'.format(BODY_PART)]
        body_part_cols = [camera_angle[ANGLE][BODY_PART][0], camera_angle[ANGLE][BODY_PART][1],
                          camera_angle[ANGLE][BODY_PART][2]]
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

    def preprocess(self):
        self.load_info_file(INFO_FILE)
        self.load_ed_files()

    def run_analysis(self):
        valid_filmed_trials = self.trial_data.loc[self.trial_data['csvNum'].notna() & self.trial_data['valid'] == 1]
        for csv_index in valid_filmed_trials['csvNum']:
            self.process_data(csv_index, DATA_PATH)
            # TODO find rows in excel that correspond to trialtimes
            # TODO run analysis function


if __name__ == "__main__":
    day = Day()
    day.preprocess()
