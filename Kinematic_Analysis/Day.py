import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mat4py import loadmat
import math
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

# dictionary that links the body parts from the analysis to the colomns names
body_cols_side = {'finger': ['DLC_3D_SIDE', 'DLC_3D_SIDE.1', 'DLC_3D_SIDE.2'],
                  'wrist': ['DLC_3D_SIDE.3', 'DLC_3D_SIDE.4', 'DLC_3D_SIDE.5'],
                  'elbow': ['DLC_3D_SIDE.6', 'DLC_3D_SIDE.7', 'DLC_3D_SIDE.8']}
body_cols_top = {'finger': ['DLC_3D_TOP', 'DLC_3D_TOP.1', 'DLC_3D_TOP.2'],
                 'wrist': ['DLC_3D_TOP.3', 'DLC_3D_TOP.4', 'DLC_3D_TOP.5'],
                 'elbow': ['DLC_3D_TOP.6', 'DLC_3D_TOP.7', 'DLC_3D_TOP.8']}

camera_angle = {'SIDE': body_cols_side, 'TOP': body_cols_top}

FS = 120
#todo cut off at 750

def speed(data, body_part):
    x_dat = data['{0}_x'.format(body_part)]
    y_dat = data['{0}_y'.format(body_part)]
    z_dat = data['{0}_z'.format(body_part)]

    temp_data = pd.DataFrame()
    temp_data['x_dat'] = x_dat
    temp_data['y_dat'] = y_dat
    temp_data['z_dat'] = z_dat

    speed = np.linalg.norm(temp_data.values, axis=1)  # does the same as np.sqrt(x_dat**2 + y_dat**2 + z_dat**2)
    # acc['acc_head'] = (speed['head'].diff()) / ((speed['Time'].diff()))
    # velocity = np.linalg.norm(temp_data.diff(), axis=1)

    return list(speed)


def velocity(data, body_part):
    x_dat = data['{0}_x'.format(body_part)]
    y_dat = data['{0}_y'.format(body_part)]
    z_dat = data['{0}_z'.format(body_part)]
    time = pd.Series(np.linspace(0, len(x_dat)/FS, len(x_dat)))

    temp_data = pd.DataFrame()
    temp_data['x_dat'] = x_dat.diff() #/ time.diff()
    temp_data['y_dat'] = y_dat.diff() #/ time.diff()
    temp_data['z_dat'] = z_dat.diff() #/ time.diff()


    #speed_res = pd.DataFrame([speed(data, body_part)])
    velocity_res = np.linalg.norm(temp_data.values, axis=1) #np.linalg.norm(speed_res.diff(axis=1), axis=1)

    return list(velocity_res)

def two_d(data, body_part):
    p = np.array([9.095956756, -2.976828776, 8.603911929])  # point of frame 46 from trail 1 from SIDE project
    q = np.array([6.940065994, -3.029730601, 11.10148329]) # point of frame 120 from trail 50 from SIDE project
    d = np.array([8.885579723, -2.428987168, 11.60233444])# point of frame 25 from trail 46 from SIDE project
    u = q - p
    v = d - q

    w2 = np.array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - v[0] * u[1]])
    w = w2 / np.linalg.norm(w2)
    #print(np.linalg.norm(w))
    T = np.array([2, 4, 7, 5, 1, 6, 3, 8])
    n = 0
    Mov_point_data = data
    Mov_point_data = Mov_point_data.reset_index()
    L = np.zeros([len((Mov_point_data)), 2])

    x = np.cross(w, v) / np.linalg.norm(np.cross(w, v))
    #print(x)
    y = v / np.linalg.norm(v)

    for i in range(len(Mov_point_data)):
        p = np.array(
            [Mov_point_data.loc[i, 'finger_x'], Mov_point_data.loc[i, 'finger_y'], Mov_point_data.loc[i, 'finger_z']])
        proj_point = p - np.dot(p, w) * w
        L[i, :] = np.array([np.dot(x, p), np.dot(y, p)])
    plt.scatter(L[:, 0], L[:, 1], c=np.linspace(1, L.shape[0], L.shape[0]), cmap='Reds')
        #plt.text(L[-1, 0] + 0.1, L[-1, 0] + 0.1, s=str(T[n]) + ',' + str(i), c='red')
    n = n + 1
        # ניסינו להדפיס את מספר המטרה על הגרף באמצעות TEXT עבור הנקודת האחרונה. משהו לא הסתדר
    if L.shape[0] != 0:
        center = L[0]
        plt.xlim(center[0] - 4, center[0] + 4)
        plt.ylim(center[1] - 4, center[1] + 4)
    plt.show()
    return L





class Day:
    date = None
    body_part = None
    angle = None
    analysis_func = None
    data_path = None
    ed_path = None
    info_file = None
    index_file = None
    id = 0
    num_of_subsessions = 0
    subsess_files = None
    hfs_subsess = None
    burst_subsess = None
    csv_indices = None
    trial_data = None
    burst_settings = None

    def __init__(self, date, body_part, angle, analysis_func, data_path, ed_path, info_file, index_file, video_info_file):
        self.date = date
        self.body_part = body_part
        self.angle = angle
        self.analysis_func = analysis_func
        self.data_path = data_path
        self.ed_path = ed_path
        self.info_file = info_file
        self.index_file = index_file
        self.video_info_file = video_info_file
        self.trial_data = pd.DataFrame(
            {'TrialNum': pd.Series([], dtype=str), 'csvNum': pd.Series([], dtype=str),
             'valid': pd.Series([], dtype=str),
             'subSess': pd.Series([], dtype=str), 'target': pd.Series([], dtype=str),
             'update': pd.Series([], dtype=str),
             'HFS': pd.Series([], dtype=str), 'Burst': pd.Series([], dtype=str),  'Go_End': pd.Series([], dtype=str),
             'TrialTimes': pd.Series([], dtype=str)})

    def load_info_file(self):
        """
        Loads info from Info File for this day
        :param info_path:
        :return:
        """
        info_file = loadmat(self.info_file)
        self.id = info_file['DDFparam']['ID']
        self.num_of_subsessions = len(info_file['SESSparam']['SubSess']['Files'])
        self.subsess_files = info_file['SESSparam']['SubSess']['Files']
        self.hfs_subsess = info_file['SESSparam']['fileConfig']['HFS']
        self.burst_subsess = info_file['SESSparam']['fileConfig']['BURST']
        index_file = loadmat(self.index_file)
        self.csv_indices = index_file
        self.burst_settings = info_file['SESSparam']['SubSess']['Electrode']

    def load_ed_files(self):
        """
        Fills Dataframe with each row representing a trial, containing all the relevant information from the ED files
        :return:
        """
        video_info_file = loadmat(self.video_info_file)
        self.vidInfo = video_info_file['vidinfo']
        running_count = 0  # counter for ALL trials from day
        for subsess in range(self.num_of_subsessions):
            files_start, files_end = self.subsess_files[subsess]
            for subsess_file in range(files_end + 1 - files_start):  # file index in files from ONE subsession
                path = self.ed_path.format(trial_num=str(self.id) + '0' + str(subsess + 1), file_num=subsess_file + 1)
                ed_file = loadmat(path)
                invalid_counter = 0
                for trial in range(len(ed_file['trials'])):  # trial is trial index from one file from one subsession
                    running_count += 1
                    file_offset = self.subsess_files[subsess][0] + subsess_file - 1  # file index in all files from day
                    csv_ind = self.find_csv_index(running_count)
                    invalid_counter, trial_times_lst = self.find_trialtimes_index(ed_file, invalid_counter, trial)
                    burst_type = self.find_burst_type(subsess, file_offset)

                    # find from go signal to in periphery
                    is_valid = ed_file['trials'][trial][2]
                    go_end = self.set_go_end(csv_ind, is_valid)

                    temp = {'TrialNum': running_count, 'csvNum': int(csv_ind) if csv_ind else 0, 'subSess': subsess + 1,
                            'valid': is_valid, 'target': ed_file['trials'][trial][4],
                            'update': ed_file['trials'][trial][5], 'HFS': self.hfs_subsess[file_offset],
                            'Burst': burst_type, 'Go_End': go_end,
                            'TrialTimes': trial_times_lst}
                    self.trial_data = pd.concat([self.trial_data, pd.DataFrame([temp])], ignore_index=True)
                    # self.trial_data = self.trial_data.append(temp, ignore_index=True)

    def find_burst_type(self, subsess, file_num):
        # different kinds of burst: 25 = low (assign 2), empty/130 = high (assign 1), from electrode.1.stim.amp
        is_burst = self.burst_subsess[file_num]
        if not is_burst:
            return 0
        elif self.burst_settings[subsess]['Stim'][0]['Freq'] == 25:
            return 2
        else:
            return 1

    def set_go_end(self, csv_ind, is_valid):
        if csv_ind and is_valid:
            go_time = self.vidInfo[1][0][csv_ind - 1]
            end_time = self.vidInfo[1][1][csv_ind - 1]
            if go_time >= 0 and end_time >= 0:
                return [go_time, end_time]
        return False


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

    def process_data(self, trial_index):
        data = pd.read_csv(self.data_path.format(trial_num=trial_index), header=0,
                           usecols=camera_angle[self.angle][self.body_part])
        # rename headers:
        headers_names = ['{0}_x'.format(self.body_part),
                         '{0}_y'.format(self.body_part),
                         '{0}_z'.format(self.body_part)]
        body_part_cols = [camera_angle[self.angle][self.body_part][0],
                          camera_angle[self.angle][self.body_part][1],
                          camera_angle[self.angle][self.body_part][2]]
        d = dict(zip(body_part_cols, headers_names))
        partial_data = data.rename(columns=d, inplace=False)
        partial_data = partial_data.drop([0, 1])

        # convert argument to a float type
        temp_str = '{0}_'.format(self.body_part)
        partial_data[temp_str + 'x'] = pd.to_numeric(partial_data[temp_str + 'x'], downcast="float")
        partial_data[temp_str + 'y'] = pd.to_numeric(partial_data[temp_str + 'y'], downcast="float")
        partial_data[temp_str + 'z'] = pd.to_numeric(partial_data[temp_str + 'z'], downcast="float")

        go = list(self.trial_data.loc[self.trial_data['csvNum'] == trial_index]['Go_End'])[0][0]
        end = list(self.trial_data.loc[self.trial_data['csvNum'] == trial_index]['Go_End'])[0][1]
        if math.isnan(go) or math.isnan(end):
            return partial_data[:0]
        go_diff = [abs(a - go) for a in self.vidInfo[0][trial_index - 1]]
        end_diff = [abs(a - end) for a in self.vidInfo[0][trial_index - 1]]
        gopos = go_diff.index(min(go_diff))
        endpos = end_diff.index(min(end_diff))


        return partial_data[gopos:endpos] #max(750,endpos)]

    def preprocess(self):
        self.load_info_file()
        self.load_ed_files()

    def run_analysis(self):
        #valid_filmed_trials = self.trial_data.loc[self.trial_data['csvNum'] > 0 & self.trial_data['valid'] == 1]
        valid_filmed_trials = self.trial_data.query("csvNum > 0 & valid == 1 & Go_End != False")
        condition_trials = valid_filmed_trials #.loc[self.trial_data['Go_End'] != None]
        temp_data = pd.DataFrame()
        for csv_index in condition_trials['csvNum']:
            if self.is_update_trial(csv_index) and self.is_HFS(csv_index):
                trial_data = self.process_data(csv_index)
                res = self.analysis_func(trial_data, self.body_part)
                #temp_data = pd.concat([temp_data, pd.DataFrame([res])])
                plt.scatter(res[:, 0], res[:, 1], c=np.linspace(1, res.shape[0], res.shape[0]), cmap='Reds')
        # for i in range(len(temp_data)):
        #     plt.plot(temp_data[i], label=i)
        #avg = temp_data.mean()
        #plt.plot(avg)
        # plt.legend()
        plt.title("{angle} {analysis}".format(angle=self.angle, analysis=self.analysis_func.__name__))
        plt.show()

    def is_update_trial(self, csv_index):
        if math.isnan(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['update']):
            return False
        return True

    def is_HFS(self, csv_index):
        if list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['HFS'])[0] == 0:
            return False
        return True


if __name__ == "__main__":
    DATA_PATH = r'Z:\elianna.rosenschein\n{date}\DLC_Analysis\nana-trial{trial_num}_DLC_3D_{angle}.csv'
    ED_PATH = r'Z:\elianna.rosenschein\n{date}\EDfiles\n{trial_num}ee.{file_num}.mat'
    INFO_FILE = r'Z:\elianna.rosenschein\n{date}\Info\n{date}_param.mat'
    INDEX_FILE = r'Z:\elianna.rosenschein\alignment_indices_n{date}.mat'  # Exported from Nirvik's Matlab code
    VIDEO_INFO_FILE = r'Z:\elianna.rosenschein\vidInfo_n{date}.mat'

    BODY_PART = 'finger'  # body part that we want to analyze
    ANGLE = 'SIDE'
    DATE = '270122'
    ANALYSIS_FUNC = two_d

    data_path = DATA_PATH.format(date=DATE, trial_num='{trial_num}', angle=ANGLE)
    ed_path = ED_PATH.format(date=DATE, trial_num='{trial_num}', file_num='{file_num}')
    info_file = INFO_FILE.format(date=DATE)
    index_file = INDEX_FILE.format(date=DATE)
    video_info_file = VIDEO_INFO_FILE.format(date=DATE)

    day = Day(DATE, BODY_PART, ANGLE, ANALYSIS_FUNC, data_path, ed_path, info_file, index_file, video_info_file)
    day.preprocess()
    day.run_analysis()

#TODO ALIGN TO MOVEMENT ONSET
#TODO AVERAGE LOCATION DATA?
#TODO JOINT ANGLES