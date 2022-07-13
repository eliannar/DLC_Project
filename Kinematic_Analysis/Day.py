import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from mat4py import loadmat
from mpl_toolkits.mplot3d import Axes3D
from pandasql import sqldf
from scipy.signal import savgol_filter

pysqldf = lambda q: sqldf(q, globals())
PERCENTAGE = 1
KERNEL_SIZE = 20

# dictionary that links the body parts from the analysis to the colomns names
body_cols_side = {'finger': ['DLC_3D_SIDE', 'DLC_3D_SIDE.1', 'DLC_3D_SIDE.2'],
                  'wrist': ['DLC_3D_SIDE.3', 'DLC_3D_SIDE.4', 'DLC_3D_SIDE.5'],
                  'elbow': ['DLC_3D_SIDE.6', 'DLC_3D_SIDE.7', 'DLC_3D_SIDE.8']}
body_cols_top = {'finger': ['DLC_3D_TOP', 'DLC_3D_TOP.1', 'DLC_3D_TOP.2'],
                 'wrist': ['DLC_3D_TOP.3', 'DLC_3D_TOP.4', 'DLC_3D_TOP.5'],
                 'elbow': ['DLC_3D_TOP.6', 'DLC_3D_TOP.7', 'DLC_3D_TOP.8']}

camera_angle = {'SIDE': body_cols_side, 'TOP': body_cols_top}

FS = 120


def check_delta(trial_data, body_part):
    if trial_data.empty:
        return 0
    kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
    vel = calculate_velocity(trial_data, body_part)
    vel = vel / np.nanmax(vel)
    yhat = np.convolve(vel, kernel, mode='same')
    yhat = yhat[~np.isnan(yhat)]
    return max(yhat) - yhat[0]


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


def ms_to_frames(ms):
    frame = ms * (FS / 1000)
    return int(frame)


def calculate_velocity(data, body_part):
    x_dat = data['{0}_x'.format(body_part)]
    y_dat = data['{0}_y'.format(body_part)]
    z_dat = data['{0}_z'.format(body_part)]
    time = pd.Series(np.linspace(0, len(x_dat) / FS, len(x_dat)))

    temp_data = pd.DataFrame()
    temp_data['x_dat'] = x_dat.diff()  # / time.diff()
    temp_data['y_dat'] = y_dat.diff()  # / time.diff()
    temp_data['z_dat'] = z_dat.diff()  # / time.diff()

    # speed_res = pd.DataFrame([speed(data, body_part)])
    velocity_res = np.linalg.norm(temp_data.values, axis=1)  # np.linalg.norm(speed_res.diff(axis=1), axis=1)

    return list(velocity_res)


def two_d(data, body_part, plot=True, title=None):
    p = np.array([9.095956756, -2.976828776, 8.603911929])  # point of frame 46 from trail 1 from SIDE project
    q = np.array([6.940065994, -3.029730601, 11.10148329])  # point of frame 120 from trail 50 from SIDE project
    d = np.array([8.885579723, -2.428987168, 11.60233444])  # point of frame 25 from trail 46 from SIDE project
    u = q - p
    v = d - q

    w2 = np.array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - v[0] * u[1]])
    w = w2 / np.linalg.norm(w2)
    # print(np.linalg.norm(w))
    T = np.array([2, 4, 7, 5, 1, 6, 3, 8])
    n = 0
    Mov_point_data = data
    Mov_point_data = Mov_point_data.reset_index()
    L = np.zeros([len((Mov_point_data)), 2])

    x = np.cross(w, v) / np.linalg.norm(np.cross(w, v))
    # print(x)
    y = v / np.linalg.norm(v)

    for i in range(len(Mov_point_data)):
        p = np.array(
            [Mov_point_data.loc[i, 'finger_x'], Mov_point_data.loc[i, 'finger_y'], Mov_point_data.loc[i, 'finger_z']])
        proj_point = p - np.dot(p, w) * w
        L[i, :] = np.array([np.dot(x, p), np.dot(y, p)])
    # plt.scatter(L[:, 0], L[:, 1], c=np.linspace(1, L.shape[0], L.shape[0]), cmap='Reds')
    # plt.plot(L[:, 0] -  L[0][0], L[:, 1]-  L[0][1])
    # plt.text(L[-1, 0] + 0.1, L[-1, 0] + 0.1, s=str(T[n]) + ',' + str(i), c='red')
    n = n + 1
    # ניסינו להדפיס את מספר המטרה על הגרף באמצעות TEXT עבור הנקודת האחרונה. משהו לא הסתדר
    if plot:
        if L.shape[0] != 0 and np.random.binomial(1, PERCENTAGE):
            plt.plot(L[:, 0] - L[0][0], L[:, 1] - L[0][1])
            center = L[0]
            plt.xlim(- 7, 7)
            plt.ylim(-7, 7)
        if title:
            plt.title(title)
        plt.show()
    return L


def rename_headers(data, body_part, angle):
    headers_names = ['{0}_x'.format(body_part),
                     '{0}_y'.format(body_part),
                     '{0}_z'.format(body_part)]
    body_part_cols = [camera_angle[angle][body_part][0],
                      camera_angle[angle][body_part][1],
                      camera_angle[angle][body_part][2]]
    d = dict(zip(body_part_cols, headers_names))
    partial_data = data.rename(columns=d, inplace=False)
    return partial_data.drop([0, 1])


def convert_val_to_str(data, body_part):
    temp_str = '{0}_'.format(body_part)
    data[temp_str + 'x'] = pd.to_numeric(data[temp_str + 'x'], downcast="float")
    data[temp_str + 'y'] = pd.to_numeric(data[temp_str + 'y'], downcast="float")
    data[temp_str + 'z'] = pd.to_numeric(data[temp_str + 'z'], downcast="float")
    return data


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

    def __init__(self, date, body_part, angle, analysis_func, data_path, ed_path, info_file, index_file,
                 video_info_file):
        analysis_func_dict = {"plot_clusters": self.plot_clusters, "plot_2d_clusters": self.plot_2d_clusters,
                              "plot_velocity": self.plot_velocity, "two_d": self.plot_2d_trajectory}
        self.date = date
        self.body_part = body_part
        self.angle = angle
        self.analysis_func = analysis_func_dict[analysis_func]
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
             'HFS': pd.Series([], dtype=str), 'Burst': pd.Series([], dtype=str), 'Go_End': pd.Series([], dtype=str),
             'TrialTimes': pd.Series([], dtype=str), 'VidTicks': pd.Series([], dtype=str)})

    def load_info_file(self):
        """
        Loads info from Info File for this day
        :param info_path:
        :return:
        """
        lcl_info_file = loadmat(self.info_file)
        self.id = lcl_info_file['DDFparam']['ID']
        self.num_of_subsessions = len(lcl_info_file['SESSparam']['SubSess']['Files'])
        self.subsess_files = lcl_info_file['SESSparam']['SubSess']['Files']
        self.hfs_subsess = lcl_info_file['SESSparam']['fileConfig']['HFS']
        self.burst_subsess = lcl_info_file['SESSparam']['fileConfig']['BURST']
        self.csv_indices = loadmat(self.index_file)
        self.burst_settings = lcl_info_file['SESSparam']['SubSess']['Electrode']

    def load_ed_files(self):
        """
        Fills Dataframe with each row representing a trial, containing all the relevant information from the ED files
        :return:
        """
        lcl_video_info_file = loadmat(self.video_info_file)
        vidinfo = pd.DataFrame(lcl_video_info_file['vidinfo']).transpose()
        all_vidticks = vidinfo[0]  # contains also invalid trials
        running_count = 0  # counter for ALL trials from day
        for subsess in range(self.num_of_subsessions):
            files_start, files_end = self.subsess_files[subsess]
            for subsess_file in range(files_end + 1 - files_start):  # file index in files from ONE subsession
                path = self.ed_path.format(trial_num=str(self.id) + '0' + str(subsess + 1), file_num=subsess_file + 1)
                ed_file = loadmat(path)
                invalid_counter = 0
                for trial in range(len(ed_file['trials'])):  # trial is trial index from one file from one subsession
                    running_count += 1
                    is_valid = ed_file['trials'][trial][2]
                    if not is_valid:
                        invalid_counter += 1
                    if True:  # is_valid:
                        file_offset = self.subsess_files[subsess][
                                          0] + subsess_file - 1  # file index in all files from day
                        trial_times_lst = self.find_trialtimes_index(ed_file, invalid_counter, trial)
                        csv_ind = self.find_csv_index(running_count)
                        burst_type = self.find_burst_type(subsess, file_offset)
                        vidticks = all_vidticks[running_count - 1] if csv_ind is not None else []
                        go_end = self.set_go_end(csv_ind, trial_times_lst)  # find from go signal to in periphery

                        temp = {'TrialNum': running_count,
                                'csvNum': int(csv_ind) if csv_ind else 0,
                                'subSess': subsess + 1,
                                'valid': is_valid,
                                'target': ed_file['trials'][trial][4],
                                'update': ed_file['trials'][trial][5],
                                'HFS': self.hfs_subsess[file_offset],
                                'Burst': burst_type,
                                'Go_End': go_end,
                                'TrialTimes': trial_times_lst,
                                'VidTicks': vidticks}
                        self.trial_data = pd.concat([self.trial_data, pd.DataFrame([temp])], ignore_index=True)
        self.trial_data = self.trial_data.query("csvNum > 0 & valid == 1")  # & Go_End != False")

    def find_burst_type(self, subsess, file_num):
        # different kinds of burst: 25 = low (assign 2), empty/130 = high (assign 1), from electrode.1.stim.amp
        is_burst = self.burst_subsess[file_num]
        if not is_burst:
            return 0
        elif self.burst_settings[subsess]['Stim'][0]['Freq'] == 25:
            return 2
        else:
            return 1

    def set_go_end(self, csv_ind, trial_times):
        if csv_ind:
            go_time = trial_times[4]  # self.vidInfo[1][0][csv_ind - 1]
            end_time = trial_times[7]  # self.vidInfo[1][1][csv_ind - 1]
            if go_time >= 0 and end_time >= 0:
                return [go_time, end_time]
        return False

    def find_csv_index(self, running_count):
        if [running_count] in self.csv_indices['I']:
            general_ind = self.csv_indices['I'].index([running_count])  # includes invalid trials
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
        trial_times_ind = trial_index - invalid_trial_counter
        trial_times_lst = edFile['TrialTimes'][trial_times_ind]
        return trial_times_lst

    def plot_velocity(self, relevant_trials, body_part, title="", aggregate=False):
        temp_data = pd.DataFrame()
        for csv_index in relevant_trials['csvNum']:
            trial_data = self.process_data(csv_index)
            if trial_data.empty:
                continue
            res = calculate_velocity(trial_data, body_part)
            res = res / np.nanmax(res)
            temp_data = pd.concat([temp_data, pd.DataFrame([res])], ignore_index=True)
        if aggregate:
            avg = temp_data.mean()
            plt.plot(avg, label='{0} mean'.format(self.analysis_func.__name__))
            yhat = savgol_filter(avg, 20, 4)  # window size 51, polynomial order 3
            plt.plot(yhat, label='smoothed')
            plt.legend()
        else:
            for i in range(len(temp_data)):
                kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
                yhat = np.convolve(temp_data.iloc[i], kernel, mode='same')
                # yhat = savgol_filter(temp_data.iloc[i], 20, 4)  # window size 51, polynomial order 3
                plt.plot(yhat, label=i)
        plt.title(title)
        plt.show()

    def plot_clusters(self, relevant_trials, body_part, title=""):
        color_dict = {1: 'b', 2: 'r', 3: 'g', 4: 'c', 5: 'm', 6: 'y', 7: 'yellow', 8: 'brown'}
        fig = plt.figure()
        ax = Axes3D(fig)
        for csv_index in relevant_trials['csvNum']:
            trial_data = self.process_data(csv_index)
            if trial_data.empty:
                continue
            x, y, z = trial_data.iloc[-1] - trial_data.iloc[0]
            target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['target'])[0]
            ax.scatter(x, y, z, color=color_dict[target])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        plt.title(title)
        plt.show()

    def plot_2d_trajectory(self, relevant_trials, body_part, title=""):
        color_dict = {1: 'b', 2: 'r', 3: 'g', 4: 'c', 5: 'm', 6: 'y', 7: 'yellow', 8: 'brown'}
        plt.figure()
        for csv_index in relevant_trials['csvNum']:
            trial_data = self.process_data(csv_index)
            if trial_data.empty:
                continue
            projection = two_d(trial_data, self.body_part, plot=False)
            x = projection[:, 0] - projection[:, 0][0]
            y = projection[:, 1] - projection[:, 1][0]
            init_target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['target'])[0]
            update_target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['update'])[0]
            update_target = update_target if update_target is not None else 'k'
            plt.plot(x, y, color=color_dict[init_target])
        plt.xlim(-4, 4)
        plt.ylim(-4,4)
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title(title)
        plt.show()

    def plot_2d_clusters(self, relevant_trials, body_part, title=""):
        color_dict = {1: 'b', 2: 'r', 3: 'g', 4: 'c', 5: 'm', 6: 'y', 7: 'yellow', 8: 'brown'}
        plt.figure()
        for csv_index in relevant_trials['csvNum']:
            trial_data = self.process_data(csv_index)
            if trial_data.empty:
                continue
            projection = two_d(trial_data, self.body_part, plot=False)
            x = projection[:, 0][-1] - projection[:, 0][0]
            y = projection[:, 1][-1] - projection[:, 1][0]
            target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['target'])[0]
            init_target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['target'])[0]
            update_target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['update'])[0]
            update_target = update_target if update_target is not None else 'k'
            plt.scatter(x, y, color=color_dict[init_target])
        plt.scatter(0, 0, color='k')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title(title)
        plt.show()

    def process_data(self, trial_index):
        data = pd.read_csv(self.data_path.format(trial_num=trial_index), header=0,
                           usecols=camera_angle[self.angle][self.body_part])
        # rename headers:
        partial_data = rename_headers(data, self.body_part, self.angle)

        # convert argument to a float type
        partial_data = convert_val_to_str(partial_data, self.body_part)

        go = list(self.trial_data.loc[self.trial_data['csvNum'] == trial_index]['Go_End'])[0][0]
        end = list(self.trial_data.loc[self.trial_data['csvNum'] == trial_index]['Go_End'])[0][1]
        vidticks = self.trial_data.loc[self.trial_data['csvNum'] == trial_index].reset_index()['VidTicks'][0]
        if math.isnan(go) or math.isnan(end) or vidticks == []:
            return partial_data[:0]

        go_diff = [abs(a - go) for a in vidticks]
        end_diff = [abs(a - end) for a in vidticks]
        gopos = go_diff.index(min(go_diff))
        start_pos = max(0, gopos - 30)
        endpos = min(end_diff.index(min(end_diff)), start_pos + ms_to_frames(1000))
        movement_onset_pos = self.find_mvmnt_pos(partial_data[start_pos:endpos])
        if check_delta(partial_data[start_pos + movement_onset_pos:endpos], self.body_part) < 0.2:
            return partial_data[:0]
        # if np.random.binomial(1, 0.3):#trial_index in (119, 139, 372, 382, 387, 637):
        #     kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
        #     vel = calculate_velocity(partial_data[start_pos:endpos], self.body_part)
        #     vel = vel / np.nanmax(vel)
        #     yhat = np.convolve(vel, kernel, mode='same')
        #     # yhat = savgol_filter(temp_data.iloc[i], 20, 4)  # window size 51, polynomial order 3
        #     plt.plot(yhat, label=trial_index)
        #     plt.legend()
        #     plt.scatter(0, yhat[0], marker='o')
        #     plt.scatter(movement_onset_pos, yhat[movement_onset_pos], marker='v')
        #     plt.scatter(len(yhat), yhat[-1], marker='^')
        #     plt.show()
        return partial_data[start_pos + movement_onset_pos:endpos]

    def find_mvmnt_pos(self, data):
        mvmnt_pos = 0
        if not data.empty:
            vel = calculate_velocity(data, self.body_part)
            kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
            vel = vel / np.nanmax(vel)
            vel = list(np.convolve(vel, kernel, mode='same'))
            max_vel = max([a for a in vel if not np.isnan(a)], default=np.nan)
            if not np.isnan(max_vel):
                max_pos = vel.index(max_vel)
                if max_pos > 1:
                    diff = [abs(a - (max_vel / 6)) for a in vel[:max_pos + 1]]
                    mvmnt_pos = diff.index(min([a for a in diff if not np.isnan(a)]))
        return mvmnt_pos  # - ms_to_frames(1)

    def preprocess(self):
        self.load_info_file()
        self.load_ed_files()

    def is_desired_update_target(self, target, csv_index):
        if list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['update'] == target)[0]:
            return True
        else:
            return False

    def is_desired_init_target(self, target, csv_index):
        if list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['target'] == target)[0]:
            return True
        else:
            return False

    def meets_requirements(self, hfs=True, update=True, init_target=None, update_target=None):
        indices = np.full(self.trial_data.shape[0], True)
        if hfs is not None:
            hfs_inds = self.trial_data.apply(
                lambda row: (hfs and self.is_HFS(row['csvNum'])) or (not hfs and not self.is_HFS(row['csvNum'])),
                axis=1)
            indices = np.logical_and(indices, hfs_inds)
        if update is not None:
            update_inds = self.trial_data.apply(lambda row: (update and self.is_update_trial(row['csvNum'])) or (
                    not update and not self.is_update_trial(row['csvNum'])), axis=1)
            indices = np.logical_and(indices, update_inds)
        if init_target is not None:
            init_target_inds = self.trial_data['target'] == init_target
            indices = np.logical_and(indices, init_target_inds)
        if update_target is not None:
            update_target_inds = self.trial_data['update'] == update_target
            indices = np.logical_and(indices, update_target_inds)
        return self.trial_data.iloc[list(indices)]

    def run_analysis(self, aggregate=True, hfs=True, update=True, init_target=None, update_target=None, title=None):
        relevant_trials = self.meets_requirements(hfs, update, init_target, update_target)
        if title is None:
            title = "{angle} {analysis}".format(angle=self.angle, analysis=self.analysis_func.__name__) + \
                    (", HFS" if hfs else "") + (", update" if update else "") + \
                    (("\ninit_target: " + str(init_target)) if init_target else "") + \
                    (("\nupdate_target: " + str(update_target)) if update_target else "")
        self.analysis_func(relevant_trials, self.body_part, title=title)

    def is_update_trial(self, csv_index):
        if math.isnan(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['update']):
            return False
        return True

    def is_HFS(self, csv_index):
        if list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['HFS'])[0] == 0:
            return False
        return True

    def calc_mean_delta(self, all_data):
        """
        helper function that finds distribution of differences between movement onset velocity and peak velocity
        :param all_data:
        :return: we should cut off the ones that are less than 0.2
        """
        delta_list = []
        for csv_index in all_data['csvNum']:
            trial_data = self.process_data(csv_index)
            delta_list.append(check_delta(trial_data, self.body_part))
        pd.Series(delta_list).plot.kde()
        plt.scatter(0.1, 0.5)
        plt.show()
        print('buffer')


if __name__ == "__main__":
    DATA_PATH = r'Z:\elianna.rosenschein\n{date}\DLC_Analysis\nana-trial{trial_num}_DLC_3D_{angle}.csv'
    ED_PATH = r'Z:\elianna.rosenschein\n{date}\EDfiles\n{trial_num}ee.{file_num}.mat'
    INFO_FILE = r'Z:\elianna.rosenschein\n{date}\Info\n{date}_param.mat'
    INDEX_FILE = r'Z:\elianna.rosenschein\alignment_indices_n{date}.mat'  # Exported from Nirvik's Matlab code
    VIDEO_INFO_FILE = r'Z:\elianna.rosenschein\vidInfo_n{date}.mat'  # Exported from Nirvik's Matlab code

    BODY_PART = 'finger'  # body part that we want to analyze
    ANGLE = 'SIDE'
    DATE = '270122'
    ANALYSIS_FUNC = "two_d"

    data_path = DATA_PATH.format(date=DATE, trial_num='{trial_num}', angle=ANGLE)
    ed_path = ED_PATH.format(date=DATE, trial_num='{trial_num}', file_num='{file_num}')
    info_file = INFO_FILE.format(date=DATE)
    index_file = INDEX_FILE.format(date=DATE)
    video_info_file = VIDEO_INFO_FILE.format(date=DATE)

    day = Day(DATE, BODY_PART, ANGLE, ANALYSIS_FUNC, data_path, ed_path, info_file, index_file, video_info_file)
    day.preprocess()
    day.run_analysis(aggregate=False, hfs=False, update=False, init_target=6, update_target=None)

# TODO JOINT ANGLES
