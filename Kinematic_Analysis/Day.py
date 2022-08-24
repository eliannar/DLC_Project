import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import rdp
from scipy import stats
import helpers
from helpers import angle, check_delta, speed, ms_to_frames, frames_to_ms, calculate_velocity, two_d, rename_headers, convert_val_to_str

from mat4py import loadmat
from mpl_toolkits.mplot3d import Axes3D
from pandasql import sqldf
from scipy.signal import savgol_filter

pysqldf = lambda q: sqldf(q, globals())
PERCENTAGE = 1
KERNEL_SIZE = 40





# # targets based on mean endpoint in this specific projection
# target_dict = {1: (0.43902833714569894, -1.4275273691008923), 2: (3.3323351848647977, -1.0233081538278552),
#                3: (3.0724469287156095, 0.8049932081901696), 4: (1.2696497115583854, 0.952799133154195),
#                5: (-0.39409753186542545, 1.1234683471889075), 6: (-1.6717630953467482, 0.6480369523813078),
#                7: (-2.3354596891361936, 0.031703546917977755), 8: (-0.46727825652621824, -1.440262648445679)}


# targets scaled from Nirvik's coordinates, and flipped on the y axis
scale_factor = 3.2
target_dict = {1: (0, -scale_factor * 0.68), 2: (scale_factor * 0.476, -scale_factor * 0.476),
               3: (scale_factor * 0.68, 0), 4: (scale_factor * 0.476, -scale_factor * -0.476),
               5: (0, -scale_factor * -0.63), 6: (scale_factor * -0.476, -scale_factor * -0.476),
               7: (scale_factor * -0.68, 0), 8: (scale_factor * -0.476, -scale_factor * 0.476)}


FS = 120


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
    a1 = 0
    a2 = 0
    b1 = 0
    b2 = 0

    def __init__(self, date, body_part, angle, analysis_func, data_path, ed_path, info_file, index_file,
                 video_info_file):
        analysis_func_dict = {"plot_clusters": self.plot_clusters, "plot_2d_clusters": self.plot_2d_clusters,
                              "plot_velocity": self.plot_velocity, "two_d": self.plot_2d_trajectory,
                              "turning_point": self.plot_2d_trajectory_with_turning_point}
        self.date = date
        self.res_dict = {}
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
             'TargetJump': pd.Series([], dtype=str),
             'updateDelay': pd.Series([], dtype=str),
             'TrialTimes': pd.Series([], dtype=str), 'VidTicks': pd.Series([], dtype=str),
             'projectionX': pd.Series([], dtype=str), 'projectionY': pd.Series([], dtype=str)})

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
                                'TargetJump': trial_times_lst[5],
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
            go_time = trial_times[3]  # self.vidInfo[1][0][csv_ind - 1]
            end_time = trial_times[10]  # self.vidInfo[1][1][csv_ind - 1] #todo maybe change back to 7
            if go_time >= 0 and end_time >= 0:
                return [go_time, end_time]
        return False

    def find_csv_index(self, running_count):
        if running_count in self.csv_indices['I']:
            general_ind = self.csv_indices['I'].index(running_count)  # includes invalid trials
            return self.csv_indices['J'][general_ind]
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
            trial_data = self.process_data(csv_index)#, from_jump=True)
            if trial_data.empty:
                continue
            res = calculate_velocity(trial_data, body_part)
            res = res / np.nanmax(res)
            kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
            yhat = np.convolve(res, kernel, mode='same')
            temp_data = pd.concat([temp_data, pd.DataFrame([yhat])], ignore_index=True)

            ##
            # kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
            # yhat = np.convolve(res, kernel, mode='same')
            # if csv_index in (442, 717, 882, 937, 969, 1082, 1113, 1132, 1142): #yhat[20] < 0.1:
            #     plt.plot(yhat, label=csv_index)
            ##
        # plt.title("my mvmnt onset as compared to trialTimes 7")
        # plt.xlabel("my MT")
        # plt.ylabel("TrialTimes 7")
        # plt.show()
        if aggregate:
            avg = temp_data.median()
            plt.plot(avg, label='{0} mean'.format(self.analysis_func.__name__))
            plt.legend()
        else:
            for i in range(len(temp_data)):
                plt.plot(temp_data.iloc[i], label=i)
        plt.title(title)
        plt.legend()
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
            ax.scatter(x, -y, z, color=color_dict[target])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        plt.title(title)
        plt.show()

    def plot_2d_trajectory_with_turning_point(self, relevant_trials, body_part, title="", aggregate=False, is_hfs=False):
        color_dict = {1: 'b', 2: 'r', 3: 'g', 4: 'c', 5: 'm', 6: 'y', 7: 'yellow', 8: 'brown'}
        turn_time = []
        for csv_index in relevant_trials['csvNum']:
            if not aggregate: fig = plt.figure()
            try:
                trial_data = self.process_data(csv_index, from_jump=True)
                projection = two_d(trial_data, self.body_part, plot=False)
                x = projection[:, 0] - projection[:, 0][0]
                y = -1 * (projection[:, 1] - projection[:, 1][0])
            except IndexError as err:
                print("trial {trial_num} threw an exception: {error}".format(trial_num=csv_index, error=err))
                continue
            init_target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['target'])[0]
            update_target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['update'])[0]
            update_target = update_target if update_target is not None else init_target

            # todo play with this
            tolerance = 0.2 # determines how simplified the path will be. The larger the tolerance, the straighter the line
            min_angle = np.pi * 0.3  # 0.22
            points = np.vstack((x, y)).T

            # Use the Ramer-Douglas-Peucker algorithm to simplify the path
            # http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
            # Python implementation: https://github.com/sebleier/RDP/
            simplified = np.array(rdp.rdp(points.tolist(), tolerance))

            sx, sy = simplified.T

            # compute the direction vectors on the simplified curve
            directions = np.diff(simplified, axis=0)
            theta = angle(directions)
            # Select the index of the points with the greatest theta
            # Large theta is associated with greatest change in direction.a
            idx = np.where(theta > min_angle)[0] + 1

            # find first turning point
            try:
                if sx.size > 0:
                    norm_proj = np.stack((x, y), axis=1)

                    ind = np.where(norm_proj == norm_proj[
                        np.where((norm_proj[:, 0] == sx[idx][0]) * (norm_proj[:, 1] == sy[idx][0]))])[0][0]
                    turn_time.append(frames_to_ms(ind))
                    # self.trial_data.loc[self.trial_data['csvNum'] == csv_index, 'projectionX'] = list(x)
                    # self.trial_data.loc[self.trial_data['csvNum'] == csv_index, 'projectionY'] = list(y) todo save projection
                    self.trial_data.loc[self.trial_data['csvNum'] == csv_index, 'updateDelay'] = frames_to_ms(ind)
            except IndexError as err:
                print("trial {trial_num} threw an exception: {error}".format(trial_num=csv_index, error=err))
                continue

            if not aggregate:
                ax = fig.add_subplot(111) #todo uncomment

                ax.plot(x, y, color=color_dict[init_target], label='original path')
                ax.plot(sx, sy, 'g--', label='simplified path')
                ax.plot(sx[idx], sy[idx], 'ro', markersize=10, label='turning points')
                # ax.invert_yaxis()
                ax.legend(loc='best')

                ax.scatter(target_dict[init_target][0], target_dict[init_target][1], s=500, c=color_dict[init_target])
                ax.scatter(target_dict[update_target][0], target_dict[update_target][1], s=500, c=color_dict[update_target])
                ax.scatter(0, 0, s=400, c='k')
                plt.xlim(-4, 4)
                plt.ylim(-4, 4)
                plt.xlabel('x-axis')
                plt.ylabel('y-axis')
                plt.title(title + ' ' + str(frames_to_ms(ind)))
                plt.show()
        # plt.figure()
        if aggregate:
            if is_hfs:
                self.res_dict[self.date] = {'mean': np.mean(np.array(turn_time)), 'std': (np.std(np.array(turn_time), ddof=1) / np.sqrt(np.size(np.array(turn_time)))), 'data': turn_time}

            else:
                self.res_dict[str(self.date) + 'no hfs'] = {'mean': np.mean(np.array(turn_time)), 'std': (np.std(np.array(turn_time), ddof=1) / np.sqrt(np.size(np.array(turn_time)))), 'data': turn_time}

            plt.show()
        # print(str(self.date) + " NO hfs" + str(np.mean(np.array(turn_time))) + " " + str(np.std(np.array(turn_time))))
        # plt.bar(0, np.mean(np.array(turn_time)), yerr=np.std(np.array(turn_time)), align='center', alpha=0.5, ecolor='black', capsize=10)
        # plt.title(str(self.date) + " NO hfs")
        # plt.show()



    def plot_2d_trajectory(self, relevant_trials, body_part, title="", aggregate=None):
        color_dict = {1: 'b', 2: 'r', 3: 'g', 4: 'c', 5: 'm', 6: 'y', 7: 'yellow', 8: 'brown'}
        for i in range(1, 9):
            plt.scatter(target_dict[i][0], target_dict[i][1], s=500, c=color_dict[i])
        plt.scatter(0, 0, s=400, c='k')

        for csv_index in relevant_trials['csvNum']:
            trial_data = self.process_data(csv_index)
            if trial_data.empty:
                # print(csv_index)
                continue
            projection = two_d(trial_data, self.body_part, plot=False)
            x = projection[:, 0] - projection[:, 0][0]
            y = -1 * (projection[:, 1] - projection[:, 1][0])
            init_target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['target'])[0]
            update_target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['update'])[0]
            update_target = update_target if update_target is not None else 'k'
            plt.plot(x, y, color=color_dict[init_target], label='traj')  # change here to color by init or update target

        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
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
            y = -1 * (projection[:, 1][-1] - projection[:, 1][0])
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

    def process_data(self, trial_index, from_jump=False):
        data = pd.read_csv(self.data_path.format(trial_num=trial_index), header=0,
                           usecols=helpers.camera_angle[self.angle][self.body_part])
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
        start_pos = max(0, gopos)
        endpos = min(end_diff.index(min(end_diff)), start_pos + 200)
        movement_onset_pos = self.find_mvmnt_pos(partial_data[start_pos:endpos])
        in_periph = list(self.trial_data.loc[self.trial_data['csvNum'] == trial_index]['TrialTimes'])[0][7]
        in_periph_diff = [abs(a - in_periph) for a in vidticks]
        in_periph_pos = in_periph_diff.index(min(in_periph_diff))
        if check_delta(partial_data[start_pos:in_periph_pos], self.body_part) < 0.15:
            if (self.is_update_trial(trial_index)):
                if self.is_HFS(trial_index):
                    self.a1 += 1
                else:
                    self.a2 += 1
            else:
                if self.is_HFS(trial_index):
                    self.b1 += 1
                else:
                    self.b2 += 1
            return partial_data[:0]
        if True:
            trial_times_7 = list(self.trial_data.loc[self.trial_data['csvNum'] == trial_index]['TrialTimes'])[0][6]
            seven_diff = [abs(a - trial_times_7) for a in vidticks]
            seven_pos = seven_diff.index(min(seven_diff))  # - start_pos
        # plt.scatter(movement_onset_pos, seven_pos - gopos)
        # plt.plot(np.linspace(0, 120), np.linspace(0, 120))
        # plt.title("my mvmnt onset as compared to trialTimes 7")
        # plt.xlabel("my MT")
        # plt.ylabel("TrialTimes 7")
        # # plt.show()
        if False:  # np.random.binomial(1, 1): # abs(movement_onset_pos - seven_pos) > 20: :#trial_index in (119, 139, 372, 382, 387, 637):
            kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
            vel = calculate_velocity(partial_data[start_pos:], self.body_part)
            vel = vel / np.nanmax(vel)
            yhat = np.convolve(vel, kernel, mode='same')
            plt.plot(yhat, label=trial_index)
            plt.legend()
            plt.scatter(0, yhat[0], marker='o')
            trial_times_7 = list(self.trial_data.loc[self.trial_data['csvNum'] == trial_index]['TrialTimes'])[0][6]
            seven_diff = [abs(a - trial_times_7) for a in vidticks]
            seven_pos = seven_diff.index(min(seven_diff)) - start_pos
            plt.scatter(seven_pos, yhat[seven_pos], marker='x')
            plt.scatter(movement_onset_pos, yhat[movement_onset_pos], marker='v')
            plt.scatter(len(yhat), yhat[-1], marker='^')
            plt.show()

        if from_jump:
            jump = list(self.trial_data.loc[self.trial_data['csvNum'] == trial_index]['TrialTimes'])[0][5]
            jump_diff = [abs(a - jump) for a in vidticks]
            jump_pos = jump_diff.index(min(jump_diff))
            start_pos = jump_pos
            movement_onset_pos = 0
        return partial_data[start_pos + movement_onset_pos:endpos]

    def find_mvmnt_pos(self, data):
        mvmnt_pos = 0
        if not data.empty:
            vel = calculate_velocity(data, self.body_part)
            kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
            vel = vel - (np.nanmean(vel[:5]))
            vel = vel / np.nanmax(vel)
            vel = list(np.convolve(vel, kernel, mode='same'))
            max_vel = max([a for a in vel if not np.isnan(a)], default=np.nan)
            if not np.isnan(max_vel):
                max_pos = vel.index(max_vel)
                if max_pos > 1:
                    diff = [(a < (max_vel / 10)) for a in vel[:max_pos + 1]]
                    try:
                        mvmnt_pos = diff[::-1].index(True)
                        mvmnt_pos = len(diff) - mvmnt_pos - 1
                    except:
                        mvmnt_pos = 0
                        return mvmnt_pos
                while not self.mvmnt_local_min(mvmnt_pos, vel):
                    pass
                    diff = diff[mvmnt_pos + 1:]
                    try:
                        offset = diff[::-1].index(min([a for a in diff if not np.isnan(a)]))
                        offset = len(diff) - offset - 1
                        mvmnt_pos += offset
                    except:
                        mvmnt_pos = mvmnt_pos
                        break
        return mvmnt_pos

    def mvmnt_local_min(self, position, velocity_vector):
        return velocity_vector[position] < (0.5 * max(velocity_vector[position:position + ms_to_frames(50)]))

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

    def run_analysis(self, aggregate=False, hfs=True, update=True, init_target=None, update_target=None, title=None):
        relevant_trials = self.meets_requirements(hfs, update, init_target, update_target)
        if title is None:
            title = "{angle} {analysis}".format(angle=self.angle, analysis=self.analysis_func.__name__) + \
                    (", HFS" if hfs else "") + (", update" if update else "") + \
                    (("\ninit_target: " + str(init_target)) if init_target else "") + \
                    (("\nupdate_target: " + str(update_target)) if update_target else "")
        self.analysis_func(relevant_trials, self.body_part, title=title, aggregate=aggregate, is_hfs=hfs)

    def is_update_trial(self, csv_index):
        if math.isnan(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['update']) or \
                len(set(self.trial_data[['target', 'update']].loc[self.trial_data['csvNum'] == csv_index])) == 1:
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
        plt.figure()
        delta_list = []
        for csv_index in all_data['csvNum']:
            trial_data = self.process_data(csv_index)
            delta_list.append(check_delta(trial_data, self.body_part))
        pd.Series(delta_list).plot.kde()
        plt.scatter(0.1, 0.5)
        plt.show()

    def calc_mean_reaction_time(self, all_data):
        """
        helper function that finds distribution of differences between movement onset velocity and peak velocity
        :param all_data:
        :return: we should cut off the ones that are less than 0.2
        """
        delta_list = []
        for csv_index in all_data['csvNum']:
            trial_data, start_pos, mvmnt_pos = self.process_data(csv_index)
            delta_list.append(int(mvmnt_pos))
        pd.Series(delta_list).plot.kde()
        plt.show()
        print('buffer')

    def find_target_center(self, relevant_trials):
        target_dict = {1: [[], []], 2: [[], []], 3: [[], []], 4: [[], []], 5: [[], []], 6: [[], []], 7: [[], []],
                       8: [[], []]}
        for csv_index in relevant_trials['csvNum']:
            trial_data = self.process_data(csv_index)
            if trial_data.empty:
                continue
            projection = two_d(trial_data, self.body_part, plot=False)
            x = projection[:, 0][-1] - projection[:, 0][0]
            y = projection[:, 1][-1] - projection[:, 1][0]
            target = list(self.trial_data.loc[self.trial_data['csvNum'] == csv_index]['target'])[0]
            target_dict[target][0].append(x)
            target_dict[target][1].append(y)
        for k in range(1, 9):
            print("target " + str(k) + ": x=" + str(statistics.mean(target_dict[k][0])) + ", y=" + str(
                statistics.mean(target_dict[k][1])))


if __name__ == "__main__":
    DATA_PATH = r'Z:\elianna.rosenschein\n{date}\DLC_Analysis\nana-trial{trial_num}_DLC_3D_{angle}.csv'
    ED_PATH = r'Z:\elianna.rosenschein\n{date}\EDfiles\n{trial_num}ee.{file_num}.mat'
    INFO_FILE = r'Z:\elianna.rosenschein\n{date}\Info\n{date}_param.mat'
    INDEX_FILE = r'Z:\elianna.rosenschein\alignment_indices_n{date}.mat'  # Exported from Nirvik's Matlab code (I, J)
    VIDEO_INFO_FILE = r'Z:\elianna.rosenschein\vidInfo_n{date}.mat'  # Exported from Nirvik's Matlab code (vidinfo)

    BODY_PART = 'finger'  # body part that we want to analyze
    ANGLE = 'SIDE'
    DATE = '281221'  #'270122'  # '301221' #
    ANALYSIS_FUNC = "turning_point"  #"plot_velocity"  #"two_d"  #

    pair_test_data = pd.DataFrame({'HFS': pd.Series([], dtype=str), 'noHFS': pd.Series([], dtype=str)})
    date_list = ['281221', '270122', '301221', '240222', '291221']
    for DATE in date_list:
        data_path = DATA_PATH.format(date=DATE, trial_num='{trial_num}', angle=ANGLE)
        ed_path = ED_PATH.format(date=DATE, trial_num='{trial_num}', file_num='{file_num}')
        info_file = INFO_FILE.format(date=DATE)
        index_file = INDEX_FILE.format(date=DATE)
        video_info_file = VIDEO_INFO_FILE.format(date=DATE)

        day = Day(DATE, BODY_PART, ANGLE, ANALYSIS_FUNC, data_path, ed_path, info_file, index_file, video_info_file)
        day.preprocess()
        # day.calc_mean_delta(day.trial_data)
        for bool in [True, False]:
            day.run_analysis(aggregate=True, hfs=bool, update=True, init_target=None, update_target=None)
        fig, ax = plt.subplots()
        materials = ['No HFS', 'HFS']
        ax.set_xticklabels(materials)
        CTEs = [day.res_dict[str(DATE) + 'no hfs']['mean'], day.res_dict[DATE]['mean']]
        error = [day.res_dict[str(DATE) + 'no hfs']['std'], day.res_dict[DATE]['std']]
        ax.bar((0, 1), CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        x_pos = np.arange(len(materials))
        ax.set_xticks(x_pos)
        _, pval = stats.ttest_ind(day.res_dict[DATE]['data'], day.res_dict[str(DATE) + 'no hfs']['data'])
        plt.title(str(day.date) + " comparison between turn time; pval: " + str(round(pval, 2)))
        ax.set_xticklabels(materials)
        plt.show()
        ax = pd.Series(day.res_dict[str(DATE) + 'no hfs']['data']).plot.kde(label="control")
        ax = pd.Series(day.res_dict[DATE]['data']).plot.kde(label="hfs")
        plt.title("distribution of reaction time " + str(DATE))
        plt.legend()
        plt.show()

        temp = {'HFS': day.res_dict[DATE]['mean'],
                'noHFS': day.res_dict[str(DATE) + 'no hfs']['mean']}
        pair_test_data = pd.concat([pair_test_data, pd.DataFrame([temp])], ignore_index=True)
    print("paired ttest results: ")
    _, pairedpval = stats.ttest_rel(pair_test_data['HFS'], pair_test_data['noHFS'])
    plt.figure()
    for index, row in pair_test_data.iterrows():
        a = row['noHFS']
        b = row['HFS']
        if a != 0 or b!= 0:
            plt.plot([[1, a], [2, b]], c='b')
    plt.title('paired ttest; pvalue=' + str(round(pairedpval, 7)))
    plt.ylim((300, 550))
    plt.show()

    # print("thrown away:")
    # print('HFS+Update: ' + str(100 * day.a1 / day.trial_data[day.trial_data.apply(lambda row: (day.is_HFS(row['csvNum']) and day.is_update_trial(row['csvNum'])), axis=1)].shape[0]) + "%")
    # print('HFS+ no update: ' + str(100 * day.a2 / day.trial_data[
    #     day.trial_data.apply(lambda row: (day.is_HFS(row['csvNum']) and (not day.is_update_trial(row['csvNum']))),
    #                          axis=1)].shape[0]) + "%")
    # print('No HFS+update: ' + str(100 * day.b1 / day.trial_data[
    #     day.trial_data.apply(lambda row: (not day.is_HFS(row['csvNum']) and (day.is_update_trial(row['csvNum']))),
    #                          axis=1)].shape[0]) + "%")
    # print('No HFS+ No update: ' + str(100 * day.b2 / day.trial_data[
    #     day.trial_data.apply(lambda row: (not day.is_HFS(row['csvNum']) and (not day.is_update_trial(row['csvNum']))),
    #                          axis=1)].shape[0]) + "%")

# TODO JOINT ANGLES
# todo change projection anchors for new days
# todo update time checks
# todo fix column choosing for dates with more columns
