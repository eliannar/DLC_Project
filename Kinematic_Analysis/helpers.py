import numpy as np
import pandas as pd

PERCENTAGE = 1
KERNEL_SIZE = 40
FS = 120

# dictionary that links the body parts from the analysis to the colomns names
body_cols_side = {'finger': ['DLC_3D_SIDE', 'DLC_3D_SIDE.1', 'DLC_3D_SIDE.2'],
                  'wrist': ['DLC_3D_SIDE.3', 'DLC_3D_SIDE.4', 'DLC_3D_SIDE.5'],
                  'elbow': ['DLC_3D_SIDE.6', 'DLC_3D_SIDE.7', 'DLC_3D_SIDE.8']}
body_cols_top = {'finger': ['DLC_3D_TOP', 'DLC_3D_TOP.1', 'DLC_3D_TOP.2'],
                 'wrist': ['DLC_3D_TOP.3', 'DLC_3D_TOP.4', 'DLC_3D_TOP.5'],
                 'elbow': ['DLC_3D_TOP.6', 'DLC_3D_TOP.7', 'DLC_3D_TOP.8']}

camera_angle = {'SIDE': body_cols_side, 'TOP': body_cols_top}

def angle(dir):
    """
    Returns the angles between vectors.

    Parameters:
    dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    The return value is a 1D-array of values of shape (N-1,), with each value
    between 0 and pi.

    0 implies the vectors point in the same direction
    pi/2 implies the vectors are orthogonal
    pi implies the vectors point in opposite directions
    """
    dir2 = dir[1:]
    dir1 = dir[:-1]
    return np.arccos((dir1 * dir2).sum(axis=1) / (
        np.sqrt((dir1 ** 2).sum(axis=1) * (dir2 ** 2).sum(axis=1))))


def check_delta(trial_data, body_part):
    if trial_data.empty or trial_data.shape[0] < 2:
        return 0
    kernel = np.ones(KERNEL_SIZE) / KERNEL_SIZE
    vel = calculate_velocity(trial_data, body_part)
    vel = vel / np.nanmax(vel)
    yhat = np.convolve(vel, kernel, mode='same')
    yhat = yhat[~np.isnan(yhat)]
    if len(yhat > 0):
        return max(yhat) - yhat[0]
    return 0


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

def frames_to_ms(frame):
    ms = frame * 1000 / FS
    return round(ms, 2)


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

