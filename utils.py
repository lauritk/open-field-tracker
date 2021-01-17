import math
import ast
import re
import numpy as np
import pandas as pd
import numexpr as ne
from configparser import ConfigParser


def sum_distance_m(data):
    return sum_distance_cm(data) / 100


def sum_distance_cm(data):
    return np.sum(data)


def speed_km_h(data, vid_duration, total_frames):
    return speed_m_s(data, vid_duration, total_frames) * 3600 / 1000


def speed_m_s(data, vid_duration, total_frames):
    return speed_cm_s(data, vid_duration, total_frames) / 100


def speed_cm_s(data, vid_duration, total_frames):
    """Data should be cm distances"""
    frame_time = vid_duration / total_frames
    return data / frame_time


def data_px_to_cm(data, corners, parameters):
    out = data.copy()
    width = corners.corners[1].coord[0] - corners.corners[0].coord[0]
    height = corners.corners[2].coord[1] - corners.corners[0].coord[1]
    pixels_in_cm_w = width / parameters['field_width']
    pixels_in_cm_h = height / parameters['field_height']
    out[:, 0] = out[:, 0] / pixels_in_cm_w
    out[:, 1] = out[:, 1] / pixels_in_cm_h
    return out


# def normalize_data(data):
#     out = data.copy()
#     out[:, 0] = (data[:, 0] - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
#     out[:, 1] = (data[:, 1] - np.min(data[:, 1])) / (np.max(data[:, 1]) - np.min(data[:, 1]))
#     return out


def correct_coordinates(data, likelihood, limit):
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    mask = likelihood < limit
    idx = np.where(~mask, np.arange(mask.shape[0]), np.nan)
    data[mask, :] = np.nan
    nans, index = np.isnan(data)[:, 0], lambda z: z.nonzero()[0]
    data[nans, 0] = np.interp(index(nans), index(~nans), data[~nans, 0])
    data[nans, 1] = np.interp(index(nans), index(~nans), data[~nans, 1])
    return data


def distance(a, b):
    """Euclidean distance"""
    return math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2))


def moved_distance(pt):
    dummy = np.tile(np.array([0, 0]), (len(pt), 1))
    distances = sqrt_sum(pt.T, dummy.T)
    differences = np.ediff1d(distances)
    return np.abs(differences)




# More efficient distances for large matrices

# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def sqrt_einsum(a, b):
    a_min_b = a - b
    return np.sqrt(np.einsum('ij,ij->i', a_min_b, a_min_b))

# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def sqrt_sum(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=0))

# https://stackoverflow.com/questions/40319433/numpy-find-the-euclidean-distance-between-two-3-d-arrays/40319768
def numexpr_based_with_slicing(a, b):
    a0 = a[..., 0]
    a1 = a[..., 1]
    b0 = b[..., 0]
    b1 = b[..., 1]
    return ne.evaluate('sqrt((a0-b0)**2 + (a1-b1)**2)')

# https://stackoverflow.com/questions/40319433/numpy-find-the-euclidean-distance-between-two-3-d-arrays/40319768
def numexpr_based(a, b):
    return np.sqrt(ne.evaluate('sum((a-b)**2,1)'))


def angle(a, b, c):
    """0-180 angle between two lines (a to b, a to c)"""
    # See: https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
    angle1 = math.atan2(b[1] - a[1], b[0] - a[0])
    angle2 = math.atan2(c[1] - a[1], c[0] - a[0])
    # Degree relative to lines and convert to positive.
    deg = math.degrees(angle1 - angle2) % 360
    # Convert degree to 0-180.
    return min(360 - deg, deg)


def angle2(a, b, c):
    """0-180 angle between two lines (a to b, a to c). Numpy matrix implementation."""
    ab = np.subtract(b, a)
    ac = np.subtract(c, a)
    a1 = np.arctan2(ab[:, [1]], ab[:, [0]])
    a2 = np.arctan2(ac[:, [1]], ac[:, [0]])
    deg = np.degrees(a1 - a2) % 360
    # return np.min((360 - deg, deg), axis=0)
    return deg


def load_parameters(filename):
    config = ConfigParser()
    config.read(filename)
    parameters = dict()
    for key, val in config.items(section='Configurations'):
        if key == 'video_file' or key == 'field_shape' or key == 'experiment' or key == 'experiment_phase':
            parameters[key] = val
        elif key == 'transformation_matrix':
            p = re.sub(r'\[|\]{2}|\n', '', val).split(']')
            values = [x.split() for x in p]
            parameters[key] = np.array(values, dtype=np.float64)
        else:
            parameters[key] = ast.literal_eval(val)
            
    return parameters


def save_parameters(filename, parameters):
    config = ConfigParser()
    config['Configurations'] = parameters
    with open(filename, 'w') as configfile:
        config.write(configfile)


def load_deeplab_csv(filename):
    # 'float_precision='round_trip'' fixes float precision to match h5-file
    return pd.read_csv(filename, index_col=0, header=[0, 1, 2],
                       float_precision='round_trip')


def load_deeplab_h5(filename):
    return pd.read_hdf(filename)


def save_deeplab_df_to_h5(filename, data):
    data.to_hdf(filename, 'df_with_missing', format='table', mode='w')

def visits_idx(visits, visit_gap, visit_length):
    x = visits.copy()
    start_idx = np.empty(len(x) - 1, dtype=bool)
    first_true = False
    if x[0]:
        x[0] = False
        first_true = True

    start_idx[0] = True
    np.not_equal(x[1:], x[:-1], out=start_idx[:])
    starts = np.nonzero(start_idx)[0]
    starts[::2] += 1
    if first_true:
        starts[0] = 0
        x[0] = True

    if starts.size % 2 != 0:
        starts = np.append(starts, starts[-1])

    seq = np.reshape(starts, (-1, 2))
    exp_start = seq[:, [0]]
    exp_end = seq[:, [1]]
    exp_dist = exp_end - exp_start + 1
    gap = np.roll(exp_start, -1) - exp_end + 1
    gap_idx = np.where(gap[:-1] < visit_gap)[0]
    gap_len_idx = np.where(exp_dist < visit_length)[0]
    exp_end[gap_idx] = exp_end[gap_idx + 1]
    exp_start = np.delete(exp_start, np.concatenate((gap_idx + 1, gap_len_idx)))
    exp_end = np.delete(exp_end, np.concatenate((gap_idx + 1, gap_len_idx)))
    return (list(exp_start), list(exp_end))