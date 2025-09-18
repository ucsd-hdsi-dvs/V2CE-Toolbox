""" Utils for loading events.
"""
import h5py
import torch
import importlib
import numpy as np
import pandas as pd
import os.path as op
import plotly.express as px

from numba import njit, cuda
# from dv import AedatFile
from math import floor, ceil
from tqdm import trange
from .vis_utils import *
from .utils import get_new_path, pkl_load

cam_settings = {
    "xz":  {'up': {'x': 0.029205246220929418, 'y': -0.07454904061931165, 'z': -0.9967895937136961}, 'center': {'x': 0.4152144677376415, 'y': -0.19700200366278003, 'z': 0.1318296812311048}, 'eye': {'x': -0.05808189772173178, 'y': 1.7511480688146275, 'z': -0.027738051796443258}},
    "side": {'up': {'x': 0.18044721455186086, 'y': -0.0326062061218738, 'z': -0.9830440672130688}, 'center': {'x': 0.4282785144271674, 'y': -0.17502657663951424, 'z': 0.21871482833583408}, 'eye': {'x': -1.1257276557476024, 'y': 1.3147917910060438, 'z': -0.11595318966741139}},
    "full": {'up': {'x': 0.16192919505818526, 'y': 0.014526753698953593, 'z': -0.9866954490696601}, 'center': {'x': 0.4282785144271674, 'y': -0.17502657663951424, 'z': 0.21871482833583408}, 'eye': {'x': -1.5964852994314518, 'y': 1.225895055808503, 'z': -0.09294925336730195}}
}


def extract_aedat4(path):
    """ Extract events from AEDAT4 file.
        Args:
            path: str, the path of input aedat4 file.
        Returns:
            events: pd.DataFrame, pandas data frame containing events.
    """
    AedatFile = importlib.import_module('dv').AedatFile
    with AedatFile(path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
    events = pd.DataFrame(events)[['timestamp', 'x', 'y', 'polarity']]
    events = events.rename(columns={'timestamp': 't', 'polarity': 'p'})
    return events


def load_events(path, slice=None, to_df=True, start0=False, verbose=False):
    """ Load the DVS events in .h5 or .aedat4 format.
    Args:
        path: str, input file name.
        slice: tuple/list, two elements, event stream slice start and end.
        to_df: whether turn the event stream into a pandas dataframe and return.
        start0: set the first event's timestamp to 0.
    """
    ext = op.splitext(path)[1]
    assert ext in ['.h5', '.aedat4']
    if ext == '.h5':
        f_in = h5py.File(path, 'r')
        events = f_in.get('events')[:]
    else:
        events = extract_aedat4(path)
        events = events.to_numpy()  # .astype(np.uint32)

    if verbose:
        print(events.shape)
    if slice is not None:
        events = events[slice[0]:slice[1]]
    if start0:
        events[:, 0] -= events[0, 0]  # Set the first event timestamp to 0
        # events[:,2] = 260-events[:,2] # Y originally is upside down
    if to_df:
        events = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])
    return events



def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def calc_floor_ceil_delta(x): 
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]

def create_update(x, y, t, dt, p, vol_size):
    assert (x>=0).byte().all() and (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all() and (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() and (t<vol_size[0] // 2).byte().all()

    # First half of volume is positive events, second half is negative events
    vol_mul = torch.where(p < 0,
                          torch.ones(p.shape, dtype=torch.long) * vol_size[0] // 2,
                          torch.zeros(p.shape, dtype=torch.long))

    inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals

def gen_discretized_event_volume(events, vol_size):
    # volume is [timestamp, x, y, polarity]
    volume = torch.zeros(vol_size, dtype=torch.float)

    x = torch.tensor(events['x'].copy(), dtype=torch.long)
    y = torch.tensor(events['y'].copy(), dtype=torch.long)
    t = torch.tensor(events['timestamp'].copy())
    p = torch.tensor(events['polarity'].copy())
    p[p == 0] = -1  # polarity should be +1 / -1

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min))
    t_scaled = torch.clamp(t_scaled, 0, vol_size[0] // 2-1)

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     p,
                                     vol_size)
        
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     p,
                                     vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)

    return volume

def gen_discretized_event_volume_from_tensor(events, vol_size, t_min=None, t_max=None):
    # volume is [timestamp, x, y, polarity]
    volume = torch.zeros(vol_size, dtype=torch.float)

    # x = torch.tensor(events[:,1], dtype=torch.long)
    # y = torch.tensor(events[:,2], dtype=torch.long)
    # t = torch.tensor(events[:,0])
    # p = torch.tensor(events[:,3])

    x = events[:,1]
    y = events[:,2]
    t = events[:,0]
    p = events[:,3]    

    p[p == 0] = -1  # polarity should be +1 / -1

    t_min = t.min() if t_min is None else t_min
    t_max = t.max() if t_max is None else t_max
    t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min))
    t_scaled = torch.clamp(t_scaled, 0, vol_size[0] // 2-1)

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     p,
                                     vol_size)
        
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     p,
                                     vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)

    return volume

def structured_events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy structured array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(len(events.dtype.names) == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((2,num_bins, height, width), np.float32)

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1]['timestamp']
    first_stamp = events[0]['timestamp']
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    ts = (num_bins - 1) * (events['timestamp'] - first_stamp) / deltaT # [0, num_bins - 1]
    xs = events['x'].astype(int)
    ys = events['y'].astype(int)
    pols = events['polarity']
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int) # Floor of the normalized timestamps
    dts = ts - tis # Fractional part of the normalized timestamps
    vals_left = pols * (1.0 - dts) 
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid[0,:,:,:].ravel(), xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid[1,:,:,:].ravel(), xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (2,num_bins, height, width))

    return voxel_grid

# @jit
# def structured_events_to_voxel_stat(events, num_bins, width, height):
#     event_grid = np.empty((2, num_bins, height, width), dtype=object)
#     event_grid.fill(None)
#     event_voxel = np.zeros((2, num_bins, height, width))
#     delta_t = np.ceil((events['timestamp'][-1] - events['timestamp'][0]) / num_bins).astype(int)

#     ts = events['timestamp'] - events['timestamp'][0]
#     tbs = ts // delta_t
#     ps = events['polarity']
#     xs = events['x']
#     ys = events['y']
#     ps[ps == -1] = 0

#     for i in range(len(events)):
#         t = ts[i]
#         tb = tbs[i]
#         p = ps[i]
#         x = xs[i]
#         y = ys[i]
#         # print(x, y, t, p)
#         if event_grid[p, tb, y, x] is None:
#             event_grid[p, tb, y, x] = [t]
#         else:
#             event_grid[p, tb, y, x].append(t)
#         event_voxel[p, tb, y, x] += 1

#     event_grid_mean = np.array([[[[np.mean(event_grid[p, tb, y, x]) if event_grid[p, tb, y, x] is not None else 0
#                                 for x in range(width)]
#                                 for y in range(height)]
#                                 for tb in range(num_bins)]
#                                 for p in range(2)])

#     event_grid_std = np.array([[[[np.std(event_grid[p, tb, y, x]) if event_grid[p, tb, y, x] is not None else 0
#                                 for x in range(width)]
#                                 for y in range(height)]
#                                 for tb in range(num_bins)]
#                             for p in range(2)])
#     return event_voxel, event_grid_mean, event_grid_std
# @cuda.jit
# @njit(parallel=True)
# def structured_events_to_voxel_stat(events, num_bins, width, height):
#     delta_t = int(np.ceil((events['timestamp'][-1] - events['timestamp'][0]) / num_bins))

#     ts = events['timestamp'] - events['timestamp'][0]
#     tbs = ts // delta_t
#     ps = events['polarity']
#     xs = events['x']
#     ys = events['y']
#     ps[ps == -1] = 0

#     event_voxel = np.zeros((2, num_bins, height, width))
#     event_grid_sum = np.zeros((2, num_bins, height, width))
#     event_grid_count = np.zeros((2, num_bins, height, width))
#     event_grid_sum_sq = np.zeros((2, num_bins, height, width))

#     for i in range(len(events)):
#         tb, p, x, y, t = tbs[i], ps[i], xs[i], ys[i], ts[i]
#         t = t % delta_t
#         event_voxel[p, tb, y, x] += 1
#         event_grid_sum[p, tb, y, x] += t
#         event_grid_count[p, tb, y, x] += 1
#         event_grid_sum_sq[p, tb, y, x] += t**2

#     event_grid_mean = event_grid_sum / np.maximum(event_grid_count, 1)
#     event_grid_var = (event_grid_sum_sq - (event_grid_sum ** 2) / np.maximum(event_grid_count, 1)) / np.maximum(event_grid_count - 1, 1)
#     event_grid_std = np.sqrt(event_grid_var)

#     return event_voxel, event_grid_mean, event_grid_std

# @njit(parallel=True)
def structured_events_to_voxel_stat(events, num_bins, width, height):
    delta_t = int(np.ceil((events['timestamp'][-1] - events['timestamp'][0]) / num_bins))

    ts = events['timestamp'] - events['timestamp'][0]
    tbs = ts // delta_t
    trs = ts % delta_t
    ps = events['polarity']
    xs = events['x']
    ys = events['y']
    ps[ps == -1] = 0

    event_voxel = np.zeros((2, num_bins, height, width))
    event_grid_sum = np.zeros((2, num_bins, height, width))
    event_grid_count = np.zeros((2, num_bins, height, width))
    event_grid_sum_sq = np.zeros((2, num_bins, height, width))

    np.add.at(event_voxel, (ps, tbs, ys, xs), 1)
    np.add.at(event_grid_sum, (ps, tbs, ys, xs), trs)
    np.add.at(event_grid_count, (ps, tbs, ys, xs), 1)
    np.add.at(event_grid_sum_sq, (ps, tbs, ys, xs), trs**2)

    event_grid_mean = event_grid_sum / np.maximum(event_grid_count, 1)
    event_grid_var = (event_grid_sum_sq - (event_grid_sum ** 2) / np.maximum(event_grid_count, 1)) / np.maximum(event_grid_count - 1, 1)
    event_grid_std = np.sqrt(event_grid_var)

    return event_voxel, event_grid_mean, event_grid_std

def plot_events_3d(events: pd.DataFrame, cam_setting=None):
    """ Visualize events in 3D space.
        Used for better understanding the sample events. 
        !!! Don't input too long event streams.
        The aspect ratio for x, y, t is 1:1:2.
        Args:
            events: pd.DataFrame.
            cam_setting: camera settings used in plotly. Used to set the camera's 
                position and view angle to the correct place.
    """
    if cam_setting is None:
        cam_setting = cam_settings['full']
    fig = px.scatter_3d(events, x='x', y='t', z='y', color='p', width=1000, height=600)
    fig.update_traces(marker={'size': 1, 'opacity': 0.25})
    fig.layout.scene = {'camera': cam_setting, 'aspectratio': {'x': 1, 'y': 2, 'z': 1}}
    fig.show()
    fig.write_image(get_new_path('results/events_vis.png'), scale=2)
    return fig


def accumulate_frame(events, frame_size=(260, 346)):
    """ Accumulate input events to a frame using the same way of DHP19
        Args: 
            events: np.array.
            frame_size: (H, W)
    """
    h, w = frame_size
    img = np.zeros((w, h))
    for event in events:
        timestamp, x, y, p = event
        img[x, y] += 1

    # Normalize
    sum_img = np.sum(img)
    count_img = np.sum(img > 0)
    mean_img = sum_img / count_img
    var_img = np.var(img[img > 0])
    sig_img = np.sqrt(var_img)

    if sig_img < 0.1/255:
        sig_img = 0.1/255

    num_sdevs = 3.0
    mean_grey = 0
    ranges = num_sdevs * sig_img
    half_range = 0
    range_new = 255

    def norm_img(z):
        if z == 0:
            res = mean_grey
        else:
            res = np.floor(np.clip((z+half_range) * range_new / ranges, 0, range_new))
        return res

    norm_img = np.vectorize(norm_img)
    img = norm_img(img)
    return img.T


def process_data_case(exp_name, camera_view, pose_name,
                      dvs_root, pose_root, camera_root,
                      out_root, events_per_frame=7500):
    """ Process the camera, pose, and events to constant event count representation.
        Args:
            exp_name: case name.
            camera_view: camera view file.
            pose_name: pose name.
            dvs_root: events root folder.
            pose_root: poses root folder.
            camera_root: camera root folder.
            out_root: output folder.
            events_per_frame: fixed event count number.
    """
    # Set paths
    out_root_frames = op.join(out_root, 'frames')
    out_root_labels = op.join(out_root, 'labels')
    dvs_path = op.join(dvs_root, exp_name, camera_view, 'events.h5')
    pose_path = op.join(pose_root, pose_name, 'motion_dict_kp13.pkl')
    camera_path = op.join(camera_root, 'front_50_camera_matrix_dvs346.pkl')

    # Load files
    dvs = h5py.File(dvs_path, 'r')
    pose = pkl_load(pose_path)['data']
    cm = pkl_load(camera_path)

    # Intrinsic Matrix
    intrinsic = np.pad(cm['K'], ((0, 0), (0, 1)), 'constant')
    # Extrinsic Matrix
    extrinsic = cm['RT']

    events = dvs['events']
    events_num = len(events)

    packet_num = events_num // events_per_frame
    # Generate files
    for event_idx in trange(packet_num, desc=f'{exp_name}_{camera_view}'):
        labels = {}
        out_stem = f'{exp_name}_frame_{event_idx}_{camera_view}'
        frame_path = op.join(out_root_frames, out_stem+'.npy')
        label_path = op.join(out_root_labels, out_stem+'.npz')

        start_idx = events_per_frame * event_idx
        end_idx = start_idx + events_per_frame
        packet_events = events[start_idx: end_idx]
        packet_frame = accumulate_frame(packet_events)

        start_time = packet_events[0][0]*1e-6
        end_time = packet_events[-1][0]*1e-6

        packet_poses = pose[int(floor(start_time*300)): int(ceil(end_time*300))]
        packet_poses_mean = packet_poses.mean(axis=0).T

        labels['camera'] = intrinsic
        labels['M'] = extrinsic
        labels['xyz'] = packet_poses_mean

        with open(frame_path, 'wb') as f:
            np.save(f, packet_frame)
        with open(label_path, 'wb') as f:
            np.savez(f, **labels)
    return exp_name, camera_view

# def fields_view(arr, fields):
#     dtype2 = np.dtype({name:arr.dtype.fields[name] for name in fields})
#     return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)

