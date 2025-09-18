import os
import sys
import logging
import torch
import pickle
import pickle as pkl
import argparse
import numpy as np
import pandas as pd
import os.path as op
import h5py
from tqdm import tqdm
from copy import deepcopy

from sample_methods.LDATI import sample_voxel_statistical as our_sample
from sample_methods.random_even_sample import sample_voxel_baseline as baseline_sample
from sample_methods.pure_slope_sample import sample_voxel_statistical as slope_sample

sys.path.append(op.abspath('../..'))
from scripts.utils.utils import tic_toc, Timer
from scripts.utils.events_utils import gen_discretized_event_volume

def ts_diff_metric(event_gt, event_pred, search_range = 0, fps=30):
    """ Compute the temporal difference metric between GT and predicted event.
    Args:
        event_gt: GT event grid. Shape: (N, 4)
        event_pred: predicted event grid. Shape: (N, 4)
        search_range: search range for each event. Default: 0
    Returns:
        temporal difference metric
    """

    total_diff = 0
    gt_event_count = 0
    overflow = 0

    # If the polarity of event_gt has -1, set it to 0
    gt_ps = event_gt['polarity']
    gt_ps[gt_ps == -1] = 0
    event_gt['polarity'] = gt_ps
    
    # Create a 3D array to store all the events in each voxel
    all_events = []
    for W in range(346):
        all_events.append([])
        for H in range(260):
            all_events[-1].append([])
            for P in range(2):
                all_events[-1][-1].append([])
    
    # Store the events in each voxel's list
    
    for event in event_pred:
        if event[3] == 0:
            all_events[event[1]][event[2]][0].append((event[0]))
        else:
            all_events[event[1]][event[2]][1].append((event[0]))

    # Convert the list to numpy array
    for W in range(346):
        for H in range(260):
            for P in range(2):
                all_events[W][H][P] = np.array(all_events[W][H][P])

    for event in event_gt:
        diff = 1e6

        # Calculate the search range (xy) for each ground truth event
        a_low = max(event[1] - search_range, 0)
        a_high = min(event[1] + search_range+1, 346)
        b_low = max(event[2] - search_range, 0)
        b_high = min(event[2] + search_range+1, 260)

        # Search for the nearest event in the search range
        for a in range(a_low, a_high):
            for b in range(b_low, b_high):
                if len(all_events[a][b][event[3]]) == 0:
                    continue
                diff = min(diff, np.min(np.abs(all_events[a][b][event[3]] - (event[0]))))

        # Record the overflow situation
        if diff > 1e6/fps/10*3:
            diff = 1e6/fps/10*3
            overflow += 1
        
        total_diff += diff
        gt_event_count += 1
    logger.debug(f"gt_event_count: {gt_event_count}")
    logger.debug(f"avg_diff: {total_diff / gt_event_count}")
    logger.debug(f"overflow: {overflow}")
    return np.array([total_diff / gt_event_count, overflow])

# @tic_toc
def run_metric_for_data(file_path, v2e_pred_path = '', esim_pred_path='', eventgan_voxel_path='',args=None):
    """ Run the metric for a single data file.
    Args:
        file_path: path to the data file
        args: arguments
    Returns:
        metric: metric for the data file
    """
    data = pkl.load(open(file_path, 'rb'))
    v2e_es = deepcopy(h5py.File(v2e_pred_path, 'r')['events'][:])
    esim_es = pickle.load(open(esim_pred_path, 'rb'))

    metric = {k: None for k in args.evaluate_on}
    info = {k: 0 for k in args.evaluate_on}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    index=0

    #! Correct the 10x timestamp Bug in the previous data processing. Please comment this part if you are using the new data.
    data['timestamps'] = data['timestamps'] // 10
    for i in range(len(data['events'])):
        data['events'][i]['timestamp'] = (data['events'][i]['timestamp'] - data['events'][i]['timestamp'][0]) // 10

    esim_es['t'] = esim_es['t'] / 1e3
    time_step = (data['timestamps'][-1] - data['timestamps'][0])/16

    esim_events_pieces = []
    for i in range(16):
        if i == 15:
            esim_events_piece = esim_es[np.searchsorted(esim_es['t'], i*time_step):]
        else:
            esim_events_piece = esim_es[np.searchsorted(esim_es['t'], i*time_step):np.searchsorted(esim_es['t'], (i+1)*time_step)]
        esim_events_pieces.append(esim_events_piece)

    v2e_events_pieces = []
    for i in range(16):
        # use np.searchsorted to find the index of the first event in the next time step
        if i == 15:
            v2e_events_piece = v2e_es[np.searchsorted(v2e_es[:,0], i*time_step):]
        else:
            v2e_events_piece = v2e_es[np.searchsorted(v2e_es[:,0], i*time_step):np.searchsorted(v2e_es[:,0], (i+1)*time_step)]
        v2e_events_pieces.append(v2e_events_piece)

    with open(eventgan_voxel_path, 'rb') as f:
        eventgan_recorder = pkl.load(f)

    for idx, event_batch in enumerate(data['events']):
        frame_time_diff = (data['timestamps'][index+1]-data['timestamps'][index])
        fps = 1e6 / frame_time_diff

        if 'v2e' in args.evaluate_on:
            pred = v2e_events_pieces[idx]#.to(device)
        
            if metric['v2e'] is None:
                metric['v2e'] = ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            else:
                metric['v2e'] += ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            info['v2e'] += len(pred) / len(event_batch)

        if 'esim' in args.evaluate_on:
            pred = esim_events_pieces[idx]#.to(device)
            if metric['esim'] is None:
                metric['esim'] = ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            else:
                metric['esim'] += ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            info['esim'] += len(pred) / len(event_batch)

        if 'eventgan' in args.evaluate_on:
            eventgan_voxel = torch.tensor(eventgan_recorder['gen_event_volume'][idx]).reshape(1, 2, 10, 260, 346).to(device)
            pred = baseline_sample(eventgan_voxel, fps=fps, random=True)[0]
            if metric['eventgan'] is None:
                metric['eventgan'] = ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            else:
                metric['eventgan'] += ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            info['eventgan'] += len(pred) / len(event_batch)
        break

    for k in metric.keys():
        metric[k] = np.append(metric[k], info[k])
    return {k: v for k, v in metric.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_path', type=str, default=r"/tsukimi/datasets/MVSEC/data_paths_new.pkl", help='path to the data info file')
    parser.add_argument('--pred_root', type=str, default=r"/tsukimi/v2ce-project/best_model_log/recorder", help='path to the data root')
    parser.add_argument('--data_root', type=str, default=r"/tsukimi/datasets/MVSEC/event_chunks_10t", help='path to the data root')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--search_range', type=int, default=0, help='search range for each GT event')
    parser.add_argument('--bin_num', type=int, default=10, help='number of bins for the event volume')
    parser.add_argument('--log_level', type=str, default='INFO', help='logging level')
    parser.add_argument('--file_num', type=int, default=1, help='number of files to run the metric. If set to -1, run the metric for all files')
    parser.add_argument('--file_start', type=int, default=0, help='the index of the first file to run the metric')
    parser.add_argument('--evaluate_on', default=['v2e', 'eventgan', 'esim'], nargs='*', help="Loss type.")
    parser.add_argument('-a', '--additional_events_strategy', default='slope', choices=['random', 'slope', 'none'], help="Type2 events processing method.")
    parser.add_argument('-p', '--pooling_type', default='none', choices=['none', 'weighted', 'avg'], help="Pooling type of the y_pool used to calculate slope.")
    parser.add_argument('-b', '--bidirectional', action='store_true', help="Whether to use bidirectional y_relocate inference.")
    parser.add_argument('--exp_name', type=str, default='exp', help='name of the experiment')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger('main')
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    info = pkl.load(open(args.info_path, 'rb'))

    v2ce_recorder_paths = [op.join(args.pred_root, f) for f in os.listdir(args.pred_root) if f.endswith('.pkl')]

    v2e_root = r'/tsukimi/v2ce-project/raw_events_baselines/v2e'
    esim_root = r'/tsukimi/v2ce-project/raw_events_baselines/esim'
    eventgan_root = r'/tsukimi/v2ce-project/pred_voxel_baselines/eventgan-retrained'

    result = None
    file_count = 0
    recorder = {}
    
    files_to_run = info['test'][args.file_start:args.file_start+args.file_num] if args.file_num != -1 else info['test']
    
    for path in tqdm(files_to_run):
        stem = op.basename(path).split('.')[0]
        raw_data_path = op.join(args.data_root, stem.lstrip('test-')+'.pkl')
        data_name = stem.lstrip('test-')+'.pkl'
        file_result=run_metric_for_data(raw_data_path, v2e_pred_path = op.join(v2e_root, data_name, 'events.h5'),esim_pred_path = op.join(esim_root, data_name, 'events.pkl'), eventgan_voxel_path=op.join(eventgan_root, data_name), args=args)
        logger.debug(f"file_result: {file_result}")
        recorder[stem] = file_result
        
        if result is None:
            result = file_result
        else:
            result = {k: v + file_result[k] for k, v in result.items()}
        file_count += 1

        # break
        print(result)
        # break
    result = {k: v/file_count for k, v in result.items()}
    result_df = pd.DataFrame(result, index=['Avg Error', '#Overflow', 'Pred GT Event # Ratio']).T
    # keep the first 3 digits
    result_df = result_df.map(lambda x: round(x, 3))
    # set the dtype of '#Overflow' to int
    result_df['#Overflow'] = result_df['#Overflow'].astype(int)
    
    our_dir_name = '{}-{}'.format('-'.join(args.evaluate_on), args.exp_name)
    out_dir = op.join('results', our_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    
    result_df.to_csv(op.join(out_dir, 'abbr_result.csv'))
    with open(op.join(out_dir, 'full_record.npy'), 'wb') as f:
        np.save(f, recorder)
    print(result_df)
