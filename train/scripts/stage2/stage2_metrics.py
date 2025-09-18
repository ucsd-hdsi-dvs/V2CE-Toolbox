import os
import sys
import logging
import torch
import pickle as pkl
import argparse
import numpy as np
import pandas as pd
import os.path as op

from tqdm import tqdm
from multiprocessing import Pool, Manager, Process

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

        if diff > 1e6/fps/10*3:
            diff = 1e6/fps/10*3
            overflow += 1
        
        total_diff += diff
        gt_event_count += 1
    logger.debug(f"gt_event_count: {gt_event_count}")
    logger.debug(f"avg_diff: {total_diff / gt_event_count}")
    logger.debug(f"overflow: {overflow}")
    return np.array([total_diff / gt_event_count, overflow])


def run_metric_for_data(pred_path, args, recorder):
    """ Run the metric for a single data file.
    Args:
        file_path: path to the data file
        args: arguments
    Returns:
        metric: metric for the data file
    """
    stem = op.basename(pred_path).split('.')[0].lstrip('test-')
    file_path = op.join(args.data_root, stem+'.pkl')
    
    data = pkl.load(open(file_path, 'rb'))
    stem = op.basename(file_path).split('.')[0]
    metric = {k: None for k in args.evaluate_on}
    info = {k: 0 for k in args.evaluate_on}
    # info['gt_event_count'] = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_count = 0
    index=0
    
    #! Correct the 10x timestamp Bug in the previous data processing. Please comment this part if you are using the new data.
    data['timestamps'] = data['timestamps'] // 10
    for i in range(len(data['events'])):
        data['events'][i]['timestamp'] = data['events'][i]['timestamp'] // 10
    #! End of the correction block

    with open(pred_path, 'rb') as f:
        v2ce_recorder = pkl.load(f)

    outputs = v2ce_recorder['outputs']
    pred_voxel = outputs['pred']['voxels'][0]    
    pred_event_stream = []
    
    for idx, event_batch in enumerate(data['events']):
        event_voxel = torch.from_numpy(pred_voxel[idx].reshape(1, 2, 10, 260, 346))
        event_voxel = event_voxel.to(device)
        frame_time_diff = (data['timestamps'][index+1]-data['timestamps'][index])
        logger.debug(torch.sum(event_voxel.view(-1)))
        event_batch ['timestamp'] = (event_batch ['timestamp'] - data['timestamps'][index])
        fps = 30 / frame_time_diff*33333
        logger.debug(f"fps: {fps}")
        logger.debug(f"frame_time_diff: {frame_time_diff}")

        if 'ours' in args.evaluate_on:
            # Calculate the metric for our method
            logger.debug("Now calculating our method...")
            pred = our_sample(event_voxel, fps=fps, additional_events_strategy=args.additional_events_strategy, bidirectional=args.bidirectional)[0]
            logger.debug(f"Predicted event number: {len(pred)}")
            pred_event_stream.append(pred)
            
            if metric['ours'] is None:
                metric['ours'] = ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            else:
                metric['ours'] += ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            info['ours'] += len(pred) / len(event_batch)
        
        if 'random' in args.evaluate_on:
            # Calculate the metric for random method
            logger.debug("\nNow calculating random method...")
            pred = baseline_sample(event_voxel, fps=fps, random=True)[0]
            
            logger.debug(f"Predicted event number: {len(pred)}")
            if metric['random'] is None:
                metric['random'] = ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            else:
                metric['random'] += ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            info['random'] += len(pred) / len(event_batch)
        
        if 'even' in args.evaluate_on:
            # Calculate the metric for even method
            logger.debug("\nNow calculating even method...")
            pred = baseline_sample(event_voxel, fps=fps, even=True)[0]
            
            logger.debug(f"Predicted event number: {len(pred)}")
            if metric['even'] is None:
                metric['even'] = ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            else:
                metric['even'] += ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            info['even'] += len(pred) / len(event_batch)

        if 'slope' in args.evaluate_on:
            # Calculate the metric for slope method
            logger.debug("\nNow calculating slope method...")
            pred = slope_sample(event_voxel, fps=fps, pooling_type=args.pooling_type)[0]

            logger.debug(f"Predicted event number: {len(pred)}")
            if metric['slope'] is None:
                metric['slope'] = ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            else:
                metric['slope'] += ts_diff_metric(event_batch, pred, search_range=args.search_range, fps=fps)
            info['slope'] += len(pred) / len(event_batch)

        batch_count += 1
        index +=1 
        
        recovered_pred_event_voxel = gen_discretized_event_volume(pred, (args.bin_num * 2, 260, 346)).reshape(1, 2, args.bin_num, 260, 346).to(device)

        logger.debug(f"Accumulated error between gt voxel and predicted and regenerated voxel: {torch.sum(torch.abs(torch.clamp(event_voxel,0,1)-recovered_pred_event_voxel)>=0.9).item()}")
        logger.debug(f"Average error between gt voxel and predicted and regenerated voxel: {torch.mean((torch.abs(torch.clamp(event_voxel,0,1)-recovered_pred_event_voxel))).item()}")
    for k in metric.keys():
        metric[k] = np.append(metric[k], info[k])
        
    # save pred_event_stream to file
    if args.recorder_out_path is not None:
        out_es_recorder_name = op.basename(pred_path).split('.')[0].lstrip('test-') + '.pkl'
        with open(op.join(args.recorder_out_path, out_es_recorder_name), 'wb') as f:
            pkl.dump(pred_event_stream, f)
            
    recorder[stem] = {k: v/batch_count for k, v in metric.items()}
    logger.info("File {} done.".format(stem))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_path', type=str, default=r"/tsukimi/datasets/MVSEC/data_paths_new.pkl", help='path to the data info file')
    parser.add_argument('--pred_root', type=str, default=r"/tsukimi/v2ce-project/best_model_log/recorder", help='path to the data root')
    parser.add_argument('--data_root', type=str, default=r"/tsukimi/datasets/MVSEC/event_chunks_10t", help='path to the data root')
    parser.add_argument('--recorder_out_path', type=str, default=r"/tsukimi/v2ce-project/stage2-results/v2ce_pred_event_stream", help='path to the output recorder dir.')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--search_range', type=int, default=0, help='search range for each GT event')
    parser.add_argument('--bin_num', type=int, default=10, help='number of bins for the event volume')
    parser.add_argument('--log_level', type=str, default='INFO', help='logging level')
    parser.add_argument('--file_num', type=int, default=1, help='number of files to run the metric. If set to -1, run the metric for all files')
    parser.add_argument('--file_start', type=int, default=0, help='the index of the first file to run the metric')
    parser.add_argument('--evaluate_on', default=['ours', 'random', 'slope'], nargs='*', help="Loss type.")
    parser.add_argument('-a', '--additional_events_strategy', default='slope', choices=['random', 'slope', 'none'], help="Type2 events processing method.")
    parser.add_argument('-p', '--pooling_type', default='none', choices=['none', 'weighted', 'avg'], help="Pooling type of the y_pool used to calculate slope.")
    parser.add_argument('-b', '--bidirectional', action='store_true', help="Whether to use bidirectional y_relocate inference.")
    parser.add_argument('--exp_name', type=str, default='exp', help='name of the experiment')
    parser.add_argument('--proc_num', type=int, default=8, help='number of processes to run the metric')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger('main')
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    info = pkl.load(open(args.info_path, 'rb'))

    v2ce_recorder_paths = [op.join(args.pred_root, f) for f in os.listdir(args.pred_root) if f.endswith('.pkl')]

    result = None    
    manager = Manager()
    recorder = manager.dict()
    
    v2ce_recorder_paths = v2ce_recorder_paths[args.file_start:args.file_start+args.file_num] if args.file_num != -1 else v2ce_recorder_paths
    pool = Pool(args.proc_num)
    for path in tqdm(v2ce_recorder_paths):
        pool.apply_async(run_metric_for_data, args=(path, args, recorder))
    pool.close()
    pool.join()
    
    stem = op.basename(v2ce_recorder_paths[0]).split('.')[0].lstrip('test-')
    result_merged = {k:None for k in recorder[stem].keys()}
    print(recorder)
    print(recorder.keys())
    print(recorder[stem].keys())
    print(result_merged)
    
    for k in recorder[stem].keys():
        result_merged[k] = np.array([recorder[f][k] for f in recorder.keys()]).mean(axis=0)
    
    result_df = pd.DataFrame(result_merged, index=['Avg Error', '#Overflow', 'Pred GT Event # Ratio']).T
    # keep the first 3 digits
    result_df = result_df.map(lambda x: round(x, 3))
    # set the dtype of '#Overflow' to int
    result_df['#Overflow'] = result_df['#Overflow'].astype(int)
    
    our_dir_name = '{}-{}-{}'.format(args.additional_events_strategy, args.pooling_type, args.exp_name)
    out_dir = op.join('results', args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    
    result_df.to_csv(op.join(out_dir, 'abbr_result.csv'))
    with open(op.join(out_dir, 'full_record.npy'), 'wb') as f:
        np.save(f, recorder)
    print(result_df)
