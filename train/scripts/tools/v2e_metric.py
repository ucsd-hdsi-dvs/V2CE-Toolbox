import sys
import h5py
import os.path as op
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib2 import Path

sys.path.append('../..')
from scripts.utils.events_utils import gen_discretized_event_volume, gen_discretized_event_volume_from_tensor
from scripts.model.metrics import BinaryMatchF1, BinaryMatch, PoolMSE

v2e_processed_dir = '/tsukimi/datasets/MVSEC/v2e-test-correct'

def dataset_metrics(path, results_folder, metrics_dict, metrics_results, n_time_bins=10):
    """ Calculate the metrics for a single data file which contains a batch of events and images.
    Args:
        path: path to the data file
        results_folder: folder to save the results
        metrics_dict: dictionary of metrics
        metrics_results: dictionary of metrics results
    Returns:
        None (note that the metrics_results dictionary is updated in-place)
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    gt_events = data['events']
    stem = Path(path).stem
    filename = Path(path).name
    
    f= h5py.File(op.join(v2e_processed_dir, filename, "events.h5"))
    events_pred = f['events'].astype(np.int32)
    ts_pred_min = events_pred[0,0]
    ts_pred_max = events_pred[-1,0]
    # Split the events_pred into 16 batches evenly by time.
    time_splits = np.linspace(ts_pred_min, ts_pred_max, 16+1)
    events_pred_split = []
    for i in range(16):
        # events_pred[:,0] is the sorted timestamps, use searchsorted to find the index of the first element >= time_splits[i]
        events_pred_split.append(events_pred[events_pred[:,0].searchsorted(time_splits[i]):events_pred[:,0].searchsorted(time_splits[i+1])])
        # (events_pred[(events_pred[:,0]>=time_splits[i]) & (events_pred[:,0]<time_splits[i+1])])
    
    batch_event_voxels_pred = np.zeros((16, n_time_bins*2, 260, 346))
    batch_event_voxels_gt = np.zeros((16, n_time_bins*2, 260, 346))

    for index in range (len(gt_events)):
        events_truth = data['events'][index]
        event_volume_truth = gen_discretized_event_volume(events_truth,
                                                        [n_time_bins*2,
                                                        260,
                                                        346])
        
        events_pred_piece = events_pred_split[index]
        # events_pred_piece[:,0] -= time_splits[index]
        events_pred_piece = torch.tensor(events_pred_piece)
        event_volume_est = gen_discretized_event_volume_from_tensor(events_pred_piece,
                                                        [n_time_bins*2,
                                                        260,
                                                        346])
        
        batch_event_voxels_gt[index] = event_volume_truth.cpu().numpy()
        batch_event_voxels_pred[index] = event_volume_est.cpu().numpy()
        
    for metric_name, metric in metrics_dict.items():
        metrics_results[metric_name].append(metric.forward(
            torch.Tensor(batch_event_voxels_pred).cuda().unsqueeze(0), 
            torch.Tensor(batch_event_voxels_gt).cuda().unsqueeze(0)).item()
        )
    
    # Save the GT and predicted event volumes.
    out_info = {
        'gt_event_volume': batch_event_voxels_gt,
        'gen_event_volume': batch_event_voxels_pred,
        'metrics': {k:v[-1] for k,v in metrics_results.items() if len(v) > 0},
    }
    
    out_path = op.join(results_folder, f'{stem}.pkl')
    
    with open(out_path, 'wb') as f:
        pickle.dump(out_info, f)
    
if __name__ == '__main__':    
    metrics_dict = {
        'BinaryMatchF1_sum_c': BinaryMatchF1(op_type='sum_c'),
        'BinaryMatchF1_sum_cp': BinaryMatchF1(op_type='sum_cp'),
        'BinaryMatchF1_raw': BinaryMatchF1(op_type='raw'),
        'BinaryMatch_sum_c': BinaryMatch(op_type='sum_c'),
        'BinaryMatch_sum_cp': BinaryMatch(op_type='sum_cp'),
        'BinaryMatch_raw': BinaryMatch(op_type='raw'),
        'PoolMSE_2': PoolMSE(kernel_size=2),
        'PoolMSE_4': PoolMSE(kernel_size=4),
    }

    metrics_results = {
        'BinaryMatchF1_sum_c': [],
        'BinaryMatchF1_sum_cp': [],
        'BinaryMatchF1_raw': [],
        'BinaryMatch_sum_c': [],
        'BinaryMatch_sum_cp': [],
        'BinaryMatch_raw': [],
        'PoolMSE_2': [],
        'PoolMSE_4': [],
    }

    results_folder = op.join('/tsukimi/backup', 'v2e-test-metrics-correct')
    Path(results_folder).mkdir(exist_ok=True)
    info =pickle.load(open(r"/tsukimi/datasets/MVSEC/data_paths_new.pkl",'rb'))

    for file in tqdm(info['test']): #[100:]):
        path = r"/tsukimi/datasets/MVSEC/event_chunks_processed/"+file
        dataset_metrics(path, results_folder, metrics_dict, metrics_results) #[f1,bm])
        for metric_name, metric in metrics_dict.items():
            print(metric_name, metrics_results[metric_name][-1])
        # break

    # save the metrics results
    with open(op.join(results_folder, 'metrics_results.pkl'), 'wb') as f:
        pickle.dump(metrics_results, f)

    for metric_name, metric in metrics_dict.items():
        print(metric_name, np.mean(metrics_results[metric_name]))
