import os
import re
import time
import warnings
import pickle as pkl
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, Optional, Set

import numpy as np
import torch
from torch import Tensor
from pathlib2 import Path
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Callback, Trainer, LightningModule

import logging
logger = logging.getLogger(__name__)

def _monitor_candidates(trainer: Trainer) -> Dict[str, Tensor]:
    monitor_candidates = deepcopy(trainer.callback_metrics)
    # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
    # or does not exist we overwrite it as it's likely an error
    epoch = monitor_candidates.get("epoch")
    monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
    step = monitor_candidates.get("step")
    monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
    return monitor_candidates

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self, monitor=['val_BinaryMatchF1', 'val_BinaryMatch'], modes=['max', 'max'], report_mode=['current', 'best']):
        super().__init__()
        self.monitor = monitor if isinstance(monitor, list) or isinstance(monitor, tuple) else [monitor]
        self.modes = modes if isinstance(modes, list) or isinstance(modes, tuple) else [modes]
        self.report_mode = report_mode if isinstance(report_mode, list) or isinstance(report_mode, tuple) else [report_mode]
        self.rank = int(os.environ.get('LOCAL_RANK', 0)) if 'LOCAL_RANK' in os.environ else 0
        self.best_metrics = {metric: None for metric in monitor}

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.rank == 0 and not trainer.sanity_checking:
            monitor_candidates = _monitor_candidates(trainer)
            for metric in self.monitor:
                if metric in monitor_candidates:
                    if self.best_metrics[metric] is None:
                        self.best_metrics[metric] = monitor_candidates[metric]
                    else:
                        if self.modes[self.monitor.index(metric)] == 'max':
                            if monitor_candidates[metric] > self.best_metrics[metric]:
                                self.best_metrics[metric] = monitor_candidates[metric]
                        elif self.modes[self.monitor.index(metric)] == 'min':
                            if monitor_candidates[metric] < self.best_metrics[metric]:
                                self.best_metrics[metric] = monitor_candidates[metric]
                        else:
                            raise ValueError(f"Mode {self.modes[self.monitor.index(metric)]} not supported")
                    
                else:
                    warnings.warn(f"Metric {metric} not found in monitor candidates {monitor_candidates.keys()}")

            print('\n')
            if 'best' in self.report_mode:
                metric_info = ', '.join([f"{metric}: {(self.best_metrics[metric] if ((metric in self.best_metrics) and (self.best_metrics[metric] is not None)) else 0.0):.4f}" for metric in self.monitor])
                logger.info(f'Epoch {trainer.current_epoch} finished. Best metrics: {metric_info}') 
            if 'current' in self.report_mode:
                metric_info = ', '.join([f"{metric}: {(monitor_candidates[metric] if ((metric in monitor_candidates) and (monitor_candidates[metric] is not None)) else 0.0):.4f}" for metric in self.monitor])
                logger.info(f'Epoch {trainer.current_epoch} finished. Current metrics: {metric_info}')

class WritePlotsToTensorBoardCallBack(Callback):
    """
    Callback to write plots to tensorboard.
    """
    def __init__(self, log_frequency, monitor='BinaryMatchF1', mode='max') -> None:
        self.log_frequency = log_frequency
        self.monitor = monitor
        self.mode = mode
        self.best_metric = None
        self.eval_counter = 0
        super().__init__()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.eval_counter = 0

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.eval_counter = 0

    def _write_to_tensorboard(self, trainer: Trainer, pl_module: LightningModule, batch, preds, mode, batch_idx):
        assert mode in ['val', 'test']
        """
        monitor_candidates = _monitor_candidates(trainer)
        assert self.monitor in monitor_candidates.keys(), f"Monitor {self.monitor} not found in monitor candidates {monitor_candidates.keys()}"

        if self.monitor in monitor_candidates:
            if self.best_metric is None:
                self.best_metric = monitor_candidates[self.monitor]
            else:
                if self.mode == 'max':
                    if monitor_candidates[self.monitor] > self.best_metric:
                        self.best_metric = monitor_candidates[self.monitor]
                    else:
                        return
                elif self.mode == 'min':
                    if monitor_candidates[self.monitor] < self.best_metric:
                        self.best_metric = monitor_candidates[self.monitor]
                    else:
                        return
                else:
                    raise ValueError(f"Mode {self.mode} not supported")
        """
        epoch = trainer.current_epoch
        loss = pl_module.hparams.loss
        frame_size = pl_module.hparams.frame_size
        tb = pl_module.logger.experiment

        frames = batch['image_units']
        time_voxel = batch['voxels']
        lfr = batch['lfr']
        pred_voxels = preds['voxels']        
        
        pred_voxel_vises = [pred_voxel[0].sum(dim=1).unsqueeze(1) for pred_voxel in pred_voxels]
        
        for i in range(len(pred_voxel_vises)):
            tb.add_images(f'{mode}-batch_{batch_idx}-pred_voxel_sum-stage_{i}', pred_voxel_vises[i]/pred_voxel_vises[i].max(), epoch)
        
        if epoch == 0:
            target_voxel_vis = time_voxel[0].sum(dim=1).unsqueeze(1)
            tb.add_images(f'{mode}-batch_{batch_idx}-target_voxel_sum', target_voxel_vis/target_voxel_vis.max())
            tb.add_images(f'{mode}-batch_{batch_idx}-input_frame_0', (0.165*frames[0,:,0:1]+0.153))
            tb.add_images(f'{mode}-batch_{batch_idx}-lfr_0', lfr[0])
        
        if 'physical' in loss and 'physical_atts' in preds.keys() and len(preds['physical_atts']) > 0:
            physical_att = batch['physical_att']
            pred_physical_atts = preds['physical_atts']
            
            if epoch == 0:
                target_physical_att_vis = F.interpolate(physical_att[0], size=(frame_size[0], frame_size[1]), mode='nearest')
                tb.add_images(f'{mode}-batch_{batch_idx}-target_physical_att', target_physical_att_vis)
            for i in range(len(pred_physical_atts)):
                pred_physical_atts_vis = F.interpolate(pred_physical_atts[i][0], size=(frame_size[0], frame_size[1]), mode='nearest')
                # normalize to [0,1]
                pred_physical_atts_vis = (pred_physical_atts_vis - pred_physical_atts_vis.min())/(pred_physical_atts_vis.max() - pred_physical_atts_vis.min()) 
                tb.add_images(f'{mode}-batch_{batch_idx}-pred_physical_att-stage_{i}', pred_physical_atts_vis, epoch)

        if 'tensor_vis' in preds.keys() and len(preds['tensor_vis'].keys()) > 0:
            for k, v in preds['tensor_vis'].items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        tb.add_images(f'{mode}-batch_{batch_idx}-{k}-stage_{i}', v[i][0], epoch)
                else:
                    tb.add_images(f'{mode}-batch_{batch_idx}-{k}', v[0], epoch)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.eval_counter % self.log_frequency == 0:
            self._write_to_tensorboard(trainer, pl_module, batch, outputs['pred'], 'val', batch_idx)
        self.eval_counter += 1
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self._write_to_tensorboard(trainer, pl_module, batch, outputs['pred'], 'test', batch_idx)
        self.eval_counter += 1
        

class RecorderCallback(Callback):
    """PyTorch Lightning recorder callback."""

    def __init__(self, mode=['val']):
        super().__init__()
        if isinstance(mode, str):
            mode = [mode]
        assert mode in [['val', 'test'], ['val'], ['test']]
        self.mode = mode
        
    @staticmethod
    def pre_process(pack):
        # For every element in the batch, recursively apply .cpu().numpy() if it is a tensor, else go deeper
        if isinstance(pack, Tensor):
            return pack.cpu().numpy()
        elif isinstance(pack, list) or isinstance(pack, tuple):
            return [RecorderCallback.pre_process(p) for p in pack]
        elif isinstance(pack, dict):
            return {k: RecorderCallback.pre_process(v) for k, v in pack.items()}
        else:
            return pack
    
    @staticmethod
    def get_idx_component(pack, idx):
        # For every element in the pack, select the idx-th element if it is a tensor, else go deeper
        if isinstance(pack, Tensor) or isinstance(pack, np.ndarray):
            return pack[idx]
        elif isinstance(pack, list) or isinstance(pack, tuple):
            if len(pack) == 0:
                return pack
            elif type(pack[0]) in [Tensor, np.ndarray, list, tuple, dict]:
                return [RecorderCallback.get_idx_component(p, idx) for p in pack]
            else:
                return pack[idx]
        elif isinstance(pack, dict):
            return {k: RecorderCallback.get_idx_component(v, idx) for k, v in pack.items()}
        else:
            return pack

    def save_record(self, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, record_mode: str):
        recorder_dir = pl_module.hparams.recorder_dir
        outputs = RecorderCallback.pre_process(outputs)
        batch = RecorderCallback.pre_process(batch)
        
        for idx in range(len(batch['data_path'])):
            stem = Path(batch['data_path'][idx]).stem
            # For every element in the batch, recursively apply .cpu().numpy() if it is a tensor
            # print(RecorderCallback.pre_process(outputs).keys(), RecorderCallback.pre_process(outputs)['pred'].keys())
            record = {
                'outputs': RecorderCallback.get_idx_component(outputs, idx), #{k:v[idx] for k,v in RecorderCallback.pre_process(outputs).items()},
                'batch': RecorderCallback.get_idx_component(batch, idx),#{k:v[idx] for k,v in RecorderCallback.pre_process(batch)},
            }
            recorder_path = Path(recorder_dir) / f'{record_mode}-{stem}.pkl'
            with open(recorder_path, 'wb') as f:
                pkl.dump(record, f)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if 'val' in self.mode:
            self.save_record(pl_module, outputs, batch, batch_idx, 'val')
        
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if 'test' in self.mode:
            self.save_record(pl_module, outputs, batch, batch_idx, 'test')
        