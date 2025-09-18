import importlib
import inspect
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from einops import rearrange

from .metrics import Accuracy, MeanRatio, BinaryMatch, BinaryMatchF1, PoolMSE
from .losses import Pyramid3dLoss, EncoderLoss, MatchLoss, CompensationLoss, PyramidTemporalLoss
from .gan import GANLoss

import logging
logger = logging.getLogger(__name__)

class ModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        logger.info('Entering ModelInterface...')
        self.save_hyperparameters(ignore=['in_cnn'])
        if 'callbacks' in self.hparams.keys():
            del self.hparams['callbacks']

        logger.debug(f"Losses: {self.hparams.loss}")
        if 'gan' in self.hparams.loss:
            self.GAN = GANLoss(
                use_3d_conv=self.hparams.gan_3d_conv,
                in_channels=2 if self.hparams.gan_3d_conv else 20,
                gan_k=self.hparams.gan_k, 
                use_patch_gan=self.hparams.use_patch_gan
            )

        self.loss_function = {}
        self.load_model()
        self.configure_loss()
        self.configure_metrics()
        logger.info('Model Initialized.')
        logger.debug(f"Parameters:\n{self.hparams}")
        logger.debug(f"Children modules:\n{[name for name, _ in self.named_children()]}")
        
    def setup(self, stage) -> None:
        self.tb = self.logger.experiment
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
    
    def forward(self, x):
        """
        For inference. Return normalized skeletons
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pred = self(batch)

        # Calculate loss
        loss, loss_dict = self.calculate_loss(batch, pred)
        
        # Log the loss and metrics
        self.log('loss_epoch', loss.cpu().detach().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.real_batch_size, sync_dist=True)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.hparams.real_batch_size, sync_dist=True)
        return loss

    def on_train_end(self) -> None:
        return super().on_train_end()

    def validation_step(self, batch, batch_idx):
        pred = self._eval(batch, mode='val', idx=batch_idx)  # Normalized metircs
        return {'pred': pred}

    def test_step(self, batch, batch_idx):
        pred = self._eval(batch, mode='test', idx=batch_idx)
        return {'pred': pred}

    def predict_step(self, batch, batch_idx=0):
        pred = self(batch)
        return pred

    def _eval(self, batch, mode='val', idx=0):
        assert mode in ['val', 'test', 'predict']
        
        # Do the inference
        pred = self(batch)
        pred = self.organize_pred(pred)
        
        # Calculate loss
        loss, loss_dict = self.calculate_loss(batch, pred)

        # Calculate metrics
        metrics = self.calculate_metrics(batch, pred)

        # Log the loss and metrics
        self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.hparams.real_batch_size, sync_dist=True)
        metrics_log = {f'{mode}_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics_log, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.hparams.real_batch_size, sync_dist=True)
        loss_dict_log = {f'{mode}_{k}': v for k, v in loss_dict.items()}
        self.log_dict(loss_dict_log, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.hparams.real_batch_size, sync_dist=True)

        return pred

    def organize_pred(self, pred):
        """ Organize the pred dict to adapt the predicted results from different models.
        """
        # If the model only outputs one voxel, we need to wrap it into a list
        if not isinstance(pred['voxels'], list):
            pred['voxels'] = [pred['voxels']]
        
        # If the physical_atts is not a list, we need to wrap it into a list
        if 'physical_atts' in pred.keys() and not isinstance(pred['physical_atts'], list):
            pred['physical_atts'] = [pred['physical_atts']]
        return pred
        
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def calculate_metrics(self, batch, pred):
        gt_voxels = batch['voxels']
        pred_voxels = pred['voxels']
        if not isinstance(pred_voxels, list):
            pred_voxels = [pred_voxels]
            
        if self.hparams.mean_metrics:
            metircs = {
                metric_name: torch.mean(torch.stack([metric_function(pred_voxel, gt_voxels) for pred_voxel in pred_voxels]))
                for metric_name, metric_function in self.metrics.items()
            }
        else:
            metircs = {
                metric_name: metric_function(pred_voxels[-1], gt_voxels)
                for metric_name, metric_function in self.metrics.items()
            }
        return metircs

    def calculate_loss(self, batch, pred):
        loss_dict = {}
        loss = 0
        
        gt_voxels = batch['voxels']
        pred_voxels = pred['voxels']
        
        # IMU loss
        if 'imu' in self.hparams.loss:
            gt_imu = batch['imu']
            pred_imu = pred['imu']
            _imu_loss = self.imu_loss(pred_imu, gt_imu)
            loss += self.hparams.alpha_imu * _imu_loss
            loss_dict['imu_loss'] = _imu_loss.detach().cpu()
            logger.debug(f"IMU Loss: {_imu_loss}, IMU Alpha: {self.hparams.alpha_imu}")

        # Physical Attention loss
        if 'physical' in self.hparams.loss and 'physical_atts' in pred.keys():
            _att_losses = []
            gt_phyatt = batch['physical_att']
            pred_physical_atts = pred['physical_atts']
            
            # print('gt_phyatt device: ', gt_phyatt.device, 'pred_physical_atts device: ', pred_physical_atts[0].device)
            for i in range(len(pred_physical_atts)):
                _att_loss = self.att_loss(pred_physical_atts[i], gt_phyatt)
                _att_losses.append(_att_loss)

            _att_loss = sum(_att_losses) / len(_att_losses)
            loss += self.hparams.alpha_att * _att_loss
            loss_dict['att_loss'] = _att_loss.detach().cpu()
            logger.debug(f"ATTENTION Loss: {_att_loss}, ATT Alpha: {self.hparams.alpha_att}")

        # >------------------- Voxel-Related Loss -------------------<
        if not isinstance(pred_voxels, list):
            pred_voxels = [pred_voxels]
        B, L, C, H, W = pred_voxels[0].shape
        
        # Event Frame loss
        ef_series_loss = []
        if 'ef' in self.hparams.loss:
            ef_series_loss.append('ef')
        if 'ef_splitp' in self.hparams.loss:
            ef_series_loss.append('ef_splitp')
        
        if len(ef_series_loss) > 0:
            _ef_losses = []
            for ef_loss_type in ef_series_loss:
                if ef_loss_type == 'ef_splitp':
                    gt_voxels_for_ef = rearrange(gt_voxels, 'b l (p c) h w -> b l c p h w', p=2)
                    pred_voxels_for_ef = [rearrange(pred_voxel, 'b l (p c) h w -> b l c p h w', p=2) for pred_voxel in pred_voxels]
                elif ef_loss_type == 'ef':
                    gt_voxels_for_ef = gt_voxels
                    pred_voxels_for_ef = pred_voxels
                else:
                    raise ValueError(f'Invalid ef_loss_type {ef_loss_type}!')
                
                for pred_voxel in pred_voxels_for_ef:
                    #ef_mask = (torch.sign(pred_event_frame-0.2)+1)/2
                    if self.hparams.ef_type == 'cl':
                        gt_ef = torch.sum(torch.abs(gt_voxels_for_ef),dim=(1,2))
                        pred_ef = torch.sum(torch.abs(pred_voxel),dim=(1,2))
                        _ef_loss = self.ef_loss(pred_ef, gt_ef)
                    elif self.hparams.ef_type == 'only_c':
                        gt_ef = torch.sum(torch.abs(gt_voxels_for_ef),dim=(2))
                        pred_ef = torch.sum(torch.abs(pred_voxel),dim=(2))
                        _ef_loss = self.ef_loss(pred_ef, gt_ef)
                    elif self.hparams.ef_type == 'c+cl':
                        gt_ef_c = torch.sum(torch.abs(gt_voxels_for_ef),dim=(2))
                        gt_ef_cl = torch.sum(torch.abs(gt_voxels_for_ef),dim=(1,2))
                        pred_ef_c = torch.sum(torch.abs(pred_voxel),dim=(2))
                        pred_ef_cl = torch.sum(torch.abs(pred_voxel),dim=(1,2))
                        _ef_loss = self.hparams.alpha_efc * self.ef_loss(pred_ef_c, gt_ef_c) + self.ef_loss(pred_ef_cl, gt_ef_cl)
                    else:
                        raise ValueError(f'Invalid ef_type {self.hparams.ef_type}!')
                    # Balance the ef_loss for different ef_loss_type
                    _ef_loss = _ef_loss * 2 if ef_loss_type == 'ef_splitp' else _ef_loss
                    _ef_losses.append(_ef_loss)

            _ef_loss = sum(_ef_losses) / len(_ef_losses)
            loss += self.hparams.alpha_ef * _ef_loss
            loss_dict['ef_loss'] = _ef_loss.detach().cpu()
            logger.debug(f"EVENT FRAME Loss: {_ef_loss}, EF Alpha: {self.hparams.alpha_ef}")

        # Encoder Loss
        if 'encoder' in self.hparams.loss:
            _encoder_losses = []
            for pred_voxel in pred_voxels:
                _encoder_loss = self.loss_function['encoder'](pred_voxel, gt_voxels)
                _encoder_losses.append(_encoder_loss)

            _encoder_loss = sum(_encoder_losses) / len(_encoder_losses)
            loss += self.hparams.alpha_encoder * _encoder_loss
            loss_dict['encoder_loss'] = _encoder_loss.detach().cpu()
            logger.debug(f"Encoder Loss: {_encoder_loss}, Encoder Alpha: {self.hparams.alpha_encoder}")

        """
        # Merge the first two dimensions (B, L) of pred_voxel and gt_voxels
        for i in range(len(pred_voxels)):
            # pred_voxels[i] = pred_voxels[i].reshape(-1, *pred_voxels[i].shape[2:])
            pred_voxels[i] = rearrange(pred_voxels[i], 'b l (p c) h w -> (b p) (l c) h w', p=2)
        gt_voxels = gt_voxels.reshape(-1, *gt_voxels.shape[2:])
        """

        # Pyramid Loss
        if 'pyramid' in self.hparams.loss:
            _gt_voxels = rearrange(gt_voxels, 'b l (p c) h w -> (b p) (l c) h w', p=2)
            _pyramid_losses = []
            for pred_voxel in pred_voxels:
                pred_voxel = rearrange(pred_voxel, 'b l (p c) h w -> (b p) (l c) h w', p=2)
                _pyramid_loss =  self.loss_function['pyramid'](pred_voxel, _gt_voxels)
                _pyramid_losses.append(_pyramid_loss)
            
            _pyramid_loss = sum(_pyramid_losses) / len(_pyramid_losses)
            loss += self.hparams.alpha_pyramid * _pyramid_loss
            loss_dict['pyramid_loss'] = _pyramid_loss.detach().cpu()
            logger.debug(f"PYRAMID Loss: {_pyramid_loss}, PYRAMID Alpha: {self.hparams.alpha_pyramid}")

        # Pyramid Temporal Loss
        if 'pt' in self.hparams.loss:
            _pt_losses = []
            _gt_voxels = rearrange(gt_voxels, 'b l (p c) h w -> (b p) (l c) h w', p=2)
            for pred_voxel in pred_voxels:
                pred_voxel = rearrange(pred_voxel, 'b l (p c) h w -> (b p) (l c) h w', p=2)
                _pt_loss =  self.loss_function['pt'](pred_voxel, _gt_voxels)
                _pt_losses.append(_pt_loss)
            
            _pt_loss = sum(_pt_losses) / len(_pt_losses)
            loss += self.hparams.alpha_pyramid * _pt_loss
            loss_dict['pt_loss'] = _pt_loss.detach().cpu()
            logger.debug(f"PYRAMID Temporal Loss: {_pt_loss}, PYRAMID Alpha: {self.hparams.alpha_pt}")
        
        # GAN Loss
        if 'gan' in self.hparams.loss:
            _gan_losses = []
            _gt_voxels = rearrange(gt_voxels, 'b l c h w -> (b l) c h w')
            for pred_voxel in pred_voxels:
                pred_voxel = rearrange(pred_voxel, 'b l c h w -> (b l) c h w')
                _gan_loss = self.loss_function['gan'](pred_voxel, _gt_voxels)
                _gan_losses.append(_gan_loss)
            
            # Don't average the GAN loss since we set gan_k to 1 in the config already
            _gan_loss = sum(_gan_losses) #/ len(_gan_losses)
            loss += self.hparams.alpha_gan * _gan_loss
            loss_dict['gan_loss'] = _gan_loss.detach().cpu()
            logger.debug(f"GAN Loss: {_gan_loss}, GAN Alpha: {self.hparams.alpha_gan}")

        if 'match' in self.hparams.loss:
            _match_losses = []
            for pred_voxel in pred_voxels:
                _match_loss = self.loss_function['match'](pred_voxel, gt_voxels)
                _match_losses.append(_match_loss)
            _match_loss = sum(_match_losses) / len(_match_losses)
            loss += self.hparams.alpha_match * _match_loss
            loss_dict['match'] = _match_loss.detach().cpu()
            logger.debug(f'Match Loss: {_match_loss}, Match Alpha: {self.hparams.alpha_match}')

        if 'compensation' in self.hparams.loss:
            _compensation_losses = []
            for pred_voxel in pred_voxels:
                _compensation_loss = self.loss_function['compensation'](pred_voxel, gt_voxels)
                _compensation_losses.append(_compensation_loss)
            _compensation_loss = sum(_compensation_losses) / len(_compensation_losses)
            loss += self.hparams.alpha_compensation * _compensation_loss
            loss_dict['compensation'] = _compensation_loss.detach().cpu()
            logger.debug(f'Compensation Loss: {_compensation_loss}, Compensation Alpha: {self.hparams.alpha_compensation}')

        if 'norml1' in self.hparams.loss:
            _norml1_losses = []
            for pred_voxel in pred_voxels:
                _norml1_loss = torch.norm(pred_voxel, p=1)
                _norml1_losses.append(_norml1_loss)
            _norml1_loss = sum(_norml1_losses) / len(_norml1_losses)
            loss += self.hparams.alpha_norm * _norml1_loss
            loss_dict['norml1'] = _norml1_loss.detach().cpu()
            logger.debug(f'Norm L1 Loss: {_norml1_loss}, Norm L1 Alpha: {self.hparams.alpha_norm}')

        if 'norml2' in self.hparams.loss:
            _norml2_losses = []
            for pred_voxel in pred_voxels:
                _norml2_loss = torch.norm(pred_voxel, p=2)
                _norml2_losses.append(_norml2_loss)
            _norml2_loss = sum(_norml2_losses) / len(_norml2_losses)
            loss += self.hparams.alpha_norm * _norml2_loss
            loss_dict['norml2'] = _norml2_loss.detach().cpu()
            logger.debug(f'Norm L2 Loss: {_norml2_loss}, Norm L2 Alpha: {self.hparams.alpha_norm}')

        for i in range(len(pred_voxels)):
            pred_voxels[i] = pred_voxels[i].reshape(B,L,C,H,W)

        return loss, loss_dict

    def configure_loss(self):
        loss_list = self.hparams.loss
        for loss_str in loss_list:
            if loss_str == 'l1':
                self.loss_function['l1'] = nn.L1Loss()
            elif loss_str == 'l2':
                self.loss_function['l2'] = nn.MSELoss()
            elif loss_str == 'pyramid':
                self.loss_function['pyramid'] = Pyramid3dLoss(add_base_loss=self.hparams.add_base_loss)
            elif loss_str == 'pt':
                self.loss_function['pt'] = PyramidTemporalLoss()
            elif loss_str == 'gan':
                self.loss_function['gan'] = self.GAN
            elif loss_str == 'encoder':
                self.loss_encoder = EncoderLoss()
                self.loss_function['encoder'] = self.loss_encoder
            elif loss_str=='match':
                self.loss_function['match'] = MatchLoss()
            elif loss_str=='compensation':
                self.loss_function['compensation'] = CompensationLoss()
            elif loss_str in ['physical', 'ef', 'imu', 'ef_splitp', 'norml1', 'norml2']:
                pass
            else:
                raise ValueError(f'Invalid loss type {loss_str}!')
            
        self.imu_loss = nn.MSELoss()
        self.att_loss = nn.MSELoss()
        self.ef_loss = nn.MSELoss()

    def configure_metrics(self):
        self.metrics = {}
        self.hparams.metrics = [m.lower() for m in self.hparams.metrics]

        if 'acc' in self.hparams.metrics:
            self.metrics['Acc'] = Accuracy()
        
        if 'binarymatch' in self.hparams.metrics:
            for acc_type in self.hparams.acc_types:
                self.metrics[f'BinaryMatch_{acc_type}'] = BinaryMatch(op_type=acc_type)
        
        if 'binarymatchf1' in self.hparams.metrics:
            for f1_type in self.hparams.f1_types:
                self.metrics[f'BinaryMatchF1_{f1_type}'] = BinaryMatchF1(op_type=f1_type)

        if 'meanratio' in self.hparams.metrics:
            self.metrics['MeanRatio'] = MeanRatio()
        
        if 'poolmse' in self.hparams.metrics:
            if self.hparams.poolmse_kernel_sizes is None:
                self.hparams.poolmse_kernel_sizes = [2, 4]
            for kernel_size in self.hparams.poolmse_kernel_sizes:
                self.metrics[f'PoolMSE_{kernel_size}'] = PoolMSE(kernel_size=kernel_size)

        if 'l1' in self.hparams.metrics:
            self.metrics['L1'] = nn.L1Loss()

        if self.hparams.metrics is None:
            # Accuracy is the default metric
            self.metrics = {'Acc': Accuracy()}
            

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except Exception as e:
            logger.error(str(e))
            raise ValueError(
                f'Failed initializing the model class! This issue may be caused by invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)