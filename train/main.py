import os
import sys
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.model import ModelInterface
from scripts.data import DataInterface
from scripts.utils import load_model_path_by_args, SBool, build_working_tree, init_logging, get_gpu_num
from scripts.utils.callbacks import MetricsCallback, WritePlotsToTensorBoardCallBack, RecorderCallback

import logging
logger = logging.getLogger(__name__)

def load_callbacks(args):
    callbacks = []

    callbacks.append(plc.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor='val_BinaryMatchF1_sum_c', #'val_loss',
        filename='best-{epoch:02d}-{val_BinaryMatchF1_sum_c:.4f}-{val_BinaryMatch_raw:.4f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    callbacks.append(MetricsCallback(
        monitor=['val_BinaryMatchF1_sum_c', 'val_BinaryMatchF1_sum_cp', 'val_BinaryMatchF1_raw', 'val_BinaryMatch_raw', 'val_BinaryMatch_sum_c', 'val_BinaryMatch_sum_cp', 'val_PoolMSE_2', 'val_PoolMSE_4', 'val_pyramid_loss'], 
        modes=['max', 'max', 'max', 'max', 'max', 'max', 'min', 'min', 'min']
    ))

    callbacks.append(WritePlotsToTensorBoardCallBack(
        log_frequency=args.log_frequency
    ))

    if args.recorder_types is not None:
        callbacks.append(RecorderCallback(
            mode=args.recorder_types,
        ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))

    return callbacks


def main(args, logger):
    pl.seed_everything(args.seed)
    logger.info(f'The checkpoint directory is: {args.checkpoint_dir}')
    load_path = load_model_path_by_args(args)
    logger.info(f'The data root directory is: {args.data_dir}')
    data_module = DataInterface(**vars(args))

    args.callbacks = load_callbacks(args)

    if load_path is None or not args.load_weights_only:
        model = ModelInterface(**vars(args))
    else:
        model = ModelInterface.load_from_checkpoint(load_path, **vars(args), strict=False)
        logger.info(f'Loading weights only from checkpoint {load_path}')

    # # If you want to change the tensorboard logger's saving folder
    tb_logger = TensorBoardLogger(save_dir=args.logger_dir, name='')
    args.logger = tb_logger

    trainer = Trainer.from_argparse_args(args)
    if not args.test_only:
        if load_path is not None and not args.load_weights_only:
            logger.info(f'Loading all training states from {load_path}')
            trainer.fit(model, data_module, ckpt_path=load_path)
        else:
            trainer.fit(model, data_module)
        # automatically auto-loads the best weights from the previous run
        trainer.test(model, data_module, ckpt_path='best')
    else:
        trainer.test(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    control_group = parser.add_argument_group('Basic Training Control')
    control_group.add_argument('--batch_size', default=4, type=int)
    control_group.add_argument('--num_workers', default=4, type=int)
    control_group.add_argument('--seed', default=1234, type=int)
    control_group.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for optimizer. (Regularization)')
    control_group.add_argument('--test_only', type=SBool, default=False, nargs='?', const=True, help="Only run the test function.")
    control_group.add_argument('--all_test', type=SBool, default=False, nargs='?', const=True, help="All the data files parsed are used as test set.")
    control_group.add_argument('--early_stopping_standard', type=str, default='val_pyramid_loss', help="The standard of early stopping.")
    control_group.add_argument('--recorder_types', type=str, nargs='*', default=None, help="metrics types in BinaryMatchF1")

    # LR Scheduler
    lr_group = parser.add_argument_group('LR and Scheduler')
    lr_group.add_argument('--lr', default=1e-3, type=float)
    lr_group.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    lr_group.add_argument('--lr_decay_steps', default=10, type=int)
    lr_group.add_argument('--lr_decay_rate', default=0.5, type=float)
    lr_group.add_argument('--lr_decay_min_lr', default=1e-6, type=float)

    # Restart Control
    restart_group = parser.add_argument_group('Restart Control')
    restart_group.add_argument('--load_best', action='store_true')
    restart_group.add_argument('--load_dir', default=None, type=str)
    restart_group.add_argument('--load_ver', default=None, type=str)
    restart_group.add_argument('--load_v_num', default=None, type=int)
    restart_group.add_argument('--load_weights_only', type=SBool, default=False, nargs='?', const=True)
    
    # Logs and Training Info
    log_group = parser.add_argument_group('Logs and Training Info')
    log_group.add_argument('--log_dir', default='/tsukimi/v2ce-project/logs', type=str, help="The directory of the log file.") # default='lightning_logs'
    log_group.add_argument('--log_frequency', default=8, type=int, help="The frequency of logging.")
    log_group.add_argument('--exp_name', default=None, type=str, help="The name of the experiment.")
    log_group.add_argument('--logging_level', default='INFO', type=str)

    # Loss & Metrics Info
    loss_group = parser.add_argument_group('Loss & Metrics Info')
    loss_group.add_argument('--loss', default=['pyramid', 'gan', 'ef', 'ef_splitp', 'compensation'], nargs='*', help="Loss type.")
    loss_group.add_argument('--add_base_loss', type=SBool, default=False, nargs='?', const=True, help="Add base loss to the Pyramid loss function or not.")
    loss_group.add_argument('--ef_collapse_seq', type=SBool, default=True, nargs='?', const=True, help="Collapse the sequence dimension of the event frame or not.")
    loss_group.add_argument('--ef_type', default='c+cl', choices=('only_c', 'cl', 'c+cl'), type=str, help="The type of event frame. Only accumulate C channel, or accumulate C and L channel, or use both of them.")
    loss_group.add_argument('--reduction', default='mask_mean', type=str, help="Reduction method of loss.", choices=('mean', 'mask_mean', 'sum'))
    # Weights of each loss
    loss_group.add_argument('--alpha_imu', default=1, type=float, help="The weight of imu loss.")
    loss_group.add_argument('--alpha_att', default=10, type=float, help="The weight of attention loss.")
    loss_group.add_argument('--alpha_gan', default=1, type=float, help="The weight of gan loss.") # GAN Loss Range: [0.5, 50]
    loss_group.add_argument('--alpha_pyramid', default=1000, type=float, help="The weight of pyramid loss.") # Pyramid Loss Range: [5e-4, 5e-2]
    loss_group.add_argument('--alpha_ef', default=0.5, type=float, help="The weight of l1 loss.") 
    loss_group.add_argument('--alpha_cycle', default=1, type=float, help="The weight of cycle loss.") # Cycle Loss Range: [0.8, 2]
    loss_group.add_argument('--alpha_encoder', default=1, type=float, help="The weight of encoder loss.")
    loss_group.add_argument('--alpha_efc', default=5, type=float, help="The weight of efc loss.")
    loss_group.add_argument('--alpha_match', default=0.5, type=float, help="The weight of match loss")
    loss_group.add_argument('--alpha_compensation', default=1, type=float, help="The weight of compensation loss")
    loss_group.add_argument('--alpha_pt', default=1, type=float, help="The weight of pyramid temporal loss.") 
    # alpha_norm
    loss_group.add_argument('--alpha_norm', default=1e-5, type=float, help="The weight of norm loss.")

    # Metrics
    loss_group.add_argument('--metrics', type=str, nargs='*', default=['L1', 'BinaryMatch', 'BinaryMatchF1', 'PoolMSE'], help="metrics used in evaluation. Multiple input supported.") 
    loss_group.add_argument('--mean_metrics', type=SBool, default=False, nargs='?', const=True, help="Calculate the mean of metrics from different refinement stages or not.")
    loss_group.add_argument('--acc_types', type=str, nargs='*', default=['raw', 'sum_c', 'sum_cp'], help="metrics types in BinaryMatch")
    loss_group.add_argument('--f1_types', type=str, nargs='*', default=['raw', 'sum_c', 'sum_cp'], help="metrics types in BinaryMatchF1")
    loss_group.add_argument('--poolmse_kernel_sizes', type=int, nargs='*', default=[2, 4], help="kernel sizes in PoolMSE")
    
    # Model Info
    model_group = parser.add_argument_group('Model Info')
    model_group.add_argument('--model_name', default='v2ce_3d', type=str, help="The main model name.")
    model_group.add_argument('--in_channels', default=2, type=int, help="The input channel number of the model.")
    model_group.add_argument('--hidden_channels', default=512, type=int, help="The hidden channel number of the model.")
    model_group.add_argument('--out_channels', default=20, type=int, help="The output channel number of the model.")
    model_group.add_argument('--num_head', default=4, type=int, help="The number of head in the attention module.")
    model_group.add_argument('--refinement_block_num', default=1, type=int, help="The number of refinement block in the model.")
    model_group.add_argument('--gan_k', default=3, type=int, help="The number of steps to apply to the discriminator.")
    model_group.add_argument('--in_shape', default=[16,7,260,346], nargs='*', help="The shape of the input image frame.")
    model_group.add_argument('--use_patch_gan', type=SBool, default=True, nargs='?', const=True, help="Use PatchGAN or not.")
    model_group.add_argument('--add_ref_input', type=SBool, default=True, nargs='?', const=True, help="Add additional input to the refinement block or not.")
    model_group.add_argument('--gan_3d_conv', type=SBool, default=False, nargs='?', const=True, help="Use 3D conv in GAN")
    model_group.add_argument('--phy_merge', default='cat', type=str, help="The method of merging the physical attention.")
    model_group.add_argument('--polish_type', default='1d', type=str, help="The type of polish module.")
    model_group.add_argument('--unet_multi', type=SBool, default=False, nargs='?', const=True, help="Use multi output unet or not.")
    model_group.add_argument('--real_multi_out', type=SBool, default=False, nargs='?', const=True, help="Use multi output unet or not.")
    model_group.add_argument('--unet_all_residual', type=SBool, default=True, nargs='?', const=True, help="Make all Conv layers in UNet residual or not.")
    
    # Dataset Info
    data_group = parser.add_argument_group('Dataset Info')
    data_group.add_argument('--dataset', default='event_pack_dataset', type=str, help="The Dataset class to use.")
    data_group.add_argument('--data_dir', default='dummy_data', type=str)
    data_group.add_argument('--frame_size', type=int, nargs='*', default=[260,346], help="Characters whos are only used in the test session.")
    data_group.add_argument('--partial_dataset', default=1.0, type=float, help="The percentage of data that is going to be use in training and validation.")
    data_group.add_argument('--num_bins', default=10, type=int, help="The number of bins in the event time voxel.")
    data_group.add_argument('--seq_len', default=16, type=int, help="The length of the sequence.")
    # Physical Attention
    data_group.add_argument('--advanced_physical_att', type=SBool, default=True, nargs='?', const=True, help="Use advanced physical attention or not.")
    data_group.add_argument('--ceiling_att', default=25, type=int, help="The ceiling value for physical att gt.")
    data_group.add_argument('--phyatt_grid_size', default=8, type=int, help="The grid size of the physical attention.")
    # Image Gradient
    data_group.add_argument('--apply_image_grad', type=SBool, default=False, nargs='?', const=True, help="Apply image gradient to the image unit (point-wise multiplication) or not.")
    # Random Flip
    data_group.add_argument('--random_flip', type=SBool, default=False, nargs='?', const=True, help="Randomly flip the image horizontally/vertically or not.")
    data_group.add_argument('--flip_x_prob', default=0.5, type=float, help="The probability of flipping the image horizontally.")
    data_group.add_argument('--flip_y_prob', default=0, type=float, help="The probability of flipping the image vertically.")

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)
    parser.set_defaults(strategy='ddp')
    parser.set_defaults(precision=16)
    parser.set_defaults(accelerator='gpu')
    parser.set_defaults(devices=torch.cuda.device_count() if torch.cuda.is_available() else 0)

    args = parser.parse_args()
    args.find_unused_parameters = False
    args.logger_dir, args.checkpoint_dir, args.recorder_dir, args.log_profiler = build_working_tree(root=args.log_dir, name=args.exp_name)
    args.real_batch_size = args.batch_size * args.devices if args.strategy == 'ddp' else args.batch_size
    args.full_command = " ".join(["python"]+sys.argv)

    if args.apply_image_grad:
        args.in_channels += 1

    # Set the logging configuration
    init_logging(args.logging_level, log_dir=args.logger_dir)
    logger = logging.getLogger('main')

    logger.info(f'Real batch size: {args.real_batch_size}')
    logger.info(f"Complete input command: {args.full_command}")

    #! Experimental! USED TO SPEED UP
    torch.set_float32_matmul_precision('medium')

    main(args, logger)


# DEGUB RUN COMMAND
# python main.py --data_dir="/tsukimi/datasets/MVSEC/event_chunks_processed" --gpus=4 --model_name=video_to_event_model_unet --batch_size=1 --loss pyramid gan ef --logging_level=info --log_frequency=8 --lr=1e-3 --add_base_loss=True --alpha_cycle=1 --alpha_pyramid=100 --ef_collapse_seq=False