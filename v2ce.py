import os
import sys
import cv2
import torch
import logging
import argparse
import numpy as np
import os.path as op
from pathlib2 import Path
from torchvision import transforms
from functools import partial
from tqdm import tqdm

sys.path.append(op.abspath('../..'))
from scripts.v2ce_3d import V2ce3d
from scripts.LDATI import sample_voxel_statistical
from scripts.video_reader import VideoReader

def SBool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_trained_mode(model_path='./weights/v2ce_3d.pt'):
    """
    Get the trained model from the checkpoint
    Args:
        model_path: path to the checkpoint
        batch_size: batch size for inference
    Returns:
        model: the trained model
    """
    model = V2ce3d()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    model = model.to('cuda')
    return model

def image_pre_processing(images, height=260):
    """ Preprocess the images
    Args:
        images: the images to preprocess. Shape: (N, H, W)
        height: the height of the target image unit height (default: 260)
    Returns:
        image_units: the image units
    """
    frame_normalize = transforms.Compose([
                transforms.Normalize([0.153, 0.153], [0.165, 0.165])])
    
    images = images.astype(np.float32)/255 

    # Resize images so that the video's height is set to `height`, and the width is scaled accordingly, and crop the center height x width
    images = np.stack([cv2.resize(img, (int(img.shape[1]/img.shape[0]*height), height)) for img in images], axis=0)
    
    # Stack images into pairs
    image_units = torch.tensor(np.stack([images[:-1], images[1:]], axis=1))
    image_units = frame_normalize(image_units)
    return image_units

@torch.no_grad()
def infer_center_image_unit(model, image_units, width=346):
    """
    Infer the center of the image units
    Args:
        model: the trained model
        image_units: the image units to infer
        width: the width of the target image unit width (default: 346)
    Returns:
        pred_voxel: the predicted voxel
    """
    # Crop the center of the image on the width
    image_units = image_units[..., image_units.shape[-1]//2-width//2:image_units.shape[-1]//2+width//2]
    
    # Run the model
    inputs = image_units.float().cuda()#.unsqueeze(0)
    outputs = model(inputs)
    del inputs
    
    # Collect the outputs
    pred_voxel = outputs.cpu()

    logger.debug(f'Predicted voxel shape: {pred_voxel.shape}')
    return pred_voxel

@torch.no_grad()
def infer_pano_image_unit(model, image_units, width=346):
    """
    Infer the panorama of the image units
    Args:
        model: the trained model
        image_units: the image units to infer
        width: the width of the target image unit width (default: 346)
    Returns:
        pred_voxel_out: the predicted voxel
    """
    # Split images into 260x346 patches along the width
    patch_num = np.ceil(image_units.shape[-1]/width).astype(int)
    exact_div = image_units.shape[-1] % 346 == 0
    patch_remainder = image_units.shape[-1] % width
    image_units_patches = []
    for i in range(patch_num):
        if i == patch_num-1 and not exact_div:
            image_units_patches.append(image_units[..., -width:])
        else:
            image_units_patches.append(image_units[..., i*width:(i+1)*width])
        
    # Predict voxels
    pred_voxel_patches = []
    for i, image_unit in enumerate(image_units_patches):
        logger.debug(f'Predicting patch {i+1}/{len(image_units_patches)}')

        inputs = image_unit.float().cuda()#.unsqueeze(0)
        outputs = model(inputs)
        pred_voxel = outputs
        if i == len(image_units_patches)-1 and not exact_div:
            pred_voxel = pred_voxel[..., -patch_remainder:]
        pred_voxel_patches.append(pred_voxel.cpu())

    # Concatenate patches on the width
    pred_voxel_out = torch.cat(pred_voxel_patches, dim=-1)
    
    logger.debug(f'Predicted voxel shape: {pred_voxel_out.shape}')
    return pred_voxel_out

@torch.no_grad()
def video_to_voxels(model, image_paths=None, vidcap=None, infer_type='center', 
                              seq_len=16, width=346, height=260, batch_size=1):
    """ Infer the voxel from the video or image sequence
    Args:
        model: the trained model
        image_paths: the paths to the images
        vidcap: the video reader
        infer_type: the type of inference, can be center or pano
        seq_len: the sequence length
        width: the width of the image
        height: the height of the image
        batch_size: batch size for inference
    Returns:
        all_pred_voxel: the predicted voxel
    """
    assert image_paths is not None or vidcap is not None
    infer_video = True if vidcap is not None else False
    frame_count = vidcap.frame_count if infer_video else len(image_paths)
    sequence_num = np.ceil((frame_count-1)/seq_len).astype(int)
    mode = (frame_count-1) % seq_len
    starting_indexes = np.arange(sequence_num) * seq_len
    if mode != 0:
        starting_indexes[-1] -= (seq_len-mode)

    logger.debug(f'Found {frame_count} images, divided into {sequence_num} sequences')
    logger.debug(f'Starting indexes: {starting_indexes}')
    logger.debug(f'Mode: {mode}')
    
    all_pred_voxel = []
    batch_idx = 0
    input_image_batches = []
    for seq_idx in tqdm(range(len(starting_indexes))):
        starting_idx = starting_indexes[seq_idx]
        ending_idx = starting_idx + seq_len + 1 # +1 for geting the last frame of the last image unit
        logger.debug(f'Using images {starting_idx} to {ending_idx-1}')
            
        if infer_video:
            # Load rgb images as grayscale
            images = vidcap.read_frames_at_indices(range(starting_idx, ending_idx))
        else:
            image_paths_seq = image_paths[starting_idx:ending_idx]
            # Load rgb images as grayscale
            images = np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths_seq], axis=0)
        
        image_units = image_pre_processing(images, height=height)
        resized_width = image_units.shape[-1]
        
        input_image_batches.append(image_units[np.newaxis, ...])
        batch_idx += 1
        if batch_idx == batch_size or seq_idx == len(starting_indexes)-1:
            # Concatenate the input image batches
            if len(input_image_batches) > 1:
                input_image_batches = torch.cat(input_image_batches, dim=0)
            elif len(input_image_batches) == 1:
                input_image_batches = input_image_batches[0]
            else:
                raise ValueError('No input image batches')
            
            logger.debug(f'Input_image_batches shape: {input_image_batches.shape}')
            
            # Infer the voxel
            if infer_type == 'center':
                out_width = width
                pred_voxel = infer_center_image_unit(model, input_image_batches, width)
            elif infer_type == 'pano':
                out_width = resized_width
                pred_voxel = infer_pano_image_unit(model, input_image_batches, width)        
            else:
                raise ValueError(f'Invalid infer_type {infer_type}')
            batch_idx = 0
            input_image_batches = []
            
            all_pred_voxel.append(pred_voxel.cpu().detach().numpy())
        
    all_pred_voxel = merge_voxels(all_pred_voxel, height=height, width=out_width, mode=mode)
    
    logger.debug(f"predicted voxels shape: {all_pred_voxel.shape}")
    return all_pred_voxel
        
def merge_voxels(voxel_list, height=260, width=346, mode=0):
    """
    Merge the voxel list into a single voxel
    Args:
        voxel_list: the list of voxels
    """
    if len(voxel_list) > 1:
        pred_voxel = np.concatenate(voxel_list[:-1], axis=0).reshape(-1, 2, 10, height, width)
    else:
        pred_voxel = None
    
    if voxel_list[-1].shape[0] > 1:
        temp = voxel_list[-1][:-1].reshape(-1, 2, 10, height, width)
        if pred_voxel is None:
            pred_voxel = temp
        else:
            pred_voxel = np.concatenate([pred_voxel, temp], axis=0)
    
    if mode != 0:
        temp = voxel_list[-1][-1][-mode:].reshape(-1, 2, 10, height, width)
    else:
        temp = voxel_list[-1][-1].reshape(-1, 2, 10, height, width)

    if pred_voxel is None:
        pred_voxel = temp
    else:
        pred_voxel = np.concatenate([pred_voxel, temp], axis=0)

    return pred_voxel
    
def write_event_frame_video(voxel_grid, ef_video_path, fps, ceil, upper_bound_percentile=98, keep_polarity=True):
    """
    Write the event frame video.
    Args:
        voxel_grid: the voxel grid to generate the event frames
        ef_video_path: the path to write the video
        fps: the FPS of the video
        ceil: the ceiling of the ef value
    """
    logger.info("Writing event frame video...")
    # Write all_pred_ef into a mp4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    B, P, L, H, W = voxel_grid.shape
    if keep_polarity:
        efs = np.sum(voxel_grid, axis=2) # [B, P(2), L, H, W]
        # Concatenate a zero tensor to the blue channel to make the channel number 3
        efs = np.concatenate([efs, np.zeros((B, 1, H, W))], axis=1)
    else:
        efs = np.sum(voxel_grid, axis=(1,2))[:,np.newaxis,...] # [B, P(2), L(10), H, W]
        efs = np.repeat(efs, 3, axis=1) # [B, P(10), H, W]
    # get the <u>% percentile of the ef value to set the upper bound
    efs_flatten = efs.flatten()
    efs_flatten = efs_flatten[efs_flatten > 0]
    efs_upper_bound = min(np.percentile(efs_flatten, upper_bound_percentile), ceil)
    logger.info(f'Upper bound of the event frame value during video writing: {efs_upper_bound}')
    # Clip the ef value to the upper bound
    efs = np.clip(efs, 0, efs_upper_bound) / efs_upper_bound
    # Move the Channel dimension to the last dimension
    efs = np.moveaxis(efs, 1, -1)
    print(efs.shape)
    
    video_size = (W, H)
    video = cv2.VideoWriter(ef_video_path, fourcc, fps, video_size)
    for i in range(efs.shape[0]):
        frame = efs[i]#/efs.max() 
        frame = (frame*255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()
    logger.info(f'Event frame video written to {ef_video_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=30, help='FPS of the output video')
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length')
    parser.add_argument('--ceil', type=int, default=10, help='The ceiling of the ef value')
    parser.add_argument('-u', '--upper_bound_percentile', type=int, default=98, help='The percentile of the event frame nonzero values to set the upper bound during video writing')
    parser.add_argument('-f', '--image_folder', type=str, help='The folder containing the images to infer') # default='/tsukimi/v2ce-project/video_for_test/dash-cam-test-video'
    parser.add_argument('-i', '--input_video_path', type=str, help='The path to the input video')
    parser.add_argument('-o', '--out_folder', type=str, default='./output', help='The folder to save the output video')
    parser.add_argument('-t', '--infer_type', type=str, default='center', help='The type of inference, can be center or pano')
    parser.add_argument('-m', '--model_path', type=str, default='./weights/v2ce_3d.pt', help='The path to the trained model')
    parser.add_argument('--out_name_suffix', type=str, default='', help='The suffix of the output video name')
    parser.add_argument('--max_frame_num', type=int, default=1800, help='The maximum number of frames to process')
    parser.add_argument('--width', type=int, default=346, help='The width of the frame/tensor input to the model')
    parser.add_argument('--height', type=int, default=260, help='The height of the frame/tensor input to the model')
    parser.add_argument('--write_event_frame_video', type=SBool, default=True, nargs='?', const=True, help='Whether to write the event frame video')
    parser.add_argument('--vis_keep_polarity', type=SBool, default=True, nargs='?', const=True, help='Whether to keep the polarity of the event frame during visualization')
    parser.add_argument('-l', '--log_level', type=str, default='info', help='Logging level')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--stage2_batch_size', type=int, default=24, help='Batch size for inference')
    args = parser.parse_args()
    
    # Set the logging level to the specified level
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger('V2CE')

    # Check the input to make sure only one of image_folder and input_video_path is specified
    assert args.image_folder is not None or args.input_video_path is not None
    assert not (args.image_folder is not None and args.input_video_path is not None) 
    if args.image_folder is not None:
        assert os.path.exists(args.image_folder), f'{args.image_folder} does not exist'
    if args.input_video_path is not None:
        assert os.path.exists(args.input_video_path), f'{args.input_video_path} does not exist'
        
    name = Path(args.image_folder).name if args.image_folder is not None else Path(args.input_video_path).stem
    output_name=f'{name}-ceil_{args.ceil}-fps_{args.fps}' if args.out_name_suffix == '' else f'{name}-ceil_{args.ceil}-fps_{args.fps}-{args.out_name_suffix}'
    if not op.exists(args.out_folder):
        os.makedirs(args.out_folder, exist_ok=True)

    # Get the trained model
    model = get_trained_mode(model_path=args.model_path)
    
    # Run the core function and get the predicted voxel    
    if args.image_folder is not None:
        image_paths = sorted([op.join(args.image_folder, f) for f in os.listdir(args.image_folder) if f.endswith('.png')])[:args.max_frame_num] # 1800 is the max number of images in a sequence, which equals to 1 minute of 30FPS video
        logger.info(f'Now processing {args.image_folder}, Found {len(image_paths)} images.')

        # Generate the video
        pred_voxel = video_to_voxels(model, image_paths=image_paths, infer_type=args.infer_type, seq_len=args.seq_len, batch_size=args.batch_size,
                            width=args.width, height=args.height)
    elif args.input_video_path is not None:
        vidcap = VideoReader(args.input_video_path, color_mode='GRAY')
        if args.max_frame_num is not None and args.max_frame_num > 0 and vidcap.frame_count > args.max_frame_num:
            vidcap.frame_count = args.max_frame_num
        logger.info(f'Now processing {args.input_video_path}, processing {vidcap.frame_count} frames.')

        pred_voxel = video_to_voxels(model, vidcap=vidcap, infer_type=args.infer_type, seq_len=args.seq_len, batch_size=args.batch_size,
                            width=args.width, height=args.height)
    else:
        raise ValueError('Either image_folder or input_video_path should be specified')
    logger.info(f"Predicted voxel shape: {pred_voxel.shape}")
    
    if args.write_event_frame_video:
        if not op.exists(args.out_folder):
            os.makedirs(args.out_folder, exist_ok=True)
        vis_color = 'rgb' if args.vis_keep_polarity else 'gray'
        ef_video_path = op.join(args.out_folder, f'{args.infer_type}-{output_name}-pred_ef_{vis_color}.mp4')
        write_event_frame_video(pred_voxel, ef_video_path, args.fps, args.ceil, args.upper_bound_percentile, args.vis_keep_polarity)
    
    L, _, _, H, W = pred_voxel.shape
    stage2_input = pred_voxel.reshape(L, 2, 10, H, W)
    stage2_input = torch.from_numpy(stage2_input).cuda()
    
    # Initialize the LDATI function
    ldati = partial(sample_voxel_statistical, fps=args.fps, bidirectional=False, additional_events_strategy='slope')

    event_stream_per_frame = []
    for i in range(0, stage2_input.shape[0], args.stage2_batch_size):
        event_stream_per_frame.extend(ldati(stage2_input[i:i+args.stage2_batch_size]))

    # merge 16 event stream into a single one
    event_stream = []
    for i in range(L):
        event_stream_per_frame[i]['timestamp'] += int(i * 1 / args.fps * 1e6)
        event_stream.append(event_stream_per_frame[i])
    event_stream = np.concatenate(event_stream)
    logger.info(f"Generated event stream shape: , {event_stream.shape}")
    
    # Dump the event stream (numpy structured array)
    event_stream_path = op.join(args.out_folder, f'{output_name}-events.npz')
    np.savez(event_stream_path, event_stream=event_stream)
