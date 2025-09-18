import os
import sys
import cv2
import numpy as np
import torch
import os.path as op
from torchvision import transforms
from pathlib2 import Path
import argparse

sys.path.append(op.abspath('../..'))
from scripts.model.v2ce_3d import V2ce3d
from scripts.stage2.sample_methods.LDATI import sample_voxel_statistical
from functools import partial

def get_trained_mode(model_path='/tsukimi/v2ce-project/best_model_log/assets/v2ce_3d.pt'):
    """
    Get the trained model from the checkpoint
    Args:
        model_path: path to the checkpoint
        batch_size: batch size for inference
    """
    model = V2ce3d()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    model = model.to('cuda')
    return model

@torch.no_grad()
def infer_center_image_unit(model, image_units, width=346, verbose=False):
    """
    Infer the center of the image units
    Args:
        model: the trained model
        image_units: the image units to infer
        width: the width of the target image unit width (default: 346)
        verbose: whether to print out the shapes of the outputs
    """
    # Crop the center of the image on the width
    image_units = image_units[:, :, :, image_units.shape[-1]//2-width//2:image_units.shape[-1]//2+width//2]
    
    # Run the model
    inputs = image_units.float().cuda().unsqueeze(0)
    outputs = model(inputs)
    del inputs
    
    # Collect the outputs
    pred_voxel = outputs[0].cpu()
    pred_ef = torch.sum(pred_voxel, dim=(1))
    
    if verbose:
        print("Predicted voxel shape:", pred_voxel.shape)
        print("Predicted ef shape:", pred_ef.shape)
    return pred_voxel, pred_ef

@torch.no_grad()
def infer_pano_image_unit(model, image_units, width=346, verbose=False):
    """
    Infer the panorama of the image units
    Args:
        model: the trained model
        image_units: the image units to infer
        width: the width of the target image unit width (default: 346)
        verbose: whether to print out the shapes of the outputs
    """
    # Split images into 260x346 patches along the width
    patch_num = np.ceil(image_units.shape[-1]/width).astype(int)
    exact_div = image_units.shape[-1] % 346 == 0
    patch_remainder = image_units.shape[-1] % width
    image_units_patches = []
    for i in range(patch_num):
        if i == patch_num-1 and not exact_div:
            image_units_patches.append(image_units[:, :, :, -width:])
        else:
            image_units_patches.append(image_units[:, :, :, i*width:(i+1)*width])
        
    # Predict voxels and ef
    pred_voxel_patches = []
    pred_ef_patches = []
    for i, image_unit in enumerate(image_units_patches):
        if verbose:
            print(f'Predicting patch {i+1}/{len(image_units_patches)}')

        inputs = image_unit.float().cuda().unsqueeze(0)
        outputs = model(inputs)
        pred_voxel = outputs[0]
        pred_ef = torch.sum(pred_voxel, dim=(1))
        if i == len(image_units_patches)-1 and not exact_div:
            pred_voxel = pred_voxel[..., -patch_remainder:]
            pred_ef = pred_ef[..., -patch_remainder:]
        pred_voxel_patches.append(pred_voxel.cpu())
        pred_ef_patches.append(pred_ef.cpu())

    # Concatenate patches on the width
    pred_voxel_out = torch.cat(pred_voxel_patches, dim=-1)
    pred_ef_out = torch.cat(pred_ef_patches, dim=-1)
    
    if verbose:
        print("Predicted voxel shape:", pred_voxel_out.shape)
        print("Predicted ef shape:", pred_ef_out.shape)
    return pred_voxel_out, pred_ef_out

def kitti_image_pre_processing(images, height=260):
    frame_normalize = transforms.Compose([
                transforms.Normalize([0.153, 0.153], [0.165, 0.165])])
    
    # print("Max pixel value: ", images.max())
    # print("Min pixel value: ", images.min())
    images = images.astype(np.float32)/255 

    # Resize images so that the height becomes 260, and the width is scaled accordingly, and crop the center 260 x 346
    images = np.stack([cv2.resize(img, (int(img.shape[1]/img.shape[0]*height), height)) for img in images], axis=0)
    
    # Stack images into pairs
    image_units = torch.tensor(np.stack([images[:-1], images[1:]], axis=1))
    image_units = frame_normalize(image_units)
    return image_units

@torch.no_grad()
def gen_seq_event_frame_video(model, image_paths, output_name='temp', infer_type='center', 
                              seq_len=16, out_folder='./result_videos', width=346, height=260, 
                              verbose=False, fps=10, ceil=None):
    sequence_num = np.ceil(len(image_paths)/seq_len).astype(int)
    starting_indexes = list(range(0, len(image_paths)//seq_len*seq_len, seq_len))
    if len(image_paths) % seq_len != 1:
        starting_indexes.append(len(image_paths)-(seq_len+1))
    overlap_frame_count = seq_len-(starting_indexes[-1]-starting_indexes[-2])

    if verbose:
        print(f'Found {len(image_paths)} images, divided into {sequence_num} sequences')
        print(f'Starting indexes: {starting_indexes}')
    
    all_pred_voxel = []
    all_pred_ef = []

    for seq_idx in range(len(starting_indexes)):
        starting_idx = starting_indexes[seq_idx]
        ending_idx = starting_idx + 17
        image_paths_seq = image_paths[starting_idx:ending_idx]
        print(f'Using images {starting_idx} to {ending_idx-1}')

        # Load rgb images as grayscale
        images = np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths_seq], axis=0)
        
        image_units = kitti_image_pre_processing(images, height=height)
        if infer_type == 'center':
            pred_voxel, pred_ef = infer_center_image_unit(model, image_units, width, verbose=verbose)
        elif infer_type == 'pano':
            pred_voxel, pred_ef = infer_pano_image_unit(model, image_units, width, verbose=verbose)        
        else:
            raise ValueError(f'Invalid infer_type {infer_type}')
            
        all_pred_voxel.append(pred_voxel.cpu().detach().numpy())
        all_pred_ef.append(pred_ef.cpu().detach().numpy())

    all_pred_voxel[-1] = all_pred_voxel[-1][-seq_len+overlap_frame_count-1:]
    all_pred_ef[-1] = all_pred_ef[-1][-seq_len+overlap_frame_count-1:]

    all_pred_voxel = np.concatenate(all_pred_voxel, axis=0)
    all_pred_ef = np.concatenate(all_pred_ef, axis=0)
    
    if verbose:
        print("predicted voxels shape: ", all_pred_voxel.shape)
        print("predicted ef shape: ", all_pred_ef.shape)

    # Write all_pred_ef into a mp4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not op.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    
    video_size = all_pred_ef.shape[2], all_pred_ef.shape[1]
    video = cv2.VideoWriter(op.join(out_folder, f'{infer_type}-{output_name}-pred_ef_gray.mp4'), fourcc, fps, video_size)
    for i in range(all_pred_ef.shape[0]):
        if ceil is None:
            frame = all_pred_ef[i]/all_pred_ef.max() 
        else:
            frame = np.clip(all_pred_ef[i]/ceil, 0, 1)
        frame = (frame*255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video.write(frame)
    video.release()
    return all_pred_voxel, all_pred_ef

def core(model, image_folder, out_folder, seq_len=16, infer_type='center', output_name='temp', width=346, height=260, verbose=False, fps=10, ceil=None, max_frame_num=1800):
    image_paths = sorted([op.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])[:max_frame_num] # 1800 is the max number of images in a sequence, which equals to 1 minute of 30FPS video
    sequence_num = np.ceil(len(image_paths)/seq_len).astype(int)
    print(f'Now processing {image_folder}, Found {len(image_paths)} images, divided into {sequence_num} sequences')

    # Generate the video
    pred_voxel, pred_ef = gen_seq_event_frame_video(model, image_paths, output_name=output_name, infer_type=infer_type, seq_len=16, 
                                out_folder=out_folder, width=width, height=height, verbose=verbose, fps=fps, ceil=ceil)
    return pred_voxel, pred_ef

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=16, help='Sequence length')
    parser.add_argument('--fps', type=int, default=30, help='FPS of the output video')
    parser.add_argument('--ceil', type=int, default=10, help='The ceiling of the ef value')
    # image_folder
    parser.add_argument('--image_folder', type=str, default='/tsukimi/v2ce-project/video_for_test/dash-cam-test-video', help='The folder containing the images to infer')
    parser.add_argument('--out_folder', type=str, default='/tsukimi/v2ce-project/video_for_test', help='The folder to save the output video')
    parser.add_argument('--infer_type', type=str, default='center', help='The type of inference, can be center or pano')
    parser.add_argument('--model_path', type=str, default='/tsukimi/v2ce-project/best_model_log/assets/v2ce_3d.pt', help='The path to the trained model')
    parser.add_argument('--out_name_suffix', type=str, default='', help='The suffix of the output video name')
    # max_frame_num
    parser.add_argument('--max_frame_num', type=int, default=1800, help='The maximum number of frames to process')
    args = parser.parse_args()

    # image_folder = '/tsukimi/v2ce-project/video_for_test/dash-cam-test-video'
    name = Path(args.image_folder).name
    # out_folder = '/tsukimi/v2ce-project/video_for_test'
    # out_folder = 'test'
    if not op.exists(args.out_folder):
        os.makedirs(args.out_folder, exist_ok=True)

    # Define the path to the trained model
    # model_path = '/tsukimi/v2ce-project/logs/2023_09_05_23_00_09_ablation2-no-match/checkpoints/best-epoch=91-val_BinaryMatchF1_sum_c=0.5372-val_BinaryMatch_raw=0.9705.ckpt'
    # Get the trained model
    model = get_trained_mode(model_path=args.model_path)#, batch_size=4)
    
    output_name=f'{name}-ceil_{args.ceil}-fps_{args.fps}' if args.out_name_suffix == '' else f'{name}-ceil_{args.ceil}-fps_{args.fps}-{args.out_name_suffix}'
    pred_voxel, pred_ef = core(model, args.image_folder, args.out_folder, 
                               seq_len=args.seq_len, infer_type=args.infer_type, 
                               output_name=output_name, width=346, height=260, 
                               verbose=False, fps=args.fps, ceil=args.ceil, max_frame_num=args.max_frame_num)
    L, C, H, W = pred_voxel.shape
    stage2_input = pred_voxel.reshape(L, 2, 10, H, W)
    stage2_input = torch.from_numpy(stage2_input).cuda()
    
    ldati = partial(sample_voxel_statistical, fps=args.fps, bidirectional=False, additional_events_strategy='slope')
    event_stream_per_frame = ldati(stage2_input)

    # merge 16 event stream into a single one
    event_stream = []
    for i in range(L):
        # print(event_stream_per_frame[i]['timestamp'].min(), event_stream_per_frame[i]['timestamp'].max())
        event_stream_per_frame[i]['timestamp'] += int(i * 1 / args.fps * 1e6)
        event_stream.append(event_stream_per_frame[i])
    event_stream = np.concatenate(event_stream)
    
    # Dump the event stream (numpy structured array)
    event_stream_path = op.join(args.out_folder, f'{output_name}-events.npz')
    np.savez(event_stream_path, event_stream=event_stream)

    """
    # KITTI dataset
    out_folder = '/tsukimi/v2ce-project/result_videos'
    seq_names = ['image_00'] #, 'image_01']#, 'image_02', 'image_03']
    dataset_root = '/tsukimi/v2ce-project/kitti/2011_09_26/2011_09_26_drive_0005_sync'
    data_name = Path(dataset_root).name    
    for seq_name in seq_names:
        image_folder = op.join(dataset_root, seq_name, 'data')
        core(model, image_folder, out_folder, seq_len=seq_len)
    """
