"""
This script is used to split video to image sequence.
"""
import os
import argparse
import cv2
import os.path as op


def split_video(video_path, output_dir):
    """
    Split video to image sequence.
    Args:
        video_path: path to the video
        output_dir: path to the output directory
    """
    if not op.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Load Video
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_name = f'{frame_idx:010d}.png'
        frame_path = op.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split video to image sequence.')
    parser.add_argument('--video_path', type=str, help='path to the video')
    parser.add_argument('--output_dir', type=str, help='path to the output directory')
    args = parser.parse_args()
    
    # If the output directory is not specified, use the same directory as the video.
    if args.output_dir is None:
        args.output_dir = op.dirname(args.video_path)
    
    split_video(args.video_path, args.output_dir)