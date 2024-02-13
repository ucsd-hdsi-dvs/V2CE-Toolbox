# V2CE Toolbox

## Introduction

This toolbox is designed to help you to convert your RGB or gray-scale videos to event streams. It is a official implementation of the paper "[V2CE: Video to Continuous Events Simulator](https://arxiv.org/abs/2309.08891)". The toolbox is a release version of the original code used in the paper. It is ready to use and convert your videos or image sequences to event streams. The toolbox is written in Python and is based on the Pytorch. You can control the inference speed manually based on your GPU configuration by changing the batch size. The toolbox is designed to be user-friendly and easy to use. It is also designed to be easily integrated into other projects. The toolbox is released under the MIT license.

Our model is trained on [MVSEC](https://daniilidis-group.github.io/mvsec/) dataset which uses DAVIS 346B cameras which have a resolution of 346x260 pixels. Therefore, the best performance is achieved when the input video has a resolution of 346x260 pixels. However, this toolbox can handle videos with different resolutions by automatically resize the height to 260 pixels and crop the width to 346 pixels (`--infer_type=center`), or infer the whole width by dividing the width into 346 pixels videos and merge the results. (`--infer_type=pano`).

## Usage

Please download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1-aC6CTGZgAZk3snANZ46FAGNkPzu_Scw/view?usp=sharing), and put it into the `weights` folder.


Command Template:

```bash
python v2ce.py --out_name_suffix=<YOUR_NAME_SUFFIX> --max_frame_num=<YOUR_DESIRED_MAX_INFERENCE_FRAME_NUMBER> --infer_type=<center/pano> -i '<YOUR_INPUT_VIDEO_PATH>' -b 4 --write_event_frame_video -l info
```

An example command:

```bash
python v2ce.py --out_name_suffix='release' --max_frame_num=321 --infer_type=center -i '/tsukimi/v2ce-project/video_for_test/dash-cam-test-video.mp4' -b 4 --write_event_frame_video -l info
```

More details can be found in the help message:

```bash
python v2ce.py -h
```

