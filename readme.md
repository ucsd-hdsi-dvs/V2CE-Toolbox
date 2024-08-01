# V2CE Toolbox

## Introduction

This toolbox is designed to help you to convert your RGB or gray-scale videos to event streams. It is a official implementation of the paper "[V2CE: Video to Continuous Events Simulator](https://arxiv.org/abs/2309.08891)". The toolbox is a release version of the original code used in the paper. It is ready to use and convert your videos or image sequences to event streams. The toolbox is written in Python and is based on the Pytorch. You can control the inference speed manually based on your GPU configuration by changing the batch size. The toolbox is designed to be user-friendly and easy to use. It is also designed to be easily integrated into other projects. The toolbox is released under the MIT license.

Our model is trained on [MVSEC](https://daniilidis-group.github.io/mvsec/) dataset which uses DAVIS 346B cameras which have a resolution of 346x260 pixels. Therefore, the best performance is achieved when the input video has a resolution of 346x260 pixels. However, this toolbox can handle videos with different resolutions by automatically resize the height to 260 pixels and crop the width to 346 pixels (`--infer_type=center`), or infer the whole width by dividing the width into 346 pixels videos and merge the results(`--infer_type=pano`). You can also manually set the width and height of the input video by using the `--width` and `--height` arguments, but it is not recommended as it is not vigorously tested. You can set these parameters based on your own needs.

## Usage

Please download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1-aC6CTGZgAZk3snANZ46FAGNkPzu_Scw/view?usp=sharing), and put it into the `weights` folder.


Command Template:

```bash
python v2ce.py --out_name_suffix=<YOUR_NAME_SUFFIX> --max_frame_num=<YOUR_DESIRED_MAX_INFERENCE_FRAME_NUMBER> --infer_type=<center/pano> -i '<YOUR_INPUT_VIDEO_PATH>' -b 4 --write_event_frame_video -l info
```

An example command:

```bash
python v2ce.py --out_name_suffix=release --max_frame_num=321 --infer_type=center -i '/tsukimi/v2ce-project/video_for_test/dash-cam-test-video.mp4' -b 4 --write_event_frame_video -l info
```

More details can be found in the help message:

```bash
python v2ce.py -h
```

## Tips

- To brighten the generated event frame video, set a smaller --ceil parameter or a larger -u/--upper_bound_percentile parameter. Normalization to [0,1] is required for video generation, and outlier values can significantly affect the event frame's maximum value. The --ceil parameter fixes the maximum event frame value, while the -u/--upper_bound_percentile parameter dynamically sets the ceiling based on the specified percentile of nonzero event frame pixels. When both parameters are set, the program uses the smaller ceiling value for normalization, setting all values above the ceiling to 1. The default values are --ceil at 10 and -u/--upper_bound_percentile at 98.

- To set the --max_frame_num, if your input video has 30 FPS and you want the event frame video to cover the first 5 seconds, specify --max_frame_num as (30 * 5 + 1) = 151 frames. The additional frame accounts for the last 1/30 second, as event stream inference requires the frames before and after each time interval.
