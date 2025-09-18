# Tools

- `event_chunk.py`: Divide a `.aedat` raw event stream (with frames and imu) file into chuncks with a sequence default to 16.
- `v2e_metric.py`: Calculate the metrics of V2E model.
- `esim_metric.py`: Calculate the metrics of ESIM model.
- `speed_test.py`: Test the speed of our V2CE model stage1.
- `predictor.py`: Load a trained model and provide functions to work on a image sequence or a whole video (KITTI-like video is supported for now). There are two inference mode: image center-only (horizontal), or pano inference.
- `gen_phy_att.py`: Generate physical attention for all data chunks in a folder.
- `MVSEC_data_utils.py`: Process the MVSEC dataset to event chunks we need.
- `dummy_data_gen.py`: Generate dummy data for testing.
- `time_voxel_stat_calc.py`: Calculate the time voxel statistics for a event chunk.
