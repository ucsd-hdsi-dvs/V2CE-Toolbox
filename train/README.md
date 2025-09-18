# V2CE Training/Testing Code

**Notice: This code base is the raw training/testing code without further organizing and cleaning. Please use by exploring the code yourself with the help of this readme file. You may find the training/testing command listed below, as well as the default configurations in the `main.py` file helpful during your exploration. If you find any conflict between files in this subdirectory and files in the root directory, the later should be used as reference. Enjoy.**

## File Structure

- `main.py`: main script for training and testing.
- `scripts`: scripts for training and testing.
- `scripts/models`: model definitions.
- `scripts/data`: dataset classes.
- `scripts/stage2`: scripts for stage2.
- `scripts/utils`: utility functions.
- `scripts/tools`: independent tools serving various purposes.

## Important Files

- `scripts/models/model_interface.py`: Where the training logic locates. You can find losses, metrics, and evaluation outer logic here. This is the entrance interface for any model and training/testing logic-related operations.
- `scripts/models/metrics.py`: Where you can find metrics definitions. But full metrics calculation method should be viewed together with the  `configure_metrics` function in file `scripts/models/model_interface.py`
- `scripts/models/losses.py`: Where you can find losses definitions. But full metrics calculation method should be viewed together with the  `configure_loss` function in file `scripts/models/model_interface.py`
- `scripts/data/data_interface.py`: Where you can find data-related operations like pre-processing. This is the entrance interface for any data-related operations.

## Run training

```bash
python main.py --data_dir=/tsukimi/datasets/MVSEC/event_chunks_10t --model_name=v2ce_3d --batch_size=2 --loss pyramid gan ef compensation ef_splitp pt --log_frequency=8 --lr=1e-3 --ef_type=c+cl --alpha_efc=5 --alpha_pyramid=1000  --alpha_ef=0.5 --alpha_gan=1 --alpha_compensation=1 --alpha_match=0.5 --alpha_pt=1 --unet_all_residual=True --exp_name=debug
```

or,

```bash
python main.py --data_dir=/tsukimi/datasets/MVSEC/event_chunks_10t --model_name=v2ce_3d --batch_size=2 --exp_name=debug
```

## Run test and record results

```bash
python main.py --data_dir=/tsukimi/datasets/MVSEC/event_chunks_10t --model_name=v2ce_3d --batch_size=2 --load_dir=/tsukimi/v2ce-project/logs/2023_09_05_23_00_09_ablation2-no-match/checkpoints --recorder_types test --test
_only --load_weights_only --load_best
```

Also, you will find it useful to checkout the script `scripts/tools/predictor.py` for more details on prediction of individual video or image sequences.
