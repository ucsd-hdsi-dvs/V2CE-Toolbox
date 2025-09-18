# README for stage2 scripts

- `sample_methods`: scripts for sampling methods.
- `sample_methods/LDATI.py`: Our proposed LDATI sampling method.
- `sample_methods/random_even_sample.py`: Random and even sampling methods.
- `sample_methods/pure_slope_sample.py`: Pure slope sampling method, no chain decoulping.
- `stage2_metrics.py`: Calculate metrics for stage2. It support LDATI, random, even, and pure slope sampling methods.
- `baseline_metrics.py`: Calculate metrics for baseline methods (Here the baseline methods means baselines in stage1).
- `performance_test.py`: Test the performance of LDATI sampling method on MVSEC dataset testing set.
- `vis_stage2.ipynb`: Randomly generate a series of events, pack them into voxels, and use different sampling methods to sample them. Then visualize the sampled events together with the original events.