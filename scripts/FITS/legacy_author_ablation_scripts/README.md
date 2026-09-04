# Legacy author ablation scripts

These are the original FITS shell scripts retained for historical reference.
They define broad hyperparameter grids and selected configurations from the
authors' CUDA-era experiments; they are not the supported Aurora launchers.

They are intentionally separated from `../aurora/` because they can override
the shared data root with `./dataset/`, set fixed device-affinity masks, reuse
checkpoint identifiers across configurations, and mix legacy logging paths.
Use the scripts in `../aurora/` for Aurora training and evaluation.
