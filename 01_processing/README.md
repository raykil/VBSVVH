
## Remote env setup
Tarball should be created to setup the hbb environemnt for each worker's sandbox.
Run the following one-time commands and keep `vbsvvh-env.tar.gz` in `01_processing` directory.

```bash
conda create -n vbsvvh-ship -c conda-forge python=3.11 ndcctools xrootd conda-pack pytorch-cpu
conda activate vbsvvh-ship
pip install coffea==2025.1.1 dask-awkward==2025.5.0 scikit-learn pytz # coffea imports pytz, but pandas 3.0 made it an optional extra
conda activate hbb
poncho_package_create vbsvvh-ship vbsvvh-env.tar.gz # may take several minutes. adds the bin/run_in_env wrapper that TaskVine needs
rm -rf $(conda info --base)/envs/vbsvvh-ship # much faster than conda env remove
```

## Submit jobs
Run
```bash

```