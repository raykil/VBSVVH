# hbb-run3

## Setup environment
Two important things to make sure every time you are working in your analysis:
- Ensure you have a valid grid certificate:
     You can check that with `voms-proxy-info -all` and start one with `voms-proxy-init --rfc --voms cms -valid 192:00`.
- It is good practice to always run your analysis within a dedicated virtual environment to isolate project-specific dependencies and ensure reproducibility. Instructions on how to start the environment for this analysis below.


**Virtual environment**:

The instructions below will do the following:

- Download the micromamba setup script (change if needed for your machine https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- Install: (the micromamba directory can end up taking O(1-10GB) so make sure the directory you're using allows that quota)
    - Note: If on lpc cluster: install micromamba in `nobackup` area.

```
# Download and execute install script
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# You may need to restart your shell
```

Here is an example output:
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  3059  100  3059    0     0   3196      0 --:--:-- --:--:-- --:--:--  3196
Micromamba binary folder? [~/.local/bin] ~/nobackup/micromamba
Init shell (bash)? [Y/n] Y
Configure conda-forge? [Y/n] y
Running `shell init`, which:
 - modifies RC file: "/uscms/home/cmantill/.bashrc"
 - generates config for root prefix: "/uscms_data/d3/cmantill/micromamba"
 - sets mamba executable to: "/uscms_data/d3/cmantill/y/micromamba"
The following has been added in your "/uscms/home/cmantill/.bashrc" file

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'micromamba shell init' !!
export MAMBA_EXE='/uscms_data/d3/cmantill/y/micromamba';
export MAMBA_ROOT_PREFIX='/uscms_data/d3/cmantill/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<
```

Then create an environment:
```
micromamba create -n hbb python=3.10 root -c conda-forge
micromamba activate hbb
# install ipykernel for running jupyter notebooks
micromamba install  -n hbb ipykernel
```

Install requirements (see note on lpc below):
```
# Perform an editable installation
pip install -e .
# install requirements
pip install -r requirements.txt
```

## Workflow

### 0. Environment setup
``` bash
# One-time thing
conda create -y -n hbb python=3.10 root -c conda-forge
conda activate hbb
pip install -e . -r requirements.txt
```

``` bash
# Every login
conda activate hbb
voms-proxy-init --rfc --voms cms -valid 192:00
```

### 1. Run the categorizer
`src/hbb/processors/categorizer.py` is a coffea processor that 1) takes in NanoAOD files, 2) apply selection/corrections, 3) evaluates ABCD NN score, 4) sorts events into regions (wwh, zzh_1FJ, zzh_2FJ), and 5) return 1 parquet per region and a cutflow pkl.

`local_hvv.sh` ->`local_hvv.py` → `src/run.py` → `src/hbb/processors/categorizer.py`
``` bash
# STEP 1
python local_hvv.py -c run2_1FJ -r
python local_hvv.py -c run2_2FJ -r
python local_hvv.py -c run3_1FJ -r
python local_hvv.py -c run3_2FJ -r
```
`src/run.py` is run per dataset, i.e., keys in `sample_json/samples_run*_2L_*FJ.json`. Output is stored in `output_run*_*FJ/{year}/{dataset}`.

### 2. Create histograms
```bash
# STEP 2
python python/make_histos.py -c $channel -y $year -r $region
```
Run `python python/make_histos.py -h` to see options. Output are 1) pickles and 2) plots in `histograms/run*_*FJ/{year}/signal_*`. Pickles contain cutflow. To look at the content, run:
```bash
python inspect_cutflow.py -c $channel -y $year -r $region
```

### 3. Draw histograms
```bash
# STEP 3
python python/plot_histos.py
```