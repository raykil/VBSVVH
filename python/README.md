# Hbb Analysis Plotting

This repository contains scripts to produce histograms from skimmed parquet files and generate various analysis plots.

## 1. Making Histograms from Parquet Files

The `make_histos.py` script reads the skimmed parquet files for a given year and region, applies event selections, and saves the resulting histograms into a `.pkl` file.

### Usage

Run the script specifying the year, region, and an output directory for the `.pkl` files.

**Example:**
```bash
python python/make_histos.py --year 2022EE --region signal_wwh --outdir histograms/
```
## 2. Plotting from Histograms


```bash
python python/plot_histos.py --year 2024 --region signal_wwh --indir histograms/25Aug27 --outdir plots/

```
