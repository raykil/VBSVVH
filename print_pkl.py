import pickle
from pathlib import Path

import click
import hist
import pandas as pd

"""
Script to verify that histograms saved in pkl files and histograms produced from parquet files produce the same results
e.g.
python print_pkl.py 0-1.pkl signal-ggf_0-1.parquet
"""


@click.command()
@click.argument("filename")
@click.argument("parquetfile")
def print_pkl(filename, parquetfile):
    """
    Double check that histograms saved in pkl file and histogram file
    """
    with Path(filename).open("rb") as file:
        out_dict = pickle.load(file)

    key = next(iter(out_dict.keys()))
    sumw = next(iter(out_dict[key]["sumw"].values()))
    print("Total sum of gen weights ", sumw)

    region = parquetfile.split("_")[0]

    print(f"Histogram mass, for region {region}")
    try:
        hist_region = out_dict[key]["templates"][{"region": region}][
            {
                "systematic": "nominal",
                "dataset": key,
                "pnet1": sum,
                "mjj": sum,
                "pt1": sum,
                "genflavor": sum,
            }
        ]
        print(f"Loaded histogram for {key} from {filename}")
    except KeyError:
        print(f"KeyError for {region} in {filename}, skipping.")

    print(hist_region)
    print("Weighted histogram view ", hist_region.view())
    print("Weighted histogram, sum of weights", hist_region.sum())

    print("---- Loading parquet ----")
    load_columns = ["FatJet0_msd", "weight"]
    events = pd.read_parquet(parquetfile, columns=load_columns)

    msd_axis = hist.axis.Regular(23, 40, 201, name="msd1", label="Jet $m_{sd}$")
    hist_nonorm = hist.Hist(msd_axis)
    hist_nonorm.fill(events["FatJet0_msd"], weight=events["weight"])

    print(hist_nonorm)
    print("Weighted histogram view ", hist_nonorm.view())
    print("Weighted histogram, sum of weights", hist_nonorm.sum())

    hist_nonorm_scaled = hist_nonorm * 1 / sumw
    print("Norm Weighted histogram, sum of weights", hist_nonorm_scaled.sum())


if __name__ == "__main__":
    print_pkl()
