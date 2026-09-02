"""
- Inspects 'histograms/run*_*FJ/20**/signal_*/pickles/*.pkl' files and report cutflow yields after each stage.
- Format example) signal_region_emu    3.912    1.641   0.9025   0.7258   0.3555
- selection_dict = {
    "preselection"      : ["preselection"],
    "preselection_hbb"  : ["preselection", "hbb_score_0p1"],
    "preselection_ee"   : ["preselection", "both_electrons"],
    "preselection_mumu" : ["preselection", "both_muons"],
    "preselection_emu"  : ["preselection", "opposite_flavor"],
    "signal_region"     : ["preselection", "hbb_score_0p1", "vbf_deta_2p5", "vbf_mjj_250"],
    "signal_region_ee"  : ["preselection", "hbb_score_0p1", "vbf_deta_2p5", "vbf_mjj_250", "both_electrons"],
    "signal_region_mumu": ["preselection", "hbb_score_0p1", "vbf_deta_2p5", "vbf_mjj_250", "both_muons"],
    "signal_region_emu" : ["preselection", "hbb_score_0p1", "vbf_deta_2p5", "vbf_mjj_250", "opposite_flavor"]
}
"""
import argparse, os, sys, pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/python")
from common import common_mc, data_by_year

def prettyPrint(yields):
    ncuts = np.flatnonzero(yields).max(initial=0) + 1 # drop the unused tail of the 15-slot cut axis
    print("".join(f"{y:12.4g}" for y in yields[:ncuts]))

def group_pickles(process, channel, years, region):
    total = 0
    for year in years:
        datasets = data_by_year[year] if process == "data" else common_mc[year][process]
        for dataset in datasets:
            path = Path(f"histograms/{channel}/{year}/{region}/pickles/cutflow_{year}_{region}_{dataset}.pkl")
            if path.exists(): total += pickle.loads(path.read_bytes())
    return total

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Print grouped cutflows from the per-dataset pickles.")
    parser.add_argument('-c', "--channel"  , type=str, required=True, choices=["run2_1FJ", "run2_2FJ", "run3_1FJ", "run3_2FJ"])
    parser.add_argument('-y', "--year"     , type=str, nargs="+"    , default=[], help="defaults to every year available for the channel")
    parser.add_argument('-r', "--region"   , type=str, required=True, choices=["signal_wwh", "signal_wzh_zzh_2FJ", "signal_zzh_1FJ"])
    parser.add_argument('-p', "--processes", type=str, nargs="+"    , default=[], help="keys of common_mc; defaults to every process of the first year")
    args = parser.parse_args()

    if not args.year: args.year = sorted(d.name for d in Path(f"histograms/{args.channel}").iterdir() if d.is_dir())
    if not args.processes: args.processes = [*common_mc[args.year[0]], "data"]
    print(f"Channel: {args.channel}   //   Region: {args.region}   //   available year(s): {' '.join(args.year)}")

    for process in args.processes:
        data = group_pickles(process, args.channel, args.year, args.region) # <class 'hist.hist.Hist'>
        print(f"\n---------- {process} ----------")
        # there are 5 axes in data (data.axes <class 'hist.axestuple.NamedAxesTuple'>)
        # [0,1,2]: StrCategory (region, category, dataset), [3,4]: Regular (h_pt, cut)
        
        for category in data.axes["category"]: # keys in selection_dict (9 categories)
            yields = data[:, category, :, :, :].project("cut").values()
            print(f"{category:<20}", end="")
            prettyPrint(yields)