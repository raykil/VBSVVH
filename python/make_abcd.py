#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hist
import numpy as np
from common import common_mc, data_by_year

from hbb import utils
import rhalphalib as rl
import re


def fill_h_ABCD(h_ABCD, events, process, dataset_name, year, isData=False):
    """
    Fills a histogram with events from a single dataset.
    """

    for _process_name, data in events.items():
        weight_val = data["finalWeight"].astype(float)
        if isData:
            # print("DATA HOORAY")
            weight_val = data['weight_noxsec'].astype(float)

        # Event selection
        Txbb = data["HiggsAK8_ParTPXbbVsQCD"]
        msd = data["HiggsAK8_msd"]
        pt = data["HiggsAK8_pt"]
        mjj = data["VBFPair_mjj"]
        deta = data["VBFPair_deta"]
        ll_flav = data["LeadingLep_flavor"]
        sl_flav = data["SubLeadingLep_flavor"]
        pre_selection = (msd >= 40) & (pt >= 250) # & (mjj > 250) & (deta > 2.5)

        all_selections = {
            "preselection": pre_selection,
            "hbb_score_0p1": (Txbb > 0.1),
            "vbf_deta_2p5": (deta > 2.5),
            "vbf_mjj_250": (mjj > 250),
            "same_flavor": (ll_flav==sl_flav),
            "opposite_flavor": (ll_flav!=sl_flav),
            "both_electrons": (ll_flav==sl_flav) & (ll_flav == 1.),
            "both_muons": (ll_flav==sl_flav) & (ll_flav == 0.)
        }

        selection_dict = {
            "preselection": ["preselection"],
            "preselection_ee": ["preselection", "both_electrons"],
            "preselection_mumu": ["preselection", "both_muons"],
            "preselection_emu": ["preselection", "opposite_flavor"],
            "signal_region": ["preselection", "hbb_score_0p1", "vbf_deta_2p5", "vbf_mjj_250"],
            "A": ["preselection"],
            "B": ["preselection"],
            "C": ["preselection"],
            "D": ["preselection"],
        }

        # Fill histograms
        for cat in ["A", "B", "C", "D"]:
            selection_list = selection_dict[cat]
            for i, selection in enumerate(selection_list):
                if i==0:
                    full_selection = all_selections[selection]
                else:
                    full_selection = full_selection & all_selections[selection]

            if len(pt[full_selection]) < 1:
                continue

            h_ABCD.fill(
                dataset = dataset_name,
                process = process,
                category = cat,
                h_pt = pt[full_selection],
                weight=weight_val[full_selection]
            )



def main(args):
    year = args.year
    region = args.region

    MAIN_DIR = "/eos/uscms/store/group/lpchbbrun3/"
    # dir_name = "lzygala/hvv_26May11/merged_2lep_1FJ_r3_2lep_1FJ_20260506174059/"
    # dir_name = "lzygala/hvv_26May11/merged_2lep_1FJ_r2_2lep_1FJ_20260506173848/"
    # dir_name = "lzygala/hvv_26May11/merged_2lep_2FJ_r2_2lep_2FJ_20260608142908/"
    dir_name = "lzygala/hvv_26May11/merged_2lep_2FJ_r3_2lep_2FJ_20260608143322/"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"

    load_columns = [
        "weight",
        "HiggsAK8_pt",
        "HiggsAK8_msd",
        "HiggsAK8_ParTPXbbVsQCD",
        "VAK8_pt",
        "VAK8_msd",
        "VBFPair_mjj",
        "VBFPair_deta",
        'weight_noxsec',
        'LeadingLep_flavor',
        'SubLeadingLep_flavor'
    ]
    filters = None

    data_dir = Path(path_to_dir) / year
    samples = {
        "data": data_by_year[year],
        **common_mc[year],
    }

    h_ABCD = hist.Hist(
        hist.axis.StrCategory([], growth=True, name="process", label="Process"),
        hist.axis.StrCategory([], growth=True, name="category", label="Category"),
        hist.axis.StrCategory([], growth=True, name="dataset", label="Dataset"),
        hist.axis.Regular(1, 0, 10000, name="h_pt", label="Higgs AK8 pt"),
        storage=hist.storage.Weight()
    )
    counter = 0

    # Loop through each process
    for process, datasets in samples.items():
        print(f"Processing {process} for year {year}...")


        # Loop through each dataset within the process
        for dataset in datasets:
            # Load only one dataset at a time to save memory
            counter = counter + 1

            events = utils.load_samples(
                data_dir,
                {process: [dataset]},  # Pass a list with a single dataset
                columns=load_columns,
                region=region,
                filters=filters,
            )

            if not events:
                print(f"No events found for dataset {dataset} in year {year}. Skipping.")
                continue


            # Fill the histogram with the events from this single dataset
            fill_h_ABCD(
                h_ABCD=h_ABCD,
                events=events, 
                process=process,
                dataset_name=dataset,
                year=year,
                isData="_Run20" in dataset,
                )


    # ---------------------------------------------------------------------------------

    model = rl.Model(f"srModel_{year}")

    for category in ["A", "B", "C", "D"]:
        ch_name = f"{category}{year}"
        ch = rl.Channel(ch_name)
        model.addChannel(ch)

        h_cat = h_ABCD[{"category": category}]

        for proc in h_cat.axes["process"]:
            h_proc = h_cat[{"process": proc}][{"dataset": sum}]

            template = (h_proc.values(), np.array([0.0, 1.0]), "onebin", h_proc.variances() ** 2)

            if proc == "data":
                if category == "A":
                    template = (np.array([0.0]), np.array([0.0, 1.0]), "onebin", np.array([0.0]))
                ch.setObservation(template, read_sumw2=True)
            else:

                s_type = rl.Sample.SIGNAL if "c2v" in proc else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(f"{ch_name}_{proc}", s_type, template)
                sample.autoMCStats(lnN=True)
                ch.addSample(sample)

            # print(h_proc.values())
            # print(h_proc.variances())

    with Path(f"srModel_{year}.pkl").open("wb") as fout:
        pickle.dump(model, fout)
    modeldir = f"srModel_{year}"
    model.renderCombine(modeldir)
    print(f"Datacards saved to {modeldir}")

    out_cards = ""
    for ch in model:
        if "/" in ch.name:
            continue
        out_cards += f"{ch.name}={ch.name}.txt "

        with Path(f"{modeldir}/{ch.name}.txt").open("r") as f:
            lines = f.readlines()
        with Path(f"{modeldir}/{ch.name}.txt").open("w") as f:
            for line in lines:
                if not line.startswith("shapes"):
                    f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make histograms for a given year.")
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2018", "2016", "2016APV", "2017","2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument(
        "--region",
        help="region",
        type=str,
        required=True,
        choices=[
            "signal_wwh",
            "signal_zzh_1FJ",
            "signal_wzh_zzh_2FJ"
        ],
    )
    parser.add_argument(
        "--outdir", help="Output directory to save histograms.", type=str, default="histograms"
    )
    args = parser.parse_args()

    main(args)
