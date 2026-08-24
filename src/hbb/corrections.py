"""
Collection of utilities for corrections and systematics in processors.

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import contextlib

import awkward as ak
import dask_awkward as dak
import numpy as np
import correctionlib
import pickle
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)
package_path = str(pathlib.Path(__file__).parent.parent.resolve())

# Important Run3 start of Run
FirstRun_2022C = 355794
FirstRun_2022D = 357487
LastRun_2022D = 359021
FirstRun_2022E = 359022
LastRun_2022F = 362180

"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms-griddata.cern.ch/cat/metadata/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "muon_pt" : ["MUO", "muon_scalesmearing.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "photon": ["EGM", "photon.json.gz"],
    "photon2024": ["EGM", "photonID_v1.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "fatjet_jec": ["JME", "fatJet_jerc.json.gz"],
    "jet_jec": ["JME", "jet_jerc.json.gz"],
    "jetveto": ["JME", "jetvetomaps.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
    "jetid" : ["JME", "jetid.json.gz"],
}

years = {
    "2017": "Run2-2017-UL-NanoAODv15",
    "2018": "Run2-2018-UL-NanoAODv15",
    "2022": "Run3-22CDSep23-Summer22-NanoAODv12",
    "2022EE": "Run3-22EFGSep23-Summer22EE-NanoAODv12",
    "2023": "Run3-23CSep23-Summer23-NanoAODv12",
    "2023BPix": "Run3-23DSep23-Summer23BPix-NanoAODv12",
    "2024": "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15",
}


def ak_clip(arr: ak.Array, min_value: float, max_value: float):
    """
    Clip the values of an awkward array using where
    """
    return ak.where(arr < min_value, min_value, ak.where(arr > max_value, max_value, arr))


def get_pog_json(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = years[year]

    return f"{pog_correction_path}/{pog_json[0]}/{year}/latest/{pog_json[1]}"


def add_pileup_weight(weights: Weights, year: str, nPU):
    # clip nPU from 0 to 150
    nPU = ak_clip(nPU, 0, 150)

    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3
    values = {}

    if not year == "2024":
        cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))
    else:
        pog_json_file = f"{package_path}/hbb/data/puWeights_2024.json"
        cset = correctionlib.CorrectionSet.from_file(pog_json_file)

    corr = {
        "2018": "Collisions18_UltraLegacy_goldenJSON",
        "2022": "Collisions2022_355100_357900_eraBCD_GoldenJson",
        "2022EE": "Collisions2022_359022_362760_eraEFG_GoldenJson",
        "2023": "Collisions2023_366403_369802_eraBC_GoldenJson",
        "2023BPix": "Collisions2023_369803_370790_eraD_GoldenJson",
        "2024": "Pileup",
    }[year]
    # evaluate and clip up to 4 to avoid large weights
    values["nominal"] = ak_clip(cset[corr].evaluate(nPU, "nominal"), 0, 4)
    values["up"] = ak_clip(cset[corr].evaluate(nPU, "up"), 0, 4)
    values["down"] = ak_clip(cset[corr].evaluate(nPU, "down"), 0, 4)

    weights.add("pileup", values["nominal"], values["up"], values["down"])

def add_pdf_weight(weights: Weights, pdf_weights):
    """
    Apply pdf weight variation for standard Hessian set
    """

    nom = ak.ones_like(weights.weight())
    if pdf_weights is None:
        weights.add('PDF_weight', nom)
        weights.add('aS_weight', nom)
        weights.add('PDFaS_weight', nom)
        return
                               
    arg = pdf_weights[:,1:-2]-(ak.ones_like(weights.weight())[:, None] * ak.Array(np.ones(100)))
    summed = ak.sum(np.square(arg),axis=1)
    pdf_unc = np.sqrt( (1./99.) * summed )
    weights.add('PDF_weight', nom, pdf_unc + nom)

    # alpha_S weights
    as_unc = 0.5*(pdf_weights[:,102] - pdf_weights[:,101])
    weights.add('aS_weight', nom, as_unc + nom)

    # PDF + alpha_S weights
    pdfas_unc = np.sqrt( np.square(pdf_unc) + np.square(as_unc) )
    weights.add('PDFaS_weight', nom, pdfas_unc + nom) 

def add_ps_weight(weights: Weights, ps_weights):
    """
    Parton Shower Weights (FSR and ISR)
    """
    nom = ak.ones_like(weights.weight())

    up_isr = ak.ones_like(nom)
    down_isr = ak.ones_like(nom)
    up_fsr = ak.ones_like(nom)
    down_fsr = ak.ones_like(nom)

    if ak.num(ps_weights[0], axis=0) == 4:
        up_isr = ps_weights[:, 0]  # ISR=2, FSR=1
        down_isr = ps_weights[:, 2]  # ISR=0.5, FSR=1

        up_fsr = ps_weights[:, 1]  # ISR=1, FSR=2
        down_fsr = ps_weights[:, 3]  # ISR=1, FSR=0.5

    elif ak.num(ps_weights[0], axis=0) > 1:
        print("PS weight vector has length ", ak.num(ps_weights[0]))

    weights.add("ISRPartonShower", nom, up_isr, down_isr)
    weights.add("FSRPartonShower", nom, up_fsr, down_fsr)

def add_scalevar_7pt(weights: Weights, var_weights):
    """
    QCD scale variations for the case muF = muR
    For application to high pt ggf and ttH higgs production mc
    Recommendation by LHCXSWG cds.cern.ch/record/2669113
    """
    nom   = ak.ones_like(weights.weight())
    up    = ak.ones_like(nom)
    down  = ak.ones_like(nom)

    if var_weights is None:
        weights.add('scalevar_7pt', nom)
        return
 
    try:
        selected = var_weights[:, [0, 1, 3, 5, 7, 8]]
        up = ak.max(selected, axis=1)
        down = ak.min(selected, axis=1)
    except Exception as e:
        print("Scale variation structure unexpected:", e)

    weights.add('scalevar_7pt', nom, up, down)

def add_scalevar_3pt(weights: Weights, var_weights):
    """
    QCD scale variations for the case muF^2 = muR^2
    For application to high pt VBF and VH higgs production mc
    Recommendation by LHCXSWG cds.cern.ch/record/2669113
    """
    nom   = ak.ones_like(weights.weight())
    up    = ak.ones_like(nom)
    down  = ak.ones_like(nom)

    if var_weights is None:
        weights.add('scalevar_3pt', nom)
        return

    try:
        selected = var_weights[:, [0, 8]]
        up = ak.max(selected, axis=1)
        down = ak.min(selected, axis=1)
    except Exception as e:
        print("Scale variation structure unexpected:", e)

    weights.add('scalevar_3pt', nom, up, down)
