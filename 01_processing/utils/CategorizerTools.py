"""
This v1 contains ABCD score inference logic.
As of 261105, I will separate inference from the categorizer, 
use infer.py to make inference, and use HVV_2L_RDF_parq files as inputs of categorizer.
This is CategorizerTools for Categorizer that accepts root files as input.
"""

import re, torch
from pathlib import Path
import numpy as np
from hist.dask import Hist
import awkward as ak
import dask_awkward as dak

from utils.GenSelection import (gen_selection_Hbb, gen_selection_V, gen_selection_Vg)
from utils.abcd_model import *

from coffea.analysis_tools import PackedSelection, Weights

scriptPath = Path(__file__).resolve().parent
# makeOutput = lambda: {
#     "skim"      : {},
#     "btagWeight": Hist.new.Reg(50, 0, 3, name="val", label="BTag correction").Weight(),
#     "cutflow"   : Hist.new.StrCat([], growth=True, name="region", label="Region").StrCat([], growth=True, name="dataset", label="Dataset")
#                           .Reg(4, 0, 10000, name="h_pt", label="Higgs AK8 pt").Reg(15, 0, 15, name="cut", label="Cut index").Weight(),
# }
# makeBTagOutput = lambda: (
#     Hist.new.StrCat([], growth=True, name="tagger", label="Tagger").Reg(2, 0, 2, name="passWP", label="passWP")
#             .Variable([0, 4, 5], name="flavor", label="Jet hadronFlavour").Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt", label="Jet pt")
#             .Reg(4, 0, 2.5, name="abseta", label="Jet abseta").Weight()
# )
# gen_selection_dict = {
#     "Hto2B": gen_selection_Hbb,
#     "Wto2Q-": gen_selection_V,
#     "Zto2Q-": gen_selection_V,
#     "ZGto2QG-": gen_selection_Vg,
# }

def fieldsToSave(events):
    # columns read by python/make_abcd.py and make_histos.py (+ axis_info.py).
    FieldsDict = {
        'L1_pt'      : events.LeadLep.pt,
        'L1_flav'    : events.LeadLep.pdgId,
        'L2_pt'      : events.SubleadLep.pt,
        'L2_flav'    : events.SubleadLep.pdgId,
        'dilep_mass' : events.LeptonInfo.dilepMass,
        'HiggsFJ_pt' : events.HiggsFJ.pt,
        'HiggsFJ_msd': events.HiggsFJ.msd,
        'HbbScore'   : events.HiggsFJ.ParTPXbbVsQCD,
        'VFJ_pt'     : events.VFJ.pt,  # None in 1FJ
        'VFJ_msd'    : events.VFJ.msd, # None in 1FJ
        'VBS_Mjj'    : events.vbs.mjj,
        'VBS_deta'   : events.vbs.detajj,
        'VBS_score'  : events.vbs.score,
        'MET'        : events.met.pt,
        'ABCD_score' : events.ABCD_score,
        'weight'     : events.weight, # preselection weight * WEIGHTS, set in categorizer.py
    }
    return FieldsDict

def makeCutflow(selection, cutlist, weight):
    # weighted yield after each cumulative cut, as in working_categorizer: cut=0 is every event, cut=i after cutlist[:i]
    cutflow = Hist.new.Integer(0, len(cutlist) + 1, name="cut", label="Cut index").Weight()
    for i in range(len(cutlist) + 1):
        passed = weight[selection.all(*cutlist[:i])] if i else weight # all() with no names requires every selection, not none
        cutflow.fill(cut=ak.full_like(passed, i, dtype=int), weight=passed)
    return cutflow

# def load_triggers():
#     with Path(f"{scriptPath}/../../src/hbb/dilep_triggers.json").open() as f:
#         triggers = json.load(f)
#     return triggers

def load_abcd_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    toFeatureMap = lambda name: re.sub(r"^(L|FJ)([12])_(.+)", lambda m: f"{'lepton' if m[1] == 'L' else 'fatjet'}_{m[3]}_{m[2]}", name) # training's parquet names (L1_pt, FJ2_tau1) -> feature_map's (lepton_pt_1, fatjet_tau1_2)
    inputs = {toFeatureMap(name): spec for name, spec in checkpoint["inputs"].items()} # training's transform and min/max per input, in column order
    model = ABCDModel(
        input_size       = len(inputs),
        hidden_layers    = [64, 32, 16],
        learning_rate    = 0.001,
        bce_weight       = 1.0,
        disco_lambda     = 10.0,
        flavor           = 'single',
        use_batchnorm    = True,
        dropout          = 0.2,
        weight_decay     = 0.01,
        label_smoothing  = 0.0,
        use_lr_scheduler = True,
        lr_scheduler_patience = 5,
        lr_scheduler_factor   = 0.5,
        lr_scheduler_min_lr   = 1e-6,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.inputs = inputs
    model.lo = np.array([v["min"] for v in inputs.values()], dtype=np.float32)
    model.hi = np.array([v["max"] for v in inputs.values()], dtype=np.float32)
    return model

def load_abcd_models(year, nFJ):
    """Built on the manager so the models themselves, not their paths, travel with the task. Keyed by signal point."""
    run = "run2" if year.startswith("201") else "run3"
    paths = sorted((Path(__file__).resolve().parents[1] / "models").glob(f"*/{run}_2L_{nFJ}FJ_best_model.pt"))
    return {p.parent.name: load_abcd_model(p) for p in paths}

def eval_ABCD_model(events, models):
    inputs = next(iter(models.values())).inputs # every signal point's net takes the same inputs; only their min/max differ
    feature_map = {
        "lepton_pt_1"     : events.lepton.pt       ,    "lepton_eta_1"      : events.lepton.eta      , "lepton_phi_1": events.lepton.phi, "lepton_mass_1": events.lepton.mass,
        "lepton_pt_2"     : events.lepton.pt       ,    "lepton_eta_2"      : events.lepton.eta      , "lepton_phi_2": events.lepton.phi, "lepton_mass_2": events.lepton.mass,
        "met_phi"         : events.met.phi         ,    "met_pt"            : events.met.pt          ,
        "fatjet_pt"       : events.fatjet.pt       ,    "fatjet_eta"        : events.fatjet.eta      , "fatjet_phi"  : events.fatjet.phi, "fatjet_mass"  : events.fatjet.mass,
        "fatjet_pt_1"     : events.fatjet.pt       ,    "fatjet_eta_1"      : events.fatjet.eta      , "fatjet_phi_1": events.fatjet.phi, "fatjet_mass_1": events.fatjet.mass,
        "fatjet_pt_2"     : events.fatjet.pt       ,    "fatjet_eta_2"      : events.fatjet.eta      , "fatjet_phi_2": events.fatjet.phi, "fatjet_mass_2": events.fatjet.mass,
        "fatjet_tau1"     : events.fatjet.tau1     ,    "fatjet_tau2"       : events.fatjet["tau2"]  , # .tau2 is the Lorentz vector's proper time², i.e. mass²
        "fatjet_tau1_1"   : events.fatjet.tau1     ,    "fatjet_tau2_1"     : events.fatjet["tau2"]  ,
        "fatjet_tau1_2"   : events.fatjet.tau1     ,    "fatjet_tau2_2"     : events.fatjet["tau2"]  ,
        "fatjet_msoftdrop": events.fatjet.msoftdrop,    "fatjet_msoftdrop_1": events.fatjet.msoftdrop, "fatjet_msoftdrop_2": events.fatjet.msoftdrop,
        "fatjet_massGloParT3": events.fatjet.massGloParT3,    "fatjet_massGloParT3_1": events.fatjet.massGloParT3, "fatjet_massGloParT3_2": events.fatjet.massGloParT3,
        "ht_fatjets"      : events.ht.fatjets
        }
    if ak.backend(events.met.pt) == "typetracer":
        for arr in feature_map.values():
            arr.layout._touch_data(recursive=True)
        return {signal: ak.zeros_like(events.met.pt) for signal in models}  # correct shape/dtype, no real data
    if len(events.met.pt) == 0:
        return {signal: ak.zeros_like(events.met.pt) for signal in models}  # no events: nothing to feed the model

    columns = []
    for feature, spec in inputs.items():
        if feature.endswith("_2"):
            x = ak.to_numpy(ak.fill_none(ak.pad_none(feature_map[feature], 2, clip=True)[:, 1], 0)).astype(np.float32)
        elif feature_map[feature].ndim > 1: # _1, and 1FJ's unsuffixed fatjet inputs: skims keep a few events with no fatjet
            x = ak.to_numpy(ak.fill_none(ak.pad_none(feature_map[feature], 1, clip=True)[:, 0], 0)).astype(np.float32)
        else:
            x = ak.to_numpy(feature_map[feature]).astype(np.float32)
        if spec["transform"] == "log":
            if feature == "lepton_mass_1" or feature == "lepton_mass_2":
                print(f"{feature}: {(x < 0).sum()}/{len(x)} negative ({100 * (x < 0).mean():.1f}%), min={x.min(initial=0):.3f}, max={x.max(initial=0):.3f}, mean={x.mean():.3f}, std={x.std():.3f}")
            x = np.log1p(np.clip(x, 0, None)) # negatives are rounded-m² artifacts, physically ~0
        columns.append(x)
    X = np.column_stack(columns).astype(np.float32) # min-max comes from training, not this chunk
    with torch.no_grad():
        return {signal: torch.sigmoid(model(torch.from_numpy((X - model.lo) / (model.hi - model.lo)))).numpy().flatten() for signal, model in models.items()}

def add_abcd_scores(events, models):
    for signal, score in eval_ABCD_model(events, models).items():
        events = ak.with_field(events, ak.Array(score), f"ABCD_score_{signal}")
    return ak.with_field(events, events.ABCD_score_c2v1p0_c3_1p0, "ABCD_score") # SM point, under the pre-benchmark name

def addTriggerCut(self, events, selection):
    # Already done in run3-vbsvvh/preselection/src/selections.cpp#L680,700
    dilepTrigDict = {
        "2018"    : ["Ele30_WPTight_Gsf", "IsoMu24"],
        "2022"    : ["Ele30_WPTight_Gsf", "IsoMu24"],
        "2022EE"  : ["Ele30_WPTight_Gsf", "IsoMu24"],
        "2023"    : ["Ele30_WPTight_Gsf", "IsoMu24"],
        "2023BPix": ["Ele30_WPTight_Gsf", "IsoMu24"],
        "2024"    : ["Ele30_WPTight_Gsf", "IsoMu24"]
    }
    trigger = dak.zeros_like(events.run, dtype=bool)
    for path in dilepTrigDict[self._year]:
        if path in events.HLT.fields: trigger = trigger | events.HLT[path]
    selection.add("trigger", trigger)

def setVBFJets(events, AK4Jets):
    vbf_deta  = events.vbs.detajj # events.vbs defined in selections.cpp#L380-399.
    vbf_mjj   = events.vbs.mjj
    # vbf_score = events.vbs.score
    isVBSJet  = (AK4Jets.idx == events.vbs.jet1_idx) | (AK4Jets.idx == events.vbs.jet2_idx) # -1 in events.vbs.jet1(2)_idx if no match.
    isVBF     = (vbf_deta > 2.5) & (vbf_mjj > 750) & (ak.num(AK4Jets[isVBSJet], axis=1)==2) # both BDT-tagged VBS jets survive SelectGoodAK4Jets
    return isVBF

