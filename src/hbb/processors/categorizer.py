from __future__ import annotations

import time, json, torch, logging
from pathlib import Path

import awkward as ak
import dask_awkward as dak
import numpy as np
# import xgboost as xgb
from coffea.analysis_tools import PackedSelection, Weights
# from coffea.ml_tools import xgboost_wrapper
from hist.dask import Hist
from sklearn.preprocessing import MinMaxScaler

from hbb.processors.SkimmerABC import SkimmerABC

from .GenSelection import (bosonFlavor, gen_selection_Hbb, gen_selection_V, gen_selection_Vg, getBosons)
from .objects import (good_ak4jets, good_ak8jets, good_electrons, loose_muons, highpt_muons, set_ak4jets, set_ak8jets)
from .abcd_model import *
logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "Hto2B": gen_selection_Hbb,
    "Wto2Q-": gen_selection_V,
    "Zto2Q-": gen_selection_V,
    "ZGto2QG-": gen_selection_Vg,
}
_abcd_model_cache = {}

def _safe_minmax_scale(values):
    scaled = np.zeros_like(values, dtype=np.float64)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).ravel()
    return scaled

def eval_ABCD_model(events, nFJ=1, saved_model="best_model.pt"):
    if saved_model not in _abcd_model_cache:
        if nFJ == 1:
            model = ABCDModel(
                input_size=16,
                hidden_layers=[64, 32, 16],
                learning_rate=0.001,
                bce_weight=1.0,
                disco_lambda=10.0,
                flavor='single',
                use_batchnorm=True,
                dropout=0.2,
                weight_decay=0.01,
                label_smoothing=0.0,
                use_lr_scheduler=True,
                lr_scheduler_patience=5,
                lr_scheduler_factor=0.5,
                lr_scheduler_min_lr=1e-6,
            )
        else:   
            model = ABCDModel(
                input_size=25,
                hidden_layers=[64, 32, 16],
                learning_rate=0.001,
                bce_weight=1.0,
                disco_lambda=10.0,
                flavor='single',
                use_batchnorm=True,
                dropout=0.2,
                weight_decay=0.01,
                label_smoothing=0.0,
                use_lr_scheduler=True,
                lr_scheduler_patience=5,
                lr_scheduler_factor=0.5,
                lr_scheduler_min_lr=1e-6,
            )
        state_dict = torch.load(saved_model, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        _abcd_model_cache[saved_model] = model
    model = _abcd_model_cache[saved_model]
    if nFJ == 1:
        feature_transforms = {"lepton_pt_1":"log","lepton_pt_2":"log",
                        "lepton_eta_1":"minmax","lepton_eta_2":"minmax",
                        "lepton_phi_1":"minmax","lepton_phi_2":"minmax",
                        "lepton_mass_1":"log","lepton_mass_2":"log",
                        "met_phi":"minmax","met_pt":"log",
                        "fatjet_pt":"log","fatjet_eta":"minmax","fatjet_phi":"minmax","fatjet_mass":"log",
                        "fatjet_tau2":"none","fatjet_tau1":"none"}
    elif nFJ == 2:
        feature_transforms = {"lepton_pt_1":"log","lepton_pt_2":"log",
                        "lepton_eta_1":"minmax","lepton_eta_2":"minmax",
                        "lepton_phi_1":"minmax","lepton_phi_2":"minmax",
                        "lepton_mass_1":"log","lepton_mass_2":"log",
                        "met_phi":"minmax","met_pt":"log",
                        "fatjet_pt_1":"log","fatjet_pt_2":"log",
                        "fatjet_eta_1":"minmax","fatjet_eta_2":"minmax",
                        "fatjet_phi_1":"minmax","fatjet_phi_2":"minmax",
                        "fatjet_mass_1":"log", "fatjet_mass_2":"log",
                        "fatjet_msoftdrop_1":"log", "fatjet_msoftdrop_2":"log",
                        "fatjet_tau2_1":"none","fatjet_tau1_1":"none",
                        "fatjet_tau2_2":"none","fatjet_tau1_2":"none",
                        "ht_fatjets":"log"}
    else:
        raise ValueError(f"Invalid nFJ value: {nFJ}. Must be 1 or 2.")
    feature_map = {
        "lepton_pt_1":events.lepton.pt,"lepton_eta_1":events.lepton.eta,"lepton_phi_1":events.lepton.phi,"lepton_mass_1":events.lepton.mass,
        "lepton_pt_2":events.lepton.pt,"lepton_eta_2":events.lepton.eta,"lepton_phi_2":events.lepton.phi,"lepton_mass_2":events.lepton.mass,
        "met_phi":events.met.phi,"met_pt":events.met.pt,
        "fatjet_pt":events.fatjet.pt,"fatjet_eta":events.fatjet.eta,"fatjet_phi":events.fatjet.phi,"fatjet_mass":events.fatjet.mass,
        "fatjet_tau2":events.fatjet.tau2, "fatjet_tau1":events.fatjet.tau1,
        "fatjet_pt_1":events.fatjet.pt,"fatjet_eta_1":events.fatjet.eta,"fatjet_phi_1":events.fatjet.phi,"fatjet_mass_1":events.fatjet.mass,
        "fatjet_pt_2":events.fatjet.pt,"fatjet_eta_2":events.fatjet.eta,"fatjet_phi_2":events.fatjet.phi,"fatjet_mass_2":events.fatjet.mass,
        "fatjet_tau2_1":events.fatjet.tau2,"fatjet_tau1_1":events.fatjet.tau1,
        "fatjet_tau2_2":events.fatjet.tau2,"fatjet_tau1_2":events.fatjet.tau1,
        "fatjet_msoftdrop_1":events.fatjet.msoftdrop,"fatjet_msoftdrop_2":events.fatjet.msoftdrop,
        "ht_fatjets":events.ht.fatjets
        }
    if ak.backend(events.met.pt) == "typetracer":
        for arr in feature_map.values():
            arr.layout._touch_data(recursive=True)
        return ak.zeros_like(events.met.pt)  # correct shape/dtype, no real data

    features = list(feature_transforms.keys())
    columns = []
    for feature in features:
        if feature.endswith("_1"):
            x = ak.to_numpy(ak.fill_none(ak.pad_none(feature_map[feature], 1, clip=True)[:, 0], 0)).astype(np.float32)
        elif feature.endswith("_2"):
            x = ak.to_numpy(ak.fill_none(ak.pad_none(feature_map[feature], 2, clip=True)[:, 1], 0)).astype(np.float32)
        else:
            x = ak.to_numpy(feature_map[feature]).astype(np.float32)
        transform = feature_transforms[feature]
        if transform == "log":
            x = np.log1p(x)
            x = _safe_minmax_scale(x)
        elif transform == "minmax":
            x = _safe_minmax_scale(x)
        elif transform == "none":
            x = _safe_minmax_scale(x)
            pass
        columns.append(x)
    X = np.column_stack(columns).astype(np.float32)
    with torch.no_grad():
        x_tensor = torch.from_numpy(X)
        logits = model(x_tensor)
        scores = torch.sigmoid(logits).detach().cpu().numpy().flatten()
#    scores = dak.from_awkward(ak.Array(scores), npartitions)
#    print(scores)
    return scores
def add_abcd_score(events, nFJ=1, saved_model="best_model.pt"):
    score = eval_ABCD_model(events, nFJ, saved_model)
    return ak.with_field(events, ak.Array(score), "ABCD_score")
model_file = {}
model_file["2022"] = {}
model_file["2023"] = {}
model_file["2024"] = {}
model_file["2025"] = {}
model_file["2016APV"] = {}
model_file["2016"] = {}
model_file["2017"] = {}
model_file["2018"] = {}


model_file["2022"]["1"] = "src/hbb/data/run3_2L_1FJ_best_model.pt"
model_file["2022"]["2"] = "src/hbb/data/run3_2L_2FJ_best_model.pt"
model_file["2023"]["1"] = "src/hbb/data/run3_2L_1FJ_best_model.pt"
model_file["2023"]["2"] = "src/hbb/data/run3_2L_2FJ_best_model.pt"
model_file["2024"]["1"] = "src/hbb/data/run3_2L_1FJ_best_model.pt"
model_file["2024"]["2"] = "src/hbb/data/run3_2L_2FJ_best_model.pt"
model_file["2025"]["1"] = "src/hbb/data/run3_2L_1FJ_best_model.pt"
model_file["2025"]["2"] = "src/hbb/data/run3_2L_2FJ_best_model.pt"

model_file["2016APV"]["1"] = "src/hbb/data/run2_2L_1FJ_best_model.pt"
model_file["2016APV"]["2"] = "src/hbb/data/run2_2L_2FJ_best_model.pt"
model_file["2016"]["1"] = "src/hbb/data/run2_2L_1FJ_best_model.pt"
model_file["2016"]["2"] = "src/hbb/data/run2_2L_2FJ_best_model.pt"
model_file["2017"]["1"] = "src/hbb/data/run2_2L_1FJ_best_model.pt"
model_file["2017"]["2"] = "src/hbb/data/run2_2L_2FJ_best_model.pt"
model_file["2018"]["1"] = "src/hbb/data/run2_2L_1FJ_best_model.pt"
model_file["2018"]["2"] = "src/hbb/data/run2_2L_2FJ_best_model.pt"

class categorizer(SkimmerABC):
    def __init__(
        self,
        year="2022",
        nano_version="v12",
        xsecs: dict = None,
        systematics=False,
        save_skim=False,
        skim_outpath="",
        btag_eff=False,
        save_skim_nosysts=False,
        nFJ=1,
        dataset=""
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb
        self._dataset = dataset
        self._year = year
        self._isData = "_Run20" in self._dataset
        self._nano_version = nano_version
        self._systematics = systematics
        self._skip_syst = save_skim_nosysts
        self._save_skim = save_skim
        if self._skip_syst:
            self._save_skim = True
        self._skim_outpath = skim_outpath
        self._btag_eff = btag_eff
        self._mupt_type = "pt" #"ptcorr"
        self._nFJ = nFJ
        
        with Path("src/hbb/dilep_triggers.json").open() as f:
            self._triggers = json.load(f)

        self.make_output = lambda: {
            "cutflow": Hist.new.StrCat([], growth=True, name="region", label="Region")
            .StrCat([], growth=True, name="dataset", label="Dataset")
            .Reg(4, 0, 10000, name="h_pt", label="Higgs AK8 pt")
            .Reg(15, 0, 15, name="cut", label="Cut index")
            .Weight(),
            "btagWeight": Hist.new.Reg(50, 0, 3, name="val", label="BTag correction").Weight(),
            "skim": {},
        }

        # btag efficiency plots - binning according to:
        # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#b-tagging-efficiencies-in-simulation
        self.make_btag_output = lambda: (
            Hist.new.StrCat([], growth=True, name="tagger", label="Tagger")
            .Reg(2, 0, 2, name="passWP", label="passWP")
            .Variable([0, 4, 5], name="flavor", label="Jet hadronFlavour")
            .Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt", label="Jet pt")
            .Reg(4, 0, 2.5, name="abseta", label="Jet abseta")
            .Weight()
        )

    def process(self, events):

        # process only nominal case
        meta = ak.with_field(
            events._meta,
            np.array([], dtype=np.float32),
            "ABCD_score"
        )
        events = dak.map_partitions(add_abcd_score, events, self._nFJ, model_file[self._year][str(self._nFJ)], meta=meta)
        if self._skip_syst or not self._save_skim or not hasattr(events, "genWeight"):
            return {"nominal": self.process_shift(events, "nominal")}

        """
        Add `Up and Down` to total list of energy variations
        Muon Energy: `MuonPTScale and MuonPTResolution` defined in mupt_variations
        Jet Energy: `JES, JER, UES` defined in jerc_variations
        """
        total_variations = (
            ["nominal"]
        )

        """
        run processor for each shift defined in total_variations
        return output as dict {variation: output}
        """
        return {var: self.process_shift(events, var) for var in total_variations}

    def add_common_weights(self, weights, events, dataset):
        """
        Add weights that are not region specific
        """

        # print(events.weight)

        weights.add("genweight", events.baseweight / events.xsecweight )

        # if not self._skip_syst:
        #     weights.add("pileup", events.weight.pileup[0], events.weight.pileup[1], events.weight.pileup[2])
        #     weights.add("ISRPartonShower", events.weight_PSISR[0], events.weight_PSISR[1], events.weight_PSISR[2])
        #     weights.add("FSRPartonShower", events.weight_PSFSR[0], events.weight_PSFSR[1], events.weight_PSFSR[2])
        #     weights.add("scalevarF", events.weight_muF[0], events.weight_muF[1], events.weight_muF[2])
        #     weights.add("scalevarR", events.weight_muR[0], events.weight_muR[1], events.weight_muR[2])

            # Easier to save nominal weights for rest of MC with all of the syst names for grabbing columns in post-processing
            # Need to fix
            # flag_syst = ("Hto2B" in dataset) or ("Hto2C" in dataset) or ("VBFZto" in dataset)
            # add_pdf_weight(weights, getattr(events, "LHEPdfWeight", None) if flag_syst else None)
            # add_scalevar_7pt(
            #     weights, getattr(events, "LHEScaleWeight", None) if flag_syst else None
            # )
            # add_scalevar_3pt(
            #     weights, getattr(events, "LHEScaleWeight", None) if flag_syst else None
            # )

        return

    def add_region_weights(
        self, region, weights, events, btag_jets=None, muons=None, muon_type="", photons=None
        ):
        """
        Add weights that are region specific, depending on objects queried.
        Weights will be differentiated by "REGION{region}_" , which will be used for sorting in the partial_weight call
        """

        weight_str = f"REGION{region}_"

        btag_SF = ak.ones_like(events.run)
        # if not self._skip_syst:

            # if not self._btag_eff and btag_jets is not None:
            #     btag_SF = add_btag_weights(
            #         weights, btag_jets, self._btagger, self._btag_wp, self._year, alt_str=weight_str
            #     )


        return btag_SF

    def get_weight_dict(self, region, weights, events, dataset) -> tuple[dict, dict]:
        """
        Calculate the partial weights and the systematic variations for specified region.
        Saved to dictionary to be output in skim files.
        """

        #Sort the region specific weights
        include_weights = []
        for weight_key in weights._weights.keys():
            if "REGION" in weight_key:
                if region in weight_key:
                    include_weights.append(weight_key)
            else:
                include_weights.append(weight_key)

        logger.debug("weights", extra=weights._weights.keys())
        # dictionary of all weights and variations
        weights_dict = {}
        # dictionary of total # events for norm preserving variations for normalization in postprocessing
        totals_dict = {}

        # nominal
        weights_dict["weight"] = weights.partial_weight(include=include_weights)

        # systematics
        for systematic in weights.variations:
            if "REGION" in systematic:
                if region in systematic:
                    syst_dict = systematic.replace(f"REGION{region}_", "")
                    weights_dict[syst_dict] = weights.partial_weight(include=include_weights, modifier=systematic)
            else:
                weights_dict[systematic] = weights.partial_weight(include=include_weights, modifier=systematic)

        ###################### Normalization (Step 1) ######################
        # strip the year from the dataset name
        # dataset_no_year = dataset.replace(f"{self._year}_", "")
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        if not self._isData: # data has sumw=0, so skip it rather than divide by zero
            weight_norm = 1000.0 * events.lumi * events.xsec / events.sumw
            for key, val in weights_dict.items():
                weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.partial_weight(include=include_weights)

        return weights_dict, totals_dict

    def process_shift(self, events, shift_name):

        if "2022" in self._year or "2023" in self._year:
            return
        dataset = self._dataset
        selection = PackedSelection()
        output = self.make_output() if not self._btag_eff else self.make_btag_output()
        weights = Weights(None, storeIndividual=True)
        # print(events.fields) # ['v1qq', 'nElectron', 'excluded', 'v1q2', 'lepton', 'b2', 'name', 'run', 'year', 'empty', 'shortname', 'met', 'is2017', 'muon', 'truth', 'nFatJets', 'ht', 'v2q1', 'luminosityBlock', 'vbs1', 'HLT', 'hbb', 'nMuon', 'is2018', 'v2qq', 'is2023', 'HEMVeto', 'is2024', 'LHEReweightingWeight', 'sumw', 'kind', 'xsec', 'do', 'fatjet', 'matched', 'xsecweight', 'v1q1', 'isData', 'vbs2', 'event', 'weight', 'isRun2', 'gen', 'b1', 'is2016', 'is2022', 'v2q2', 'lumi', 'electron', 'vbs', 'baseweight', 'isRun3', 'jet', 'ABCD_score']

        # TODO -------- HLT CURRENTLY NOT IN RDF PATHS
        # Below implements the dilepton triggers in the selection
        # trigger = ak.values_astype(ak.zeros_like(events.run), bool)
        # for t in self._triggers[self._year]:
        #     if t in events.HLT.fields:
        #         trigger = trigger | events.HLT[t]
        # selection.add("trigger", trigger)
        # del trigger

        fatjets = set_ak8jets(events.fatjet, self._isData, self._year, self._nano_version)
        jets = set_ak4jets(events.jet, self._isData, self._year, self._nano_version)

        met = events.met

        goodfatjets = good_ak8jets(fatjets)
        goodjets = good_ak4jets(jets)
        
        # ----- LEPTONS -----
        # v15 corrections not available for run2 ul - so not implemented in current test - /cvmfs/cms-griddata.cern.ch/cat/metadata//MUO/
        # muons = correct_muons(events.muon, events, self._year, isRealData)
        muons = events.muon

        goodmuons = highpt_muons(muons, self._mupt_type)
        nmuons = ak.num(goodmuons, axis=1)
        leadingmuon = ak.firsts(goodmuons)
        ttbarmuon = ak.firsts(goodmuons[getattr(goodmuons, self._mupt_type) > 55.0])
        # low pt muons break sf (lower bound 15GeV)

        goodelectrons = good_electrons(events.electron)
        nelectrons = ak.num(goodelectrons, axis=1)

        goodmuons["flavor"] = ak.zeros_like(goodmuons.pt)
        goodelectrons["flavor"] = ak.ones_like(goodelectrons.pt)

        # 2L Leading Leptons
        all_leps = ak.concatenate([goodmuons, goodelectrons], axis=1)
        all_leps = ak.with_name(all_leps, "PtEtaPhiMLorentzVector")
        leps_ordered = all_leps[ak.argsort(all_leps.pt, axis=1, ascending=False)]
        leadinglep = ak.firsts(leps_ordered[:, 0:1])
        subleadinglep = ak.firsts(leps_ordered[:, 1:2])
        lep_mass = (leadinglep + subleadinglep).mass

        selection.add("highmet", met.pt > 100.0)

        selection.add("noleptons", (nmuons == 0) & (nelectrons == 0))
        selection.add("twoleptons", (ak.num(all_leps) == 2))
        selection.add("oppsign", (leadinglep.charge * subleadinglep.charge) < 0)
        selection.add("lepdR", (leadinglep.delta_r(subleadinglep) > 0.5))
        selection.add("sameflavor", leadinglep.flavor == subleadinglep.flavor)

        inZpeak = ((lep_mass) <= 101) & ((lep_mass) >= 81)
        notZpeak = ((lep_mass) >= 101) | ((lep_mass) <= 81)

        selection.add("inZpeak", inZpeak)
        selection.add("notZpeak", (notZpeak) | (leadinglep.flavor != subleadinglep.flavor))

        selection.add("onemuon", (nmuons == 1) & (nelectrons == 0))
        selection.add("muonkin", (getattr(leadingmuon, self._mupt_type) > 55.0) & (abs(leadingmuon.eta) < 2.1))


        # ---- Higgs AK8 ----
        goodfatjets = good_ak8jets(fatjets)
        candfatjets = goodfatjets[(goodfatjets.pt > 250) & (goodfatjets.msd > 40)]
        dR_leadlep = candfatjets.delta_r(leadinglep)
        dR_subleadlep = candfatjets.delta_r(subleadinglep)
        ak8_outside_leps = candfatjets[(dR_leadlep > 0.8) & (dR_subleadlep > 0.8)]

        if "v12" in self._nano_version:
            xbbfatjets = ak8_outside_leps[ak.argsort(ak8_outside_leps.pnetXbbVsQCD, axis=1, ascending=False)]
        else:
            xbbfatjets = ak8_outside_leps[ak.argsort(ak8_outside_leps.ParTPXbbVsQCD, axis=1, ascending=False)]

        candidatejet = ak.firsts(xbbfatjets)
        Hscore = candidatejet.globalParT3_Xbb / (candidatejet.globalParT3_Xbb + candidatejet.globalParT3_QCD)
        # ---- 2ND AK8 ----
        dR_candHiggs = candfatjets.delta_r(candidatejet)
        ak8_outside_objs = candfatjets[(dR_leadlep > 0.8) & (dR_subleadlep > 0.8) & (dR_candHiggs > 0.8)] 
        
        #sorted in pt already
        candidateVjet = ak.firsts(ak8_outside_objs)

        selection.add("onegoodAK8", (ak.num(ak8_outside_leps) == 1))
        selection.add("twogoodAK8", (ak.num(ak8_outside_leps) == 2))
        VZscore = (candidateVjet.globalParT3_Xbb + candidateVjet.globalParT3_Xcc + candidateVjet.globalParT3_Xqq) / (candidateVjet.globalParT3_Xbb + candidateVjet.globalParT3_Xcc + candidateVjet.globalParT3_Xqq + candidateVjet.globalParT3_QCD)
        VWscore = (candidateVjet.globalParT3_Xcs + (1/3)*candidateVjet.globalParT3_Xqq) / (candidateVjet.globalParT3_Xcs + (1/3)*candidateVjet.globalParT3_Xqq + candidateVjet.globalParT3_QCD)

        # ---- AK4 Jets ----
        goodjets = good_ak4jets(jets)
        dR = goodjets.delta_r(candidatejet)
        dR_V_pass = goodjets.delta_r(candidateVjet) > 0.8 #for events with no V jet
        dR_V_pass = ak.fill_none(dR_V_pass, True)
        ak4_outside_ak8 = goodjets[(dR > 0.8)]
        ak4_medb_outside_ak8 = ak4_outside_ak8[ak4_outside_ak8.isMediumBTag]
        nb_veto = ak.num(ak4_medb_outside_ak8, axis=1)
        

        #AK4 b-jet vetos
        selection.add("antiak4btagMedium", nb_veto == 0)
        selection.add("ak4btagMedium08", nb_veto > 0)

        # ---- VBF Jets ----
        dR_higgs = goodjets.delta_r(candidatejet)
        dR_leadlep = goodjets.delta_r(leadinglep)
        dR_subleadlep = goodjets.delta_r(subleadinglep)
        dR_V_pass = goodjets.delta_r(candidateVjet) > 0.8 #for events with no V jet
        dR_V_pass = ak.fill_none(dR_V_pass, True) #for events with no V jet
        ak4_outside_objs = goodjets[((dR_higgs > 0.8) & (dR_leadlep > 0.4) & (dR_subleadlep > 0.4))]
        # ak4_outside_objs = goodjets[((dR_higgs > 0.8) & (dR_leadlep > 0.4) & (dR_subleadlep > 0.4))]

        # pairs = ak.combinations(ak4_outside_objs, 2, fields=["j1", "j2"])
        # deta_pairs = abs(pairs.j1.eta - pairs.j2.eta)
        # max_idx = ak.argmax(deta_pairs, axis=1, keepdims=True)

        # VBF specific variables
        # jet1_away = ak.firsts(pairs.j1[max_idx])
        # jet2_away = ak.firsts(pairs.j2[max_idx])

        jet1_away = ak.firsts(ak4_outside_objs[:, 0:1])
        jet2_away = ak.firsts(ak4_outside_objs[:, 1:2])


        #vbf_deta = abs(jet1_away.eta - jet2_away.eta)
        #vbf_mjj = (jet1_away + jet2_away).mass
        vbf_deta = events.vbs.detajj
        vbf_mjj = events.vbs.mjj
        vbf_score = events.vbs.score
        abcd_score = events.ABCD_score
        HT = events.ht.fatjets
        isvbf = (vbf_deta > 2.5) & (vbf_mjj > 750)
        isvbf = ak.fill_none(isvbf, False)

        nak4s_vbf = ak.num(ak4_outside_objs, axis=1)

        selection.add("2ak4s", nak4s_vbf > 1)
        selection.add("isvbf", isvbf)
        gen_variables = {}
        btag_SF = ak.ones_like(events.run)

        self.add_common_weights(weights, events, dataset)
        # signal regions
        btag_SF = self.add_region_weights("signal", weights, events, btag_jets=ak4_outside_ak8)


        weights_dict, totals_temp = self.get_weight_dict("signal", weights, events, dataset)

        # for d, gen_func in gen_selection_dict.items():
        #     if d in dataset:
        #         # match goodfatjets
        #         gen_variables = gen_func(events, goodfatjets)

        # ------ Boson Gen Matching
        # bosons = getBosons(events.GenPart)
        # matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
        # match_mask = (abs(candidatejet.pt - matchedBoson.pt) / matchedBoson.pt < 0.5) & (
        #     abs(candidatejet.msd - matchedBoson.mass) / matchedBoson.mass < 0.3
        # )
        # selmatchedBoson = ak.mask(matchedBoson, match_mask)
        # genflavor = bosonFlavor(selmatchedBoson)
        # genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)

        # softdrop mass, 0 for genflavor == 0
        # msd_matched = candidatejet.msd * (genflavor > 0) + candidatejet.msd * (genflavor == 0)
        msd_matched = candidatejet.msd

        regions = {
            "signal-wwh": [
                # "trigger",
                # "ak4jetveto",
                "twoleptons",
                # "highmet",
                "oppsign",
                "lepdR",
                "notZpeak",
                "onegoodAK8",
                "antiak4btagMedium",
                "2ak4s",
                # "isvbf",

            ],
            "signal-zzh-1FJ": [
                # "trigger",
                # "ak4jetveto",
                "twoleptons",
                # "highmet",
                "oppsign",
                "sameflavor",
                "lepdR",
                "inZpeak",
                "onegoodAK8",
                "antiak4btagMedium",
                "2ak4s",
                # "isvbf",
            ],
            "signal-wzh-zzh-2FJ": [
                # "trigger",
                # "ak4jetveto",
                "twoleptons",
                "oppsign",
                "sameflavor",
                "lepdR",
                "inZpeak",
                "twogoodAK8",
                "antiak4btagMedium",
                "2ak4s",
                # "isvbf",
            ],
        }

        btag_eff_cuts = [
            # "trigger",
            "lumimask",
            "metfilter",
            "ak4jetveto",
            "minjetkin",
            "lowmet",
            "noleptons",
        ]

        tic = time.time()

        nominal_weight = weights_dict["weight"]
        if self._isData:
            nominal_weight = ak.ones_like(events.run)
        gen_weight = weights_dict["weight_noxsec"]

        if self._btag_eff:
            cut = selection.all(*btag_eff_cuts)
            flat_gj = ak.flatten(goodjets)

            output.fill(
                tagger=self._btagger,
                abseta=self.normalize(abs(flat_gj.eta), cut),
                pt=self.normalize(flat_gj.pt, cut),
                flavor=self.normalize(flat_gj.hadronFlavour, cut),
                passWP=self.normalize(getattr(flat_gj, self._btagger) > self._btag_cut, cut),
            )
            return output

        output_array = None
        if self._save_skim:

            output_array = {
                # "GenBoson_pt": genBosonPt,
                # "GenFlavor": genflavor,
                "nFatJet": ak.num(goodfatjets, axis=1),
                "nJet": ak.num(goodjets, axis=1),
                "HiggsAK8_pt": candidatejet.pt,
                "HiggsAK8_phi": candidatejet.phi,
                "HiggsAK8_eta": candidatejet.eta,
                "HiggsAK8_msd": candidatejet.msd,
                "HiggsAK8_msdmatched": msd_matched,
                "HiggsAK8_n2b1": candidatejet.n2b1,
                "HiggsAK8_n3b1": candidatejet.n3b1,
                "HiggsAK8_pnetMass": candidatejet.pnetmass,
                "HiggsAK8_pnetTXbb": candidatejet.particleNet_XbbVsQCD,
                "HiggsAK8_pnetTXcc": candidatejet.particleNet_XccVsQCD,
                "HiggsAK8_pnetTXqq": candidatejet.particleNet_XqqVsQCD,
                "HiggsAK8_pnetTXgg": candidatejet.particleNet_XggVsQCD,
                "HiggsAK8_pnetTQCD": candidatejet.particleNet_QCD,
                "HiggsAK8_pnetXbbXcc": candidatejet.pnetXbbXcc,
                "Hscore": Hscore,                
                "VAK8_pt": candidateVjet.pt,
                "VAK8_phi": candidateVjet.phi,
                "VAK8_eta": candidateVjet.eta,
                "VAK8_msd": candidateVjet.msd,
                "VAK8_n2b1": candidateVjet.n2b1,
                "VAK8_n3b1": candidateVjet.n3b1,
                "VAK8_pnetMass": candidateVjet.pnetmass,
                "VAK8_pnetTXbb": candidateVjet.particleNet_XbbVsQCD,
                "VAK8_pnetTXcc": candidateVjet.particleNet_XccVsQCD,
                "VAK8_pnetTXqq": candidateVjet.particleNet_XqqVsQCD,
                "VAK8_pnetTXgg": candidateVjet.particleNet_XggVsQCD,
                "VAK8_pnetTQCD": candidateVjet.particleNet_QCD,
                "VAK8_pnetXbbXcc": candidateVjet.pnetXbbXcc,
                "VZscore": VZscore,
                "VWscore": VWscore,
                "VBFPair_mjj": vbf_mjj,
                "VBFPair_deta": vbf_deta,
                "VBFPair_score": vbf_score,
                "ABCD_score": abcd_score,
                "HT": HT,
                "MET": met.pt,
                "LepPair_mass": ak.fill_none(lep_mass, -999.),
                "LeadingLep_pt": leadinglep.pt,
                "LeadingLep_flavor": leadinglep.flavor,
                "LeadingLep_phi": leadinglep.phi,
                "LeadingLep_eta": leadinglep.eta,
                "SubLeadingLep_pt": subleadinglep.pt,
                "SubLeadingLep_flavor": subleadinglep.flavor,
                "SubLeadingLep_phi": subleadinglep.phi,
                "SubLeadingLep_eta": subleadinglep.eta,
                "weight": nominal_weight,
                "genWeight": gen_weight,
                # **gen_variables,
            }

            # reduced output array for energy variation shift
            energy_var_array = {
                # "GenBoson_pt": genBosonPt,
                # "GenFlavor": genflavor,
                "HiggsAK8_pt": candidatejet.pt,
                "HiggsAK8_msd": candidatejet.msd,
                "HiggsAK8_msdmatched": msd_matched,
                "HiggsAK8_pnetTXbb": candidatejet.particleNet_XbbVsQCD,
                "HiggsAK8_pnetTXcc": candidatejet.particleNet_XccVsQCD,
                "HiggsAK8_pnetXbbXcc": candidatejet.pnetXbbXcc,
                "VAK8_pt": candidateVjet.pt,
                "VAK8_msd": candidateVjet.msd,
                "VAK8_msdmatched": msd_matched,
                "VAK8_pnetTXbb": candidateVjet.particleNet_XbbVsQCD,
                "VAK8_pnetTXcc": candidateVjet.particleNet_XccVsQCD,
                "VAK8_pnetXbbXcc": candidateVjet.pnetXbbXcc,
                "VBFPair_mjj": vbf_mjj,
                "weight": nominal_weight,
                "genWeight": gen_weight,
            }

            if "v12" not in self._nano_version:
                parT_array = {
                    "HiggsAK8_ParTPQCD": candidatejet.ParTPQCD,
                    "HiggsAK8_ParTPXbb": candidatejet.ParTPXbb,
                    "HiggsAK8_ParTPXcc": candidatejet.ParTPXcc,
                    "HiggsAK8_ParTPXqq": candidatejet.ParTPXqq,
                    "HiggsAK8_ParTPXcs": candidatejet.ParTPXcs,
                    "HiggsAK8_ParTPXbbVsQCD": candidatejet.ParTPXbbVsQCD,
                    "HiggsAK8_ParTPXccVsQCD": candidatejet.ParTPXccVsQCD,
                    "HiggsAK8_ParTPXbbXcc": candidatejet.ParTPXbbXcc,
                    "HiggsAK8_ParTmassGeneric": candidatejet.ParTmassGeneric,
                    "HiggsAK8_ParTmassX2p": candidatejet.ParTmassX2p,
                    "VAK8_ParTPQCD": candidateVjet.ParTPQCD,
                    "VAK8_ParTPXbb": candidateVjet.ParTPXbb,
                    "VAK8_ParTPXcc": candidateVjet.ParTPXcc,
                    "VAK8_ParTPXqq": candidateVjet.ParTPXqq,
                    "VAK8_ParTPXcs": candidateVjet.ParTPXcs,
                    "VAK8_ParTPXbbVsQCD": candidateVjet.ParTPXbbVsQCD,
                    "VAK8_ParTPXccVsQCD": candidateVjet.ParTPXccVsQCD,
                    "VAK8_ParTPXbbXcc": candidateVjet.ParTPXbbXcc,
                    "VAK8_ParTmassGeneric": candidateVjet.ParTmassGeneric,
                    "VAK8_ParTmassX2p": candidateVjet.ParTmassX2p
                }
                output_array = {**output_array, **parT_array}

                energy_var_array_parT = {
                    "HiggsAK8_ParTPXbbVsQCD": candidatejet.ParTPXbbVsQCD,
                    "HiggsAK8_ParTPXccVsQCD": candidatejet.ParTPXccVsQCD,
                    "HiggsAK8_ParTPXbbXcc": candidatejet.ParTPXbbXcc,
                    "VAK8_ParTPXbbVsQCD": candidateVjet.ParTPXbbVsQCD,
                    "VAK8_ParTPXccVsQCD": candidateVjet.ParTPXccVsQCD,
                    "VAK8_ParTPXbbXcc": candidateVjet.ParTPXbbXcc,
                }
                energy_var_array = {**energy_var_array, **energy_var_array_parT}

            # extra variables for big array
            output_array_extra = {
                # AK4 Jets away from HiggsAK8
                "Jet0_pt": jet1_away.pt,
                "Jet0_eta": jet1_away.eta,
                "Jet0_phi": jet1_away.phi,
                "Jet0_mass": jet1_away.mass,
                "Jet0_btagPNetB": jet1_away.btagPNetB,
                "Jet0_btagPNetCvB": jet1_away.btagPNetCvB,
                "Jet0_btagPNetCvL": jet1_away.btagPNetCvL,
                "Jet0_btagPNetQvG": jet1_away.btagPNetQvG,
                "Jet1_pt": jet2_away.pt,
                "Jet1_eta": jet2_away.eta,
                "Jet1_phi": jet2_away.phi,
                "Jet1_mass": jet2_away.mass,
                "Jet1_btagPNetB": jet2_away.btagPNetB,
                "Jet1_btagPNetCvB": jet2_away.btagPNetCvB,
                "Jet1_btagPNetCvL": jet2_away.btagPNetCvL,
                "Jet1_btagPNetQvG": jet2_away.btagPNetQvG,
            }

        def skim(region, output_array):
            selections = regions[region]
            cut = selection.all(*selections)

            # to debug...
            # print(output_array.compute())
            # print(output_array[cut].compute())

            if "root:" in self._skim_outpath:
                skim_path = f"{self._skim_outpath}/{shift_name.replace('_', '')}/{self._year}/{dataset}/{region}"
            else:
                skim_path = (
                    Path(self._skim_outpath)
                    / shift_name.replace("_", "")
                    / self._year
                    / dataset
                    / region
                )
                skim_path.mkdir(parents=True, exist_ok=True)
            print("Saving skim to: ", skim_path)

            output["skim"][region] = dak.to_parquet(
                output_array[cut],
                str(skim_path),
                compute=False,
            )

            if shift_name == "nominal":

                # Fill cutflow hist
                allcuts = set()
                cut = selection.all(*allcuts)
                output["cutflow"].fill(
                    dataset=dataset,
                    region=region,
                    h_pt=self.normalize(candidatejet.pt, None),
                    cut=0,
                    weight=nominal_weight,
                )
                for i, cut in enumerate(selections):
                    allcuts.add(cut)
                    cumulative_cut = selection.all(*allcuts)
                    output["cutflow"].fill(
                        dataset=dataset,
                        region=region,
                        h_pt=self.normalize(candidatejet.pt, cumulative_cut),
                        cut=i + 1,
                        weight=nominal_weight[cumulative_cut],
                    )
                # Fill btag SF hist
                cut = selection.all(*selections)
                output["btagWeight"].fill(val=self.normalize(btag_SF, cut))

        if self._save_skim:
            if shift_name == "nominal":
                for region in regions:
                    skim(region, ak.zip({**output_array, **output_array_extra, **weights_dict}, depth_limit=1))

            else:  # energy variation shift case
                for region in regions:
                    if "signal" in region:
                        skim(
                            region,
                            ak.zip({**energy_var_array, **weights_dict}, depth_limit=1),
                        )

        toc = time.time()
        output["filltime"] = toc - tic
        print(f"Time to fill histograms: {toc - tic:.2f} seconds")
        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        pass
