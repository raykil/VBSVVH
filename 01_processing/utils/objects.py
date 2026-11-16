import awkward as ak
import numpy as np
from coffea.nanoevents.methods.nanoaod import (ElectronArray, FatJetArray, JetArray, MuonArray, PhotonArray)

def setAK8Jets(fatjets: FatJetArray):
    fatjets["jetidtight"] = fatjets.jetId > 1 # Defined in /users/jkil/run3-vbsvvh/preselection/src/corrections.cpp#L1740
    fatjets["ParTPQCD"] = fatjets.globalParT3_QCD # these didn't exist in nanoaod v12.
    fatjets["ParTPXbb"] = fatjets.globalParT3_Xbb
    fatjets["ParTPXcc"] = fatjets.globalParT3_Xcc
    fatjets["ParTPXcs"] = fatjets.globalParT3_Xcs
    fatjets["ParTPXqq"] = fatjets.globalParT3_Xqq

    fatjets["ParTPXbbVsQCD"]   = fatjets.Hbb # Defined in run3-vbsvvh/preselection/src/selections.cpp#L343. Xbb/(Xbb+QCD). -1 if denom is 0. Smae as Hscore in categorizer.
    fatjets["ParTPXccVsQCD"]   = fatjets.globalParT3_Xcc / (fatjets.globalParT3_Xcc + fatjets.globalParT3_QCD)
    fatjets["ParTPXbbXcc"]     = (fatjets.globalParT3_Xbb + fatjets.globalParT3_Xcc) / (fatjets.globalParT3_Xbb + fatjets.globalParT3_Xcc + fatjets.globalParT3_QCD)
    fatjets["ParTmassGeneric"] = fatjets.globalParT3_massCorrGeneric * fatjets.massGloParT3 / fatjets.globalParT3_massCorrX2p
    fatjets["ParTmassX2p"]     = fatjets.massGloParT3

    fatjets["msd"]        = fatjets.msoftdrop
    fatjets["qcdrho"]     = 2 * np.log(fatjets.msd / fatjets.pt)
    fatjets["pnetmass"]   = fatjets.particleNet_massCorr * fatjets.mass
    # fatjets["pnetXbbXcc"] = (fatjets.particleNet_XbbVsQCD + fatjets.particleNet_XccVsQCD) / (fatjets.particleNet_XbbVsQCD + fatjets.particleNet_XccVsQCD + fatjets.particleNet_QCD)
    fatjets["pnetXbbXcc"] = 1 - 1 / (1 + fatjets.particleNet_XbbVsQCD/(1-fatjets.particleNet_XbbVsQCD) + fatjets.particleNet_XccVsQCD/(1-fatjets.particleNet_XccVsQCD)) # (Xbb+Xcc)/(Xbb+Xcc+QCD): bb-or-cc vs QCD score. r/(1-r) = X/QCD, so QCD cancels.
    return fatjets

def SelectGoodAK8Jets(fatjets: FatJetArray):
    cut = fatjets.jetidtight & (fatjets.pt > 250) & (abs(fatjets.eta) < 2.5) # These cuts are already made in preselection: run3-vbsvvh/preselection/src/selections.cpp#L19-L24
    # dR against leptons also already done in selections.cpp same preselection function "_dR_ak8_lep > 0.8 && "
    return fatjets[cut]

def setAK4Jets(jets: JetArray):
    jets["jetidtight"] = jets.jetId > 1
    jets["jetidtightlepveto"] = jets.jetId > 2
    jets["idx"] = ak.local_index(jets, axis=1) # position in events.jet, which events.vbs.jet*_idx refers to. Survives SelectGoodAK4Jets.
    return jets

def SelectGoodAK4Jets(jets: JetArray):
    # AK4 collection is AK4 PUPPI collection, so PU jets already suppressed at jet clustering level
    # PUID might be needed for forward region. In such case, I can use `jet_puIdDisc`
    # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    # cut = ((jets.pt > 30) & (jets.jetidtight) & (jets.jetidtightlepveto) & (abs(jets.eta) < 5.0) & ~((jets.pt <= 50) & (abs(jets.eta) > 2.5) & (abs(jets.eta) < 3.0)))
    cut = ((jets.pt > 30) & (jets.jetidtightlepveto) & (abs(jets.eta) < 5.0) & ~((jets.pt <= 50) & (abs(jets.eta) > 2.5) & (abs(jets.eta) < 3.0)))
    return jets[cut]


def SelectGoodMuons(muons: MuonArray, nFJ: int):
    cut = (
        (muons.pt > 30) & (np.abs(muons.eta) < 2.4) & (muons.looseId) & (muons.pfRelIso04_all < 0.25) & # & (muons.isGlobal)
        (((abs(muons.eta) <  1.479) & (abs(muons.dz) < 0.1) & (abs(muons.dxy) < 0.05)) |
         ((abs(muons.eta) >= 1.479) & (abs(muons.dz) < 0.2) & (abs(muons.dxy) < 0.1 )))
    )
    if nFJ == 1: cut = cut & muons.mediumId & (muons.pfRelIso04_all < 0.15) # matches muon SF in preselection selections.cpp#L683. 2FJ's SF (#L703) is loose ID + loose iso, already in the cut above.
    return muons[cut]

def SelectGoodElectrons(electrons: ElectronArray):
    cut = (
        (electrons.pt > 10) & (abs(electrons.eta) < 2.5) & (electrons.pfRelIso03_all < 0.15) & (electrons.mvaNoIso_WP90) & (electrons.cutBased >= 2) &
        (((abs(electrons.eta) <  1.479) & (abs(electrons.dz) < 0.1) & (abs(electrons.dxy) < 0.05)) |
         ((abs(electrons.eta) >= 1.479) & (abs(electrons.dz) < 0.2) & (abs(electrons.dxy) < 0.1 )))
    )
    return electrons[cut]

def setLeptons(elecs: ElectronArray, muons: MuonArray):
    Leptons = ak.with_name(ak.concatenate([muons, elecs], axis=1), "PtEtaPhiMLorentzVector")
    Leptons = Leptons[ak.argsort(Leptons.pt, axis=1, ascending=False)]
    LeadLep, SubleadLep = ak.firsts(Leptons[:, 0:1]), ak.firsts(Leptons[:, 1:2])
    
    LeptonInfo = {
        'nMuons'   : ak.num(muons, axis=1),
        'nElecs'   : ak.num(elecs, axis=1),
        'nLeps'    : ak.num(Leptons, axis=1),
        'dilepMass': (LeadLep + SubleadLep).mass,
        'OppSign'  : (LeadLep.charge * SubleadLep.charge) < 0,
        'SameFlav' : abs(LeadLep.pdgId) == abs(SubleadLep.pdgId),
        'lepdR'    : LeadLep.delta_r(SubleadLep)
    }
    LeptonInfo['inZpeak'] = (81 <= LeptonInfo['dilepMass']) & (LeptonInfo['dilepMass'] <= 101)
    LeptonInfo['noZpeak'] = (81 >  LeptonInfo['dilepMass']) | (LeptonInfo['dilepMass'] >  101)
    return LeptonInfo, LeadLep, SubleadLep





# def good_photons(photons: PhotonArray):
#     sel = (photons.pt > 15) & (photons.isScEtaEB | photons.isScEtaEE) & (photons.mvaID_WP80)
#     return photons[sel]

# def tight_photons(photons: PhotonArray):
#     sel = (photons.pt > 30) & (photons.isScEtaEB | photons.isScEtaEE) & (photons.cutBased == 3)
#     return photons[sel]

# def loose_muons(muons: MuonArray, pt_type):
#     sel = (
#         (getattr(muons, pt_type) > 10) & (np.abs(muons.eta) < 2.4) & (muons.looseId) & (muons.pfRelIso04_all < 0.15) & 
#         (((abs(muons.eta) <  1.479) & (abs(muons.dz) < 0.1) & (abs(muons.dxy) < 0.05)) |
#          ((abs(muons.eta) >= 1.479) & (abs(muons.dz) < 0.2) & (abs(muons.dxy) < 0.1 )))
#     )
#     return muons[sel]

# def highpt_muons(muons: MuonArray, pt_type):
#     loosemuons = loose_muons(muons, pt_type)
#     sel = (
#         (getattr(loosemuons, pt_type) > 30) & 
#         (loosemuons.tightId) #1=pass tracker highPtId, 2=pass global highPtId
#         # & (loosemuons.isGlobal)
#     )
#     return loosemuons[sel]

# def good_electrons(electrons: ElectronArray):
#     sel = (
#         (electrons.pt > 10) & (abs(electrons.eta) < 2.5) & (electrons.pfRelIso03_all < 0.15) & (electrons.mvaNoIso_WP90) & 
#         (((abs(electrons.eta) <  1.479) & (abs(electrons.dz) < 0.1) & (abs(electrons.dxy) < 0.05)) |
#          ((abs(electrons.eta) >= 1.479) & (abs(electrons.dz) < 0.2) & (abs(electrons.dxy) < 0.1 )))
#     )
#     return electrons[sel]

# def set_ak4jets(jets: JetArray, isRealData: bool, nano_version: str, event_rho=None):
#     """
#     Jet ID fix for NanoAOD v12 copying
#     # https://gitlab.cern.ch/cms-jetmet/coordination/coordination/-/issues/117#note_8880716
#     """

#     jets["jetidtight"] = jets.jetId > 1
#     jets["jetidtightlepveto"] = jets.jetId > 2

#     # TODO: Add PNet pt regression

#     if event_rho is not None:
#         # jerc variables
#         jets["pt_raw"]    = (1 - jets.rawFactor) * jets.pt
#         jets["mass_raw"]  = (1 - jets.rawFactor) * jets.mass
#         jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
#         if not isRealData:  # only for jer
#             jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)

#     return jets

# def good_ak4jets(jets: JetArray):
#     # ak4 jet definition
#     # Since the main AK4 collection for Run3 is the AK4 Puppi collection, jets originating from pileup are already suppressed at the jet clustering level
#     # PuID might only be needed for forward region (WIP)
#     # JETID: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
#     sel = (
#         (jets.pt > 30) &
#         (jets.jetidtight) &
#         (jets.jetidtightlepveto) &
#         (abs(jets.eta) < 5.0) &
#         ~((jets.pt <= 50) & (abs(jets.eta) > 2.5) & (abs(jets.eta) < 3.0)) # What is it for?
#     )
#     return jets[sel]

# def set_ak8jets(fatjets: FatJetArray, isRealData: bool, nano_version: str, event_rho=None):
#     if "v12" in nano_version:
#         fatjets["jetidtight"] = fatjets.isTight
#     else:
#         fatjets["jetidtight"] = fatjets.jetId > 1

#         fatjets["ParTPQCD"] = fatjets.globalParT3_QCD
#         fatjets["ParTPXbb"] = fatjets.globalParT3_Xbb
#         fatjets["ParTPXcc"] = fatjets.globalParT3_Xcc
#         fatjets["ParTPXcs"] = fatjets.globalParT3_Xcs
#         fatjets["ParTPXqq"] = fatjets.globalParT3_Xqq

#         fatjets["ParTPXbbVsQCD"] = fatjets.globalParT3_Xbb / (fatjets.globalParT3_Xbb + fatjets.globalParT3_QCD)
#         fatjets["ParTPXccVsQCD"] = fatjets.globalParT3_Xcc / (fatjets.globalParT3_Xcc + fatjets.globalParT3_QCD)
#         fatjets["ParTPXbbXcc"]   = (fatjets.globalParT3_Xbb + fatjets.globalParT3_Xcc) / (fatjets.globalParT3_Xbb + fatjets.globalParT3_Xcc + fatjets.globalParT3_QCD)

#         # ParT masses were trained with raw masses. RDF's massGloParT3 = massCorrX2p * (raw mass before JER), which (1 - rawFactor) * mass
#         # can't recover in MC since JER rescales mass without updating rawFactor, so the Generic mass reuses that raw mass.
#         fatjets["ParTmassGeneric"] = fatjets.globalParT3_massCorrGeneric * fatjets.massGloParT3 / fatjets.globalParT3_massCorrX2p
#         fatjets["ParTmassX2p"]     = fatjets.massGloParT3

#     fatjets["msd"]        = fatjets.msoftdrop
#     fatjets["qcdrho"]     = 2 * np.log(fatjets.msd / fatjets.pt)
#     fatjets["pnetmass"]   = fatjets.particleNet_massCorr * fatjets.mass
#     fatjets["pnetXbbXcc"] = (fatjets.particleNet_XbbVsQCD + fatjets.particleNet_XccVsQCD) / (fatjets.particleNet_XbbVsQCD + fatjets.particleNet_XccVsQCD + fatjets.particleNet_QCD)

#     if event_rho is not None:
#         # jerc variables
#         fatjets["pt_raw"]    = (1 - fatjets.rawFactor) * fatjets.pt
#         fatjets["mass_raw"]  = (1 - fatjets.rawFactor) * fatjets.mass
#         fatjets["event_rho"] = ak.broadcast_arrays(event_rho, fatjets.pt)[0]
#         if not isRealData:  # only for jer
#             fatjets["pt_gen"] = ak.values_astype(ak.fill_none(fatjets.matched_gen.pt, 0), np.float32)
#     return fatjets

# def good_ak8jets(fatjets: FatJetArray):
#     # ak8 jet definition
#     sel = fatjets.jetidtight & (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) # sel = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5)
#     return fatjets[sel]

# def trig_match_sel(events, objects, trig_objects, year, trigger, filterbit, ptcut, HLTs, trig_dR=0.2):
#     """
#     Returns selection for objects which are trigger matched to the specified trigger.
#     """
#     trigger = HLTs.hlts_by_type(year, trigger, hlt_prefix=False)[0]  # picking first trigger in list
#     trig_fired = events.HLT[trigger]
#     # print(f"{trigger} rate: {ak.mean(trig_fired)}")

#     filterbit = 2**filterbit

#     pass_trig = (trig_objects.filterBits & filterbit) == filterbit
#     trig_obj = trig_objects[pass_trig]
#     trig_obj_matched = ak.any(objects.metric_table(trig_obj) < trig_dR, axis=2)
#     trig_obj_sel = trig_fired & trig_obj_matched & (objects.pt > ptcut)
#     return trig_obj_sel