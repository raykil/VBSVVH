import numpy as np
from fnmatch import fnmatch

LUMI = {"run2": 19.52 + 16.81 + 41.48 + 59.83, "run3": 109.95 + 110.84} # fb^-1
SIGNALS = ['c2v1p0_c3_1p0', 'c2v1p5_c3_1p0', 'c2v1p0_c3_10p0'] # signal categories of CategorizeSamples

def CategorizeSamples(events, exclude=[]): # events: {process: events}. Returns {category: {'color', 'names', 'events': {process: events}}}
    categories = {  # first matching glob wins, so VVV/ttX precede VV/ttbar
        'c2v1p0_c3_1p0' : {'color': 'blue'      , 'names': ['VBS*_c2v1p0_c3_1p0_*']}, # signal hypotheses
        'c2v1p5_c3_1p0' : {'color': 'green'     , 'names': ['VBS*_c2v1p5_c3_1p0_*']},
        'c2v1p0_c3_10p0': {'color': 'red'       , 'names': ['VBS*_c2v1p0_c3_10p0_*']},
        'VVV'     : {'color': 'tab:purple', 'names': ['WWW*', 'WWZ*', 'WZZ*', 'ZZZ*']},
        'ttX'     : {'color': 'tab:red'   , 'names': ['TTW*', 'TTZ*', 'ttW*', 'ttZ*', 'TTLL*', 'TTLNu*', 'TTH*', 'ttH*']},
        'ttbar'   : {'color': 'tab:green' , 'names': ['TTTo*', 'TTto*']},
        'single-t': {'color': 'tab:brown' , 'names': ['ST_*', 'TW*', 'TbarW*', 'TBbar*', 'TbarB*', 'TZQB*', 'tZq*']},
        'DY'      : {'color': 'tab:orange', 'names': ['DY*']},
        'VV'      : {'color': 'tab:cyan'  , 'names': ['WW*', 'WZ*', 'ZZ*', 'GluGluToContin*', 'GluGlutoContin*', 'SSWW*', 'VBS-SSWW*']},
        'Higgs'   : {'color': 'tab:olive' , 'names': ['ZH*', 'WplusH*', 'WminusH*', 'GluGluZH*', 'ggZH*', 'GluGluH*', 'VHTo*', 'VBFWH*', 'HZJ*', 'HWJ*']},
        'V+jets'  : {'color': '#1b3a5c'   , 'names': ['WJets*', 'WtoLNu*', 'Wto2Q*', 'ZJets*', 'Zto2Q*', 'EWK*']},
        'QCD'     : {'color': 'tab:gray'  , 'names': ['QCD*']},
    }
    categories = {k: v for k, v in categories.items() if k not in exclude}
    category_of = {p: next((cat for cat, val in categories.items() if any(fnmatch(p, n) for n in val['names'])), None) for p, ev in events.items() if len(ev)} # empty datasets dropped, so no all-zero categories
    return {cat: {**val, 'events': {p: ev for p, ev in events.items() if category_of.get(p) == cat}} for cat, val in categories.items() if cat in category_of.values()}

def getBinEdges(varName, region):
    binEdges = {
        'L1_pt'      : np.arange(20, 501, 20),
        'L2_pt'      : np.arange(10, 301, 10),
        'dilep_mass' : np.arange(81, 102, 1) if region.endswith('VZH') else np.arange(0, 501, 10), # VZH keeps only the Z peak, 81-101 GeV
        'MET'        : np.arange(0, 401, 10),
        'HiggsFJ_pt' : np.arange(250, 1001, 25),
        'HiggsFJ_msd': np.arange(40, 301, 10),
        'HbbScore'   : np.linspace(0, 1, 21),
        'VBS_score'  : np.linspace(0, 1, 21),
        'VBS_Mjj'    : np.arange(0, 3001, 100),
        'VBS_deta'   : np.arange(0, 9.1, 0.5),
        'ABCD_score' : np.linspace(0, 1, 21),
    }
    return binEdges[varName] if varName in binEdges else binEdges[varName.replace('VFJ_', 'HiggsFJ_', 1)] # e.g. VFJ_pt falls back to HiggsFJ_pt