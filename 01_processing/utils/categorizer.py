"""
Streamlining done (WWh, VVH), but it has inference logic. Compatible with CategorizerTools_v1.py. 
Also this version accepts root files as inputs.
"""

from utils.SkimmerABC import SkimmerABC
from utils.CategorizerTools import *
from utils.objects import *

class categorizer(SkimmerABC):
    def __init__(self, year='2022', nFJ=1, xsecs: dict={}, systs=False):
        super().__init__()

        self._year    = year
        self._nFJ     = nFJ
        self.XSECS    = xsecs
        self._systs   = systs

        self._models       = load_abcd_models(year, nFJ)
        # self._mupt_type    = "pt"
        # self._triggers     = load_triggers()

    def process(self, events):
        meta = add_abcd_scores(events._meta, self._models) # typetracer pass: the output fields, no data
        events = dak.map_partitions(add_abcd_scores, events, self._models, meta=meta)

        variations = ['nominal'] # 'Up and Down'
        # process_shift runs per variation, per year, per nFJ
        results = {var: self.process_shift(events, var) for var in variations} # {'nominal': {region: array, 'cutflow': xxx, 'btagWeight': xxx, 'filltime': xxx}}
        return results

    def process_shift(self, events, variation):
        # dataset   = events.metadata["dataset"] # dataset name from sample_json/files_run*_*FJ.json. Ex) VBSWWH_OS_c2v1p0_c3_1p0_UL16
        # isData    = "_Run20" in dataset
        SELECTION = PackedSelection() # initially empty. Filled with named T/F masks. Every selection.add(name, cut) adds a row here.
        WEIGHTS   = Weights(None, storeIndividual=True) # Named weights. weights.add(name, array of floats).

        # ————— Trigger ———————————————————————————————————————————————————————
        # addTriggerCut(events, SELECTION) # adds row to SELECTION # TODO: needs review. ddTriggerCut is defined as (self, events, selection) but called as addTriggerCut(events, SELECTION) (categorizer.py:37 vs CategorizerTools.py:114), and its trigger dict has no entries for 2016/2016APV/2017/2025.

        # ————— MET ———————————————————————————————————————————————————————————
        MET = events.met
        SELECTION.add("highmet", MET.pt > 100.0)

        # ————— Leptons ———————————————————————————————————————————————————————
        Muons = SelectGoodMuons(events.muon, self._nFJ) # need muon correction for NANOAOD v15. Currently muon corr available only for Run2UL v9.
        Elecs = SelectGoodElectrons(events.electron)
        LeptonInfo, LeadLep, SubleadLep = setLeptons(Elecs, Muons)
        SELECTION.add("twoLeps" , LeptonInfo['nLeps']==2) # SELECTION only stores event-level cuts. Select* function give object level selection, and set* functions only define fields
        SELECTION.add("OppSign" , LeptonInfo['OppSign'])
        SELECTION.add("dR0p5"   , LeptonInfo['lepdR']>0.5)
        SELECTION.add("SameFlav", LeptonInfo['SameFlav'])
        SELECTION.add("inZpeak" , LeptonInfo['inZpeak'])
        SELECTION.add("noZpeak" , LeptonInfo['noZpeak'])

        # ————— FatJets ———————————————————————————————————————————————————————
        AK8Jets = SelectGoodAK8Jets(setAK8Jets(events.fatjet))
        HiggsFJ = ak.firsts(AK8Jets[ak.argmax(AK8Jets.ParTPXbbVsQCD, axis=1, keepdims=True)]) # highest-Xbb FJ. None if no FJ. HiggsFJ.ParTPXbbVsQCD matches Hscore in working_categorizer.
        Hscore = HiggsFJ.ParTPXbbVsQCD
        VFJ = ak.firsts(AK8Jets[AK8Jets.delta_r(HiggsFJ) > 0.8]) # FJ with highest pt that is isolated from HiggsFJ. None if none.
        SELECTION.add("VFJisoH"   , ~ak.is_none(VFJ)) # in 1Fj case, VFJ is always None
        SELECTION.add("OneGoodAK8", ak.num(AK8Jets, axis=1) == 1)
        SELECTION.add("TwoGoodAK8", ak.num(AK8Jets, axis=1) == 2) # when selecting 2FJ events, make sure to use both TwoGoodAK8 & VFJisoH
        SELECTION.add("passNom"   , events.passes[f"2lep_{self._nFJ}FJ_nom"]) # drops JES/JER-variation leaks. Later, feed in which shift I'm doing and wire this accordingly.

        # ————— AK4 jets ——————————————————————————————————————————————————————
        AK4Jets = SelectGoodAK4Jets(setAK4Jets(events.jet))
        n_bJets = ak.num(AK4Jets[AK4Jets.isMediumBTag])
        isVBF   = setVBFJets(events, AK4Jets)
        SELECTION.add("nobJets", n_bJets==0)
        SELECTION.add('isVBF'  , isVBF)

        # ————— Regional cutflow ——————————————————————————————————————————————
        common_selections = ['twoLeps', 'OppSign', 'dR0p5', 'nobJets', 'isVBF', 'passNom']
        REGIONS = {
            "1FJ_WWH": common_selections + ['OneGoodAK8', 'noZpeak'], # wwh is tighter than signal_wwh in noZpeak(current version drops OR oppflavor) & isVBF(old: vbf jet pair X required to have deta>2.5 & mjj>750, new: vbf jet pair should satisfy that).
            "1FJ_VZH": common_selections + ['OneGoodAK8', 'SameFlav', 'inZpeak'],
            "2FJ_VZH": common_selections + ['TwoGoodAK8', 'SameFlav', 'inZpeak', 'VFJisoH']
        }

        # ————— Weights (most of SFs already multiplied in preselection and stored in "weight" branch) ———————————————————————————————————————————————————————
        # for systematics: pileup, btag SF, PS, scale, PDF (X change event selection)
        # energy systematics (JES/JER/UES, muon pT scale/resolution) rerun process_shift with shifted kinematics, so selected events change (even tho cutflow itself X change)
        WEIGHTS.add("extraSF", ak.ones_like(events.weight)) # factors on top of preselection's events.weight (= genWeight * xsecweight * all nominal SFs, weights.cpp applyMCWeights; 1 for data). None yet.
        # btag_SF = ak.ones_like(events.run)

        # ————— Saving output —————————————————————————————————————————————————
        events["weight"] = events.weight * WEIGHTS.weight()
        events["LeadLep"], events["SubleadLep"], events["LeptonInfo"] = LeadLep, SubleadLep, dak.zip(LeptonInfo) # attached for fieldsToSave
        events["HiggsFJ"], events["VFJ"] = HiggsFJ, VFJ # attached for fieldsToSave
        Events = ak.zip(fieldsToSave(events), depth_limit=1)
        
        OUTPUT = {} # run_categorizer writes OUTPUT[region]["events"] to {channel}/{year}/{region}/{shift}/{dataset}.parq, the rest to {region}/cutflow.pkl
        for region, cutlist in REGIONS.items():
            OUTPUT[region] = {
                "events" : Events[SELECTION.all(*cutlist)],
                "cutflow": makeCutflow(SELECTION, cutlist, Events.weight)
            }

        return OUTPUT
        # returns dict: {
        #     region: {
        #         "events"    : awkward array of selected events,
        #         "cutflow"   : Hist,
        #         "btagWeight": Hist, (not in current version)
        #         "filltime"  : float (not in current version)
        #     }
        # }

    def postprocess(self, accumulator): # required by coffea's ProcessorABC; unused (apply_to_fileset + dask.compute)
        pass