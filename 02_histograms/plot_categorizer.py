import os, sys, json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from subprocess import run
import awkward as ak
from PlotterTools import *

scriptPath = os.path.dirname(os.path.abspath(__file__))
sys.path += [f"{scriptPath}/..", f"{scriptPath}/../../run3-vbsvvh/abcd", f"{scriptPath}/../../run3-vbsvvh"]
from rename_to_old_names import SHORT_TO_FULL
from functions import chooseOverlaps

if __name__ == "__main__":
    parser = ArgumentParser(description='Distributions from run_categorizer.py parquets, summed over all years of the channel.')
    parser.add_argument('-c', '--channel' , type=str, default='run2_1FJ', choices=['run2_1FJ', 'run2_2FJ', 'run3_1FJ', 'run3_2FJ'])
    parser.add_argument('-r', '--region'  , type=str, default='1FJ_WWH', choices=['1FJ_WWH', '1FJ_VZH', '2FJ_VZH'])
    parser.add_argument('-v', '--variable', type=str, default='ABCD_score', help='x-axis variable (a fieldsToSave field in CategorizerTools.py).')
    parser.add_argument('-d', '--draw'    , action="store_true", help='If True, runs hist_plotter.py to plot the json.')
    args = parser.parse_args()
    run_ = args.channel.split('_')[0]
    abcd_score_cut = 0.87
    vbs_score_cut  = 0.71

    # ── Load samples ──────────────────────────────────────────────────────────
    rejected = chooseOverlaps() # full dataset names; parquets are named by SHORT_TO_FULL's keys
    paths = [p for p in Path(f"{scriptPath}/../results/{args.channel}").glob(f"*/{args.region}/nominal/*.parq") if SHORT_TO_FULL.get(p.stem) not in rejected]
    events = {p.stem: ak.from_parquet(p, columns=list({args.variable, 'weight', 'ABCD_score', 'VBS_score'})) for p in paths}
    events = {name: ev[(ev.ABCD_score < abcd_score_cut) | (ev.VBS_score < vbs_score_cut)] for name, ev in events.items()} # SR blinded in data and MC alike
    
    DATA = ak.concatenate([ev for name, ev in events.items() if '_Run20' in name])
    BGBG = CategorizeSamples(events) # data matches no category and is dropped
    SGSG = {sig: BGBG.pop(sig) for sig in SIGNALS if sig in BGBG}
    for group in [*SGSG.values(), *BGBG.values()]: group['events'] = ak.concatenate(list(group['events'].values()))
    BGMC = ak.concatenate([BG['events'] for BG in BGBG.values()])

    binEdges = getBinEdges(args.variable, args.region)
    makeHistogram = lambda ev, power=1: ak.to_numpy(np.histogram(np.clip(ev[args.variable], binEdges[0], binEdges[-1]), binEdges, weights=ev.weight**power)[0]) # clip: overflow into edge bins. to_numpy: ak rejects the in-place edit below
    nDATA = makeHistogram(DATA)
    nBGMC = makeHistogram(BGMC)
    mcmcStatUnc = np.sqrt(makeHistogram(BGMC, 2))
    nBGMC[nBGMC <= 0] = np.inf # negative MC weights can sum below 0; no ratio point there

    # ── Export to JSON ────────────────────────────────────────────────────
    HIST_DICT = {
        'channel'         : f"{args.channel}, {args.region}\nSR blinded",
        'lumitext'        : f"{LUMI[run_]:.0f} fb$^{{-1}}$ ({'13' if run_ == 'run2' else '13.6'} TeV)",
        'binEdges'        : binEdges.tolist(),
        'nMCMC'           : {cat: {'color': BG['color'], 'hist': makeHistogram(BG['events']).tolist()} for cat, BG in BGBG.items()},
        'nSGMC'           : {sig: {'color': SG['color'], 'hist': makeHistogram(SG['events']).tolist()} for sig, SG in SGSG.items()},
        'nDATA'           : nDATA.tolist(),
        'ratio'           : (nDATA / nBGMC).tolist(),
        'dataStatUnc_ax1' : np.sqrt(nDATA).tolist(),
        'mcmcStatUnc_ax1' : mcmcStatUnc.tolist(),
        'ratioStatUnc_ax2': (np.sqrt(nDATA) / nBGMC).tolist(), # data stat only; MC stat is the band
        'mcmcStatUnc_ax2' : (mcmcStatUnc / nBGMC).tolist(),
        'xlabel'          : args.variable,
    }
    jsonPath = f"{scriptPath}/{datetime.now().strftime('%y%m%d')}_categorizer/jsons/{args.channel}_{args.region}_{args.variable}.json"
    os.makedirs(os.path.dirname(jsonPath), exist_ok=True)
    with open(jsonPath, 'w') as j: json.dump(HIST_DICT, j, indent=4)
    print(f"{jsonPath.replace(f'{scriptPath}/', '')} saved!")

    if args.draw:
        run(["python", f"{scriptPath}/hist_plotter.py", "-j", jsonPath])
