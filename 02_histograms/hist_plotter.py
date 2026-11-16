import os, json
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

parser = ArgumentParser(description='Kinematic distribution plots.')
parser.add_argument('-j', '--jsonpath', type=str, required=True, help='Path to json file with hist info.')
args = parser.parse_args()

# Loading json
with open(args.jsonpath, 'r') as j:
    HIST_DICT = json.load(j)
    channel  = HIST_DICT['channel']  # str
    binEdges = np.array(HIST_DICT['binEdges'])
    lumitext = HIST_DICT['lumitext'] # str
    nMCMC    = HIST_DICT['nMCMC']    # dict. {category: {'color', 'hist'}}
    nSGMC    = HIST_DICT['nSGMC']    # dict. {signal: {'color', 'hist'}}
    mcmcStatUnc_ax1 = np.array(HIST_DICT['mcmcStatUnc_ax1'])
    xlabel   = HIST_DICT['xlabel']   # str
    nDATA    = np.array(HIST_DICT['nDATA']) if 'nDATA' in HIST_DICT else None # absent for MC-only jsons
binCenters = (binEdges[:-1] + binEdges[1:]) / 2
binWidths  = np.diff(binEdges)
xlim       = (binEdges[0] - binWidths[0], binEdges[-1] + binWidths[0])

plotname = args.jsonpath.replace('jsons', 'plots').replace('.json', '.png')
os.makedirs(os.path.dirname(plotname), exist_ok=True)

###### Plotting ######
plt.style.use(hep.style.CMS)
fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, dpi=300)

# plotting data
if nDATA is not None:
    ax1.errorbar(binCenters, nDATA, xerr=binWidths/2, yerr=HIST_DICT['dataStatUnc_ax1'], fmt='o', label='Data', color='black', capsize=2)

# plotting background MC
nBGMC = np.zeros_like(binCenters)
for name, mcHist in sorted(nMCMC.items(), key=lambda x: sum(x[1]['hist'])):
    ax1.bar(binCenters, mcHist['hist'], width=binWidths, label=name, color=mcHist['color'], bottom=nBGMC)
    nBGMC += mcHist['hist']

# plotting signal MC
for name, mcHist in nSGMC.items():
    ax1.stairs(mcHist['hist'], binEdges, label=name, color=mcHist['color'], linewidth=2)

# plotting MC statistical uncertainty
ax1.bar(binCenters, mcmcStatUnc_ax1 * 2, width=binWidths, bottom=nBGMC - mcmcStatUnc_ax1, color='gray', alpha=0.3, hatch='//', label='stat. unc.')

# ax1 setting
ax1.set_ylabel("Events")
ax1.set_xlim(xlim)
ax1.set_yscale('log')
ax1.set_ylim(min((h for mc in nSGMC.values() for h in mc['hist'] if h > 0), default=1e-2) / 2, nBGMC.max() * 1e3)
ax1.grid()
ax1.legend(ncol=3, fontsize='14', loc='upper right')
hep.cms.label("Preliminary", data=nDATA is not None, rlabel=lumitext, ax=ax1)
plt.text(0.15, 0.81, channel, size='x-small', color='darkslategray', transform=fig.transFigure)

# plotting ratio (ax2)
if nDATA is not None:
    mcmcStatUnc_ax2 = np.array(HIST_DICT['mcmcStatUnc_ax2'])
    ax2.errorbar(binCenters, HIST_DICT['ratio'], xerr=binWidths/2, yerr=HIST_DICT['ratioStatUnc_ax2'], fmt='o', color='black', capsize=2)
    ax2.bar(binCenters, mcmcStatUnc_ax2 * 2, width=binWidths, bottom=1 - mcmcStatUnc_ax2, color='gray', alpha=0.3)
else:
    ax2.axvspan(*xlim, facecolor='none', edgecolor='silver', hatch='//', linewidth=0)
    ax2.text(0.5, 0.5, 'Blinded', color='gray', transform=ax2.transAxes, ha='center', va='center', size='small', bbox=dict(facecolor='white', edgecolor='none'))

# ax2 setting
ax2.set_xlabel(xlabel)
ax2.set_ylabel('Data/Bkg')
ax2.set_ylim(0.5, 1.5)
ax2.grid()
ax2.axhline(y=1, color='gray', linestyle='dashed')

plt.savefig(plotname)
print(f"{plotname} saved!")
