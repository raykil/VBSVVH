"""
Plotting functions based on https://github.com/andrzejnovak/combine_postfits
by Andrzej Novak
"""

from __future__ import annotations

import warnings

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import math
from hist.intervals import ratio_uncertainty

hep.style.use("CMS")

def sci_2_before_decimal(x):
    if x == 0:
        return "0.00e+00"
    
    exp = int(math.floor(math.log10(abs(x))))
    
    # Shift exponent so mantissa has 2 digits before decimal
    if abs(x) / 10**exp < 10:
        exp -= 1
    
    mantissa = x / (10**exp)
    
    return f"{mantissa:.2f}e{exp:+03d}"

def extract_mergemap(style):
    """
    Extracts the merge map from the style dictionary.
    """
    compound_keys = [
        key for key in style if "contains" in style[key] and style[key]["contains"] is not None
    ]
    return {key: style[key]["contains"] for key in compound_keys}

def merge_hists(hist_dict, merge_map):
    """
    Merges histograms in hist_dict according to the merge_map.
    The merge_map is a dictionary where keys are the names of the histograms to be created
    and values are lists of histogram names to be merged into the key.
    """
    for k, v in merge_map.items():
        # if k in hist_dict and k != v[0]:
        #     warnings.warn(
        #         f"  Mapping `'{k}' : {v}` will replace existing histogram: '{k}'.", stacklevel=2
        #     )
        to_merge = []
        for name in v:
            if name not in hist_dict:
                warnings.warn(
                    f"  Histogram '{name}' is not available in channel for a merge {v} -> '{k}' and won't be part of the merge.",
                    stacklevel=2,
                )
            else:
                to_merge.append(hist_dict[name])
        if len(to_merge) > 0:
            hist_dict[k] = sum(to_merge)
        # else:
        #     warnings.warn(f"  No histograms available for merge {v} -> '{k}'.", stacklevel=2)
    return hist_dict

def format_legend(ax, ncols=2, handles_labels=None, title=None, **kwargs):
    if handles_labels is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles, labels = handles_labels
    nentries = len(handles)

    kw = dict(framealpha=1, title=title, **kwargs)
    split = nentries // ncols * ncols
    leg1 = ax.legend(
        handles=handles[:split],
        labels=labels[:split],
        ncol=ncols,
        loc="upper right",
        **kw,
    )
    if nentries % 2 == 0:
        return leg1

    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=handles[split:],
        labels=labels[split:],
        ncol=nentries - nentries // ncols * ncols,
        **kw,
    )

    leg2.remove()

    leg1._legend_box._children.append(leg2._legend_handle_box)
    leg1._legend_box.stale = True
    return leg1

def ratio_plot(
    hist_dict: dict[hist.Hist],
    # Sample plotting opts
    sigs: list[str],  # List of samples considered signals  | None = None
    bkgs: (
        list[str] | None
    ) = None,  # List of samples considered background - exclude "onto" sample i.e. QCD
    onto: str | None = None,  # large background to plot others onto typically QCD
    # Style opts
    style: dict | None = None,  # Style YAML
    ratio_with_uncertainty: bool = False,  # Whether to plot ratio/data uncertainty in the ratio
    sort_by_yield: bool = True,  # Whether to sort backgrounds by yield
    legend_title: str | None = None,
):
    style = style.copy()

    # merge histograms according to the style
    merge = extract_mergemap(style)
    hist_dict = merge_hists(hist_dict, merge)

    # --- NEW SORTING LOGIC ---
    if sort_by_yield and bkgs:
        bkg_keys_in_plot = [key for key in bkgs if key in hist_dict]
        bkg_yields = {key: sum(hist_dict[key].values()) for key in bkg_keys_in_plot}
        # print(f"Bkg yields before sorting: {bkg_yields}")
        bkgs = sorted(bkg_yields, key=bkg_yields.get, reverse=True)
        # print(f"Bkg order after sorting by yield: {bkgs}")
    # --- END NEW LOGIC ---

    data = hist_dict.get("data", None)
    all_bkg_keys = bkgs + ([onto] if onto else []) if bkgs else ([onto] if onto else [])
    all_bkgs_hists = [hist_dict[k] for k in all_bkg_keys if k in hist_dict]
    tot_bkg = sum(all_bkgs_hists) if all_bkgs_hists else None

    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    plt.subplots_adjust(hspace=0)
    plt.rcParams.update({"font.size": 24})
    ax.set_yscale('log')

    if(sum(tot_bkg.values()) == 0):
        return fig, (ax, rax)
    
    # print("----------------------")
    # print("TOTBKG", sum(tot_bkg.values()), tot_bkg.variances())
    # for sig in sigs:
    #     print(sig, sum(hist_dict[sig].values()), hist_dict[sig].variances())
    # print("----------------------")

    if onto is None:
        hep.histplot(
            [hist_dict[k] for k in bkgs],# + sigs],
            ax=ax,
            label=bkgs,# + sigs,
            stack=True,
            histtype="fill",
            facecolor=[style[k]["color"] for k in bkgs],# + sigs],
        )
        # hep.histplot(
        #     [hist_dict[k] for k in bkgs + sigs],
        #     ax=ax,
        #     label=bkgs + sigs,
        #     stack=True,
        #     histtype="fill",
        #     facecolor=[style[k]["color"] for k in bkgs + sigs],
        # )
    else:
        if onto in hist_dict:
            hep.histplot(
                hist_dict[onto],
                ax=ax,
                label=onto,
                stack=False,
                histtype="fill",
                facecolor=style[onto]["color"],
            )

        _hatch = [None, *[style[k]["hatch"] for k in bkgs + sigs]]
        _edgecolor = [
            style[k]["color"] if h not in ["none", None] else None
            for k, h in zip([onto] + bkgs + sigs, _hatch)
        ]
        _facecolor = [
            "none" if h not in ["none", None] or k == onto else style[k]["color"]
            for k, h in zip([onto] + bkgs + sigs, _hatch)
        ]
        _linewidth = [2] + [0] * len(bkgs + sigs)

        hep.histplot(
            [hist_dict[onto]] + [hist_dict[k] for k in bkgs + sigs],
            ax=ax,
            label=["_", *(bkgs + sigs)],
            stack=True,
            histtype="fill",
            facecolor=_facecolor,
            edgecolor=_edgecolor,
            hatch=_hatch,
            linewidth=_linewidth,
        )

    mc_vals = tot_bkg.values()
    mc_vars = tot_bkg.variances()
    mc_err  = np.sqrt(np.where(mc_vars > 0, mc_vars, 0))
    ax.stairs(
        np.maximum(mc_vals + mc_err, 0),
        tot_bkg.axes[0].edges,
        baseline=np.maximum(mc_vals - mc_err, 0),
        fill=True,
        color="gray",
        alpha=0.4,
        label="MC stat. unc.",
        zorder=3,
        hatch="gray"
    )

    # plot data
    if data is not None:
        hep.histplot(
            data,
            ax=ax,
            histtype="errorbar",
            label="Data",
            xerr=True,
            color="k",
            zorder=4,
        )

    # c2v overlay
    hep.histplot(
        [hist_dict[k] * 1000.0 for k in sigs if k in hist_dict],
        ax=ax,
        label=[f"{i}_scaled" for i in sigs if i in hist_dict],
        stack=False,
        histtype="step",
        color=[style[k]["color"] for k in sigs if k in hist_dict],
        # linewidth=[2] + [0] * len( sigs),
    )


    # Set the grid
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    rax.xaxis.grid(True, which="major")
    rax.yaxis.grid(True, which="major")

    # Set the legend
    ax.legend(ncol=2)
    # Reformat the legend
    existing_keys = ax.get_legend_handles_labels()[-1]
    for key in existing_keys:
        if key not in style:
            style[key] = {"label": key}
    order = np.argsort([list(style.keys()).index(i) for i in existing_keys])
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[i] for i in order]
    # labels_bkg= [f"{style[labels[i]]['label']}: {sum((hist_dict[labels[i]]).values()):.2e}" 
    #           if labels[i] in hist_dict else style[labels[i]]['label']
    #           for i in order]
    labels_bkg= [f"{style[labels[i]]['label']}: {sci_2_before_decimal(sum((hist_dict[labels[i]]).values()))}" 
              if labels[i] in hist_dict else labels[i]
              for i in order]
    labels_sig= [f"{style[i.replace('_scaled','')]['label']}\nx1000 <{sci_2_before_decimal(sum((hist_dict[i.replace('_scaled','')]).values()))}>" 
              if "_scaled" in i else i
              for i in labels_bkg]
    # labels= [f"{i}: {sum((hist_dict['data']).values()):.2e}" 
    labels= [f"{i}: {sci_2_before_decimal(sum((hist_dict['data']).values()))}" 
              if "Data" == i else i
              for i in labels_sig]
    # shrink so neither the entry count nor the longest label overflows the two columns
    _longest = max((len(line) for lab in labels for line in lab.split("\n")), default=0)
    _legend_fontsize = "small" if len(labels) <= 8 and _longest <= 26 else "x-small" if _longest <= 34 else "xx-small"
    _ = format_legend(
        ax,
        ncols=2,
        handles_labels=(handles, labels),
        bbox_to_anchor=(1, 1),
        markerscale=0.8,
        fontsize=_legend_fontsize,
        labelspacing=0.4,
        columnspacing=1.5,
        title=legend_title,
    )
    hep.yscale_legend(ax, soft_fail=True)

    # Subplot/ratio
    if ratio_with_uncertainty:
        rh = data.values() - tot_bkg.values()
        rh_unc = np.zeros_like(data.values())
        print(f"Data: {data.values()}")
        print(f"Total background: {tot_bkg.values()}")
        sumw = data.values()
        sumw2 = data.variances()
        # fallback if variances are not stored
        if sumw2 is None:
            sumw2 = sumw  # assume Poisson variance (sumw2 = sumw)
        _lo, _hi = np.abs(hep.error_estimation.poisson_interval(sumw, sumw2) - data.values())

        rh_unc[rh < 0] = _hi[rh < 0]
        rh_unc[rh > 0] = _lo[rh > 0]
        rh /= rh_unc
        rax.set_ylabel(r"$\frac{Data-Bkg}{\sigma_{Data}}$", y=0.5)
        rax.axhline(0, color="gray", ls="--")
        rax.set_ylim(-2.2, 2.2)
        yerr = 1
    else:
        if data is not None:
            rh = data.values() / tot_bkg.values()
            yerr = ratio_uncertainty(data.values(), tot_bkg.values(), "poisson")
        else:
            rh = np.zeros_like(tot_bkg.values())
            yerr = np.zeros_like(tot_bkg.values())
        rax.set_ylabel(r"$\frac{Data}{Bkg}$", y=0.5)
        rax.axhline(1, color="gray", ls="--")
        rax.set_ylim(0, 2.2)

    ## Plotting subplot
    hep.histplot(
        rh,
        tot_bkg.axes[0].edges,
        ax=rax,
        yerr=yerr,
        histtype="errorbar",
        color="k",
        xerr=True,
        zorder=4,
    )

    # Axis limits
    ax.set_xlim(tot_bkg.axes[0].edges[0], tot_bkg.axes[0].edges[-1])

    # Axis labels
    ax.set_xlabel(None)
    ax.set_ylabel("Events / GeV")
    rax.set_xlabel(tot_bkg.axes[0].label)

    return fig, (ax, rax)

def c2vonly_plot(
    hist_dict: dict[hist.Hist],
    # Sample plotting opts
    sigs: list[str],  # List of samples considered signals  | None = None
    bkgs: (
        list[str] | None
    ) = None,  # List of samples considered background - exclude "onto" sample i.e. QCD
    # Style opts
    style: dict | None = None,  # Style YAML
    ratio_with_uncertainty: bool = False,  # Whether to plot ratio/data uncertainty in the ratio
    sort_by_yield: bool = True,  # Whether to sort backgrounds by yield
    legend_title: str | None = None,
):
    style = style.copy()

    # merge histograms according to the style
    merge = extract_mergemap(style)
    hist_dict = merge_hists(hist_dict, merge)

    # --- NEW SORTING LOGIC ---
    if sort_by_yield and bkgs:
        bkg_keys_in_plot = [key for key in bkgs if key in hist_dict]
        bkg_yields = {key: hist_dict[key].sum() for key in bkg_keys_in_plot}
        # print(f"Bkg yields before sorting: {bkg_yields}")
        bkgs = sorted(bkg_yields, key=bkg_yields.get, reverse=True)
        # print(f"Bkg order after sorting by yield: {bkgs}")
    # --- END NEW LOGIC ---

    data = hist_dict.get("data", None)
    all_bkg_keys = bkgs if bkgs else []
    all_bkgs_hists = [hist_dict[k] for k in all_bkg_keys if k in hist_dict]
    tot_bkg = sum(all_bkgs_hists) if all_bkgs_hists else None

    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    plt.subplots_adjust(hspace=0)
    plt.rcParams.update({"font.size": 24})

    _hatch = [None, *[style[k]["hatch"] for k in bkgs + sigs]]
    _edgecolor = [
        style[k]["color"] if h not in ["none", None] else None
        for k, h in zip(bkgs + sigs, _hatch)
    ]
    _facecolor = [
        "none" if h not in ["none", None] else style[k]["color"]
        for k, h in zip(bkgs + sigs, _hatch)
    ]
    
    hep.histplot(
        [hist_dict[k] for k in bkgs + sigs],
        ax=ax,
        label=bkgs + sigs,
        stack=False,
        histtype="fill",
        # facecolor=[style[k]["color"] for k in bkgs + sigs],
        facecolor=[style[k]["color"] for k in bkgs + sigs],
        edgecolor=[style[k]["color"] for k in bkgs + sigs],
    )
    if data is not None:
        hep.histplot(
            data,
            ax=ax,
            histtype="errorbar",
            label="Data",
            xerr=True,
            color="k",
            zorder=4,
        )

    # Set the grid
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    rax.xaxis.grid(True, which="major")
    rax.yaxis.grid(True, which="major")

    # Set the legend
    ax.legend(ncol=2)
    # Reformat the legend
    existing_keys = ax.get_legend_handles_labels()[-1]
    for key in existing_keys:
        if key not in style:
            style[key] = {"label": key}
    order = np.argsort([list(style.keys()).index(i) for i in existing_keys])
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[i] for i in order]
    labels = [style[labels[i]]["label"] for i in order]
    # shrink so neither the entry count nor the longest label overflows the two columns
    _longest = max((len(line) for lab in labels for line in lab.split("\n")), default=0)
    _legend_fontsize = "small" if len(labels) <= 8 and _longest <= 26 else "x-small" if _longest <= 34 else "xx-small"
    _ = format_legend(
        ax,
        ncols=2,
        handles_labels=(handles, labels),
        bbox_to_anchor=(1, 1),
        markerscale=0.8,
        fontsize=_legend_fontsize,
        labelspacing=0.4,
        columnspacing=1.5,
        title=legend_title,
    )
    hep.yscale_legend(ax, soft_fail=True)

    # Subplot/ratio
    rh = []
    yerr = []
    for s in sigs:
        data = hist_dict[s]
        rh.append( data.values() / tot_bkg.values() )
        yerr.append(ratio_uncertainty(data.values(), tot_bkg.values(), "poisson"))

    rax.set_ylabel(r"$\frac{C_{VV}}{SM}$", y=0.5)
    rax.axhline(1, color="gray", ls="--")
    rax.set_ylim(0, 100)

    ## Plotting subplot
    hep.histplot(
        rh,
        tot_bkg.axes[0].edges,
        ax=rax,
        yerr=yerr,
        histtype="errorbar",
        xerr=True,
        zorder=4,
        color=[style[k]["color"] for k in sigs],
    )

    # Axis limits
    ax.set_xlim(data.axes[0].edges[0], data.axes[0].edges[-1])

    # Axis labels
    ax.set_xlabel(None)
    ax.set_ylabel("Events / GeV")
    rax.set_xlabel(tot_bkg.axes[0].label)
    rax.set_yscale('log')

    return fig, (ax, rax)
