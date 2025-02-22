"""Helper function for plotting histograms and ratios with errorbars or bands
Originally from Erik Ganster, only light adaptations
"""
import numpy as np


# adapted from stackoverflow: https://stackoverflow.com/questions/36699155/how-to-get-color-of-most-recent-plotted-line-in-pythons-plt
# can also be used for calculating bin centers
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def plot_hist_errorbar(
    ax, hist, bins, yerror, capsize=4.0, capthick=2, **kwargs
):
    # Remove keywords which are for errorbar from kwargs
    kwargs.pop("capsize", None)
    kwargs.pop("capthick", None)

    lines = ax.plot(
        bins, np.append(hist, hist[-1]), drawstyle="steps-post", **kwargs
    )

    if not np.all(yerror == 0.):
        # skip errorbars that look weird if alpha!=0 is used
        ax.errorbar(
            np.mean(rolling_window(bins, 2), axis=1),
            hist,
            yerr=yerror,
            ls="none",
            capsize=capsize,
            capthick=capthick,
            color=lines[0].get_color(),
            alpha=lines[0].get_alpha()
        )

    return lines


def plot_hist_band(ax, hist, bins, yerror, **kwargs):

    p = ax.stairs(hist, bins, baseline=None, **kwargs)
    ax.stairs(
        hist + yerror,
        bins,
        baseline=hist - yerror,
        fill=True,
        alpha=0.25,
        color=p.get_edgecolor()
    )

    return p


def plot_ratio_band(
    ax, hist, bins, yerror, hist_baseline, yerr_baseline=None, **kwargs
):

    ratio = hist / hist_baseline

    if yerr_baseline is not None:
        # propagate error on ratio
        yerror_ratio = ratio * np.sqrt(
            (yerror / hist)**2 + (yerr_baseline / hist_baseline)**2
        )
    else:
        yerror_ratio = yerror / hist_baseline

    p = ax.stairs(ratio, bins, baseline=None, **kwargs)
    ax.stairs(
        ratio + yerror_ratio,
        bins,
        baseline=ratio - yerror_ratio,
        fill=True,
        alpha=0.25,
        color=p.get_edgecolor()
    )

    return p


def plot_ratio_errorbar(
    ax, hist, bins, yerror, hist_baseline, yerr_baseline=None, **kwargs
):
    """
    
    """
    ratio = hist / hist_baseline

    if yerr_baseline is not None:
        # propagate error on ratio
        yerror_ratio = ratio * np.sqrt(
            (yerror / hist)**2 + (yerr_baseline / hist_baseline)**2
        )
    else:
        yerror_ratio = yerror / hist_baseline

    lines = plot_hist_errorbar(ax, ratio, bins, yerror_ratio, **kwargs)

    return lines


def plot_data_hist_errorbar(ax, hist, bins, yerror, **kwargs):
    """Draw data as points, not histogram"""

    lines = ax.errorbar(
        np.mean(rolling_window(bins, 2), axis=1),
        hist,
        yerr=yerror,
        ls="none",
        capsize=4.0,
        capthick=2,
        **kwargs
    )

    return lines


def plot_data_ratio_errorbar(
    ax, hist, bins, yerror, hist_baseline, yerr_baseline=None, **kwargs
):
    """Draw data as points, not histogram"""
    ratio = hist / hist_baseline

    if yerr_baseline is not None:
        # propagate error on ratio
        yerror_ratio = ratio * np.sqrt(
            (yerror / hist)**2 + (yerr_baseline / hist_baseline)**2
        )
    else:
        yerror_ratio = yerror / hist_baseline

    lines = plot_data_hist_errorbar(ax, ratio, bins, yerror_ratio, **kwargs)

    return lines


# wrapper for numpy.histogram for calculating uncertainty from weighted entries
def make_hist_error(samples, bins, weights=None, normed=False):
    # set weights if not give
    if weights is None:
        weights = np.ones_like(samples, dtype=int)

    hist, bins = np.histogram(samples, bins, weights=weights)
    if normed:
        norm = 1. / np.diff(bins) / hist.sum()
        hist = hist * norm

    # calculate error
    weights_hist, _ = np.histogram(samples, bins, weights=np.power(weights, 2))
    yerror = np.sqrt(weights_hist)

    if normed:
        yerror = yerror * norm

    return hist, yerror


# wrapper for numpy.histogram2d for calculating uncertainty
#  from weighted entries
def make_hist2d_error(x, y, binning, weights=None, normed=False):
    # set weights if not give
    if weights is None:
        weights = np.ones_like(x)

    hist, _, _ = np.histogram2d(x, y, bins=binning, weights=weights)
    if normed:
        raise NotImplementedError(
            "Normalization not implemented for 2D histograms"
        )
        # norm = 1./np.diff(bins)/hist.sum()
        # hist = hist*norm

    # calculate error
    weights_hist, _, _ = np.histogram2d(
        x, y, bins=binning, weights=np.power(weights, 2)
    )
    yerror = np.sqrt(weights_hist)

    # if normed:
    #     yerror = yerror*norm

    return hist, yerror
