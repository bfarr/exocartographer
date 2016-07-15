#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A package for visualizing exo-maps using healpy
"""

import itertools
import numpy as np
import scipy.linalg as sl
import scipy.optimize as so

from matplotlib import pyplot as plt
from matplotlib import animation
import healpy as hp

import exocartographer.gp_alm_illumination_map as gim
import exocartographer.gp_map as gm

from IPython.display import display, clear_output

def anim_to_html(anim):
    """
    Function to help with inline animations in ipython notebooks:

    >>> import exocartographer.visualize as ev
    >>> from matplotlib import animation
    >>> from IPython.display import HTML

    >>> animation.Animation._repr_html_ = ev.anim_to_html

    This first appeared as a blog post on Pythonic Perambulations:
    https://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/
    """
    from tempfile import NamedTemporaryFile

    VIDEO_TAG = """<video controls>
     <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
     Your browser does not support the video tag.
    </video>"""

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=30, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)

def inline_ipynb():
    """
    Inline animations in ipython notebooks.
    """
    from IPython.display import HTML

    animation.Animation._repr_html_ = anim_to_html

def projector(map, map_min=0, map_max=1, view='orth', cmap='gray', flip='geo', title="", **kwargs):
    """Convenient function for plotting healpy projections.

    :param map: Array to plot as a healpy map.

    :param view: (Optional) The type of projection (e.g. 'orth', 'moll', etc.)

    :param cmap: (Optional) Color map to pass to healpy.

    :param flip: (Optional) Plot on the sphere or on the sky (sphere by default).

    :param title: (Optional) Title for the plot.

    :param map_min: (Optional) Minimum value ``min(map)`` by default.

    :param map_max: (Optional) Maximum value ``max(map)`` by default.

    :param kwargs: Passed to the healpy viewer.
    """

    try:
        proj = getattr(hp, view+'view')
    except AttributeError:
        print "ERROR: Couldn't find hp.{}view".format(view)

    proj(map, cmap=cmap, flip=flip, min=map_min, max=map_max, title=title, **kwargs)


def illuminate(logpost, params, map, map_min=0, map_max=1, fignum=1, **kwargs):
    """Generate a movie showing the visibility illumination of the map in time."""
    maps = logpost.visibility_illumination_maps(np.concatenate((params, map)))
    fig = plt.figure(fignum)
    def f(i):
        fig.clf()
        projector(maps[i,:], map_min, map_max, fig=fignum, **kwargs)

    display(animation.FuncAnimation(fig, f, frames=maps.shape[0]))
    plt.close(fig)


def draw_pos_maps(logpost, pbest, proj='orth', show=True, nmaps=None, fignum=1):
    """
        Draw a bunch of map realizations from `logpost`.  If `show` is ``True`` the maps will be
        shown as they're generated.  If `nmaps` is ``None``, an infinite number will be generated,
        and a :exc:`KeyboardInterrupt` will trigger a completion, and return the maps.
    """
    if nmaps is None:
        nmaps = np.inf

    map_counter = itertools.takewhile(lambda i: i < nmaps, itertools.count())

    fig = plt.figure(fignum)
    maps = []
    try:
        for _ in map_counter:
            mp = logpost.draw_map(pbest)
            maps.append(mp)

            if show:
                clear_output(wait=True)
                fig.clf()
                projector(mp, fig=fignum)
                display(fig)

    except KeyboardInterrupt:
        pass

    finally:
        maps = np.array(maps)
        plt.close(fig)

    return maps


def maximize(logpost, p0, method='powell', ftol=None, view='orth', lookback=5, epoch_starts=None, epoch_duration=np.inf, **kwargs):
    pbests = [p0]

    if ftol is None:
        ftol = .1 * len(p0)

    epoch_starts = [logpost.times[0]] if epoch_starts is None else epoch_starts
    def cb(x):
        clear_output(wait=True)

        pbests.append(x)
        pbest = x

        fig1 = plt.figure(num=1)
        axs = fig1.get_axes()
        ax1, ax2 = axs[:len(axs)/2], axs[len(axs)/2:]

        fig2 = plt.figure(num=2)
        ax3, ax4 = fig2.get_axes()

        fig3 = plt.figure(num=3)

        for a1, a2 in zip(ax1, ax2):
            a1.cla()
            a2.cla()

        ax3.cla()
        ax4.cla()

        probs = (logpost.loglikelihood(pbest)-logpost.log_map_prior(pbest), logpost.log_map_prior(pbest), logpost.log_prior(pbest))
        history.append(pbest)
        lnprobs.append(probs)

        for ax, epoch_start in zip(ax1, epoch_starts):
            sel = (logpost.times >= epoch_start) & (logpost.times - epoch_start < epoch_duration)
            ax.errorbar(logpost.times[sel], logpost.intensity[sel], logpost.sigma_intensity[sel], color='k', lw=1.5, fmt='o', markersize=0, capthick=0)

        looking_back = min(len(history), lookback)
        colors = [plt.get_cmap('viridis')(x) for x in np.linspace(0., 1., looking_back)]
        for i, color in zip(range(0, looking_back), colors):
            try:
                p = history[-(i+1)]
                lc = logpost.lightcurve(p)
                for a1, a2, epoch_start in zip(ax1, ax2, epoch_starts):
                    sel = (logpost.times >= epoch_start) & (logpost.times - epoch_start < epoch_duration)
                    a1.plot(logpost.times[sel], lc[sel], color=color)
                    a2.plot(logpost.times[sel], (logpost.intensity - lc)[sel]/(logpost.error_scale(p)*logpost.sigma_intensity)[sel], color=color)
                    a1.set_xlim(logpost.times[sel].min(), logpost.times[sel].max())
                    a2.set_xlim(logpost.times[sel].min(), logpost.times[sel].max())
            except IndexError:
                pass
        for a1, a2 in zip(ax1, ax2):
            l, h = a2.get_xlim()
            a2.plot((l, h), (0, 0), color='k', ls='--', alpha=0.5)
            a1.set_xlabel('time')
            a2.set_xlabel('time')
        ax1[0].set_ylabel('reflectance')
        ax2[0].set_ylabel('standardized residual')

        low, high = len(history)-min(lookback, len(history)), len(history)
        xs = np.arange(low, high)
        for param in params:
            ax3.plot(xs, [logpost.to_params(p)[param] for p in history[low:high]], label=param);
        ax3.legend(loc='lower left', frameon=False)
        lines = ax4.plot(xs[:-1], np.diff(lnprobs[-min(lookback, len(lnprobs)):], axis=0))
        ax4.legend(lines, ['likelihood', 'map_prior', 'prior'], loc='lower left', frameon=False)
        ax3.set_xlabel('steps')
        ax4.set_xlabel('steps')
        ax3.set_ylabel('param')
        ax4.set_ylabel('$\Delta \log$ PDF')

        display(fig1)
        display(fig2)

        fig3.clear()
        projector(logpost.hpmap(pbest), view=this_view, fig=3, **kwargs)
        display(fig3)

        display(probs)

    history = []
    lnprobs = []
    params = logpost.dtype.names

    this_view = view

    fig1, _ = plt.subplots(2, len(epoch_starts), num=1, figsize=(16, 8))
    fig2, _ = plt.subplots(1, 2, num=2, figsize=(16, 4))
    fig3 = plt.figure(num=3)

    cb(p0)
    try:
        func = lambda x: -logpost(x)
        pbest = so.minimize(func, p0, method=method, callback=cb, options={'ftol':ftol}).x
    except KeyboardInterrupt:
        return pbests[-1]

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    return pbest


def sample(logpost, p0, ftol=None, view='orth', lookback=5, epoch_starts=None, epoch_duration=np.inf, nwalkers=1024, nskip=10, nsteps=100, **kwargs):
    import emcee
    import kombine

    ndim = len(p0[0])
    pbests = [p0[0]]
    p = p0

    if ftol is None:
        ftol = .1 * len(p0)

    #sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, threads=8)
    sampler = kombine.Sampler(nwalkers, ndim, logpost)

    epoch_starts = [logpost.times[0]] if epoch_starts is None else epoch_starts
    def cb(x, probs):
        pbest = x[np.argmax(probs)]
        clear_output(wait=True)

        pbests.append(x)

        fig1 = plt.figure(num=1)
        axs = fig1.get_axes()
        ax1, ax2 = axs[:len(axs)/2], axs[len(axs)/2:]

        fig2 = plt.figure(num=2)
        ax3, ax4 = fig2.get_axes()

        fig3 = plt.figure(num=3)

        for a1, a2 in zip(ax1, ax2):
            a1.cla()
            a2.cla()

        ax3.cla()
        ax4.cla()

        probs = (logpost.loglikelihood(pbest), logpost.log_prior(pbest))
        history.append(pbest)
        lnprobs.append(probs)

        for ax, epoch_start in zip(ax1, epoch_starts):
            sel = (logpost.times >= epoch_start) & (logpost.times - epoch_start < epoch_duration)
            ax.errorbar(logpost.times[sel], logpost.intensity[sel], logpost.sigma_intensity[sel], color='k', lw=1.5, fmt='o', markersize=0, capthick=0)

        for i in range(0, lookback):
            try:
                p = history[-(i+1)]
                lc = logpost.lightcurve(p)
                for a1, a2, epoch_start in zip(ax1, ax2, epoch_starts):
                    sel = (logpost.times >= epoch_start) & (logpost.times - epoch_start < epoch_duration)
                    if i == 0:
                        color = 'orange'
                    else:
                        color = 'b'
                    a1.plot(logpost.times[sel], lc[sel], color=color, alpha=1-i*1./lookback)
                    a2.plot(logpost.times[sel], (logpost.intensity - lc)[sel]/(logpost.error_scale(p)*logpost.sigma_intensity)[sel], color=color, alpha=1-i*1./lookback)
                    a1.set_xlim(logpost.times[sel].min(), logpost.times[sel].max())
                    a2.set_xlim(logpost.times[sel].min(), logpost.times[sel].max())
            except IndexError:
                pass
        for a1, a2 in zip(ax1, ax2):
            l, h = a2.get_xlim()
            a2.plot((l, h), (0, 0), color='k', ls='--', alpha=0.5)
            a1.set_xlabel('time')
            a2.set_xlabel('time')
        ax1[0].set_ylabel('reflectance')
        ax2[0].set_ylabel('standardized residual')

        low, high = len(history)-min(lookback, len(history)), len(history)
        xs = np.arange(low, high)
        for param in params:
            ax3.plot(xs, [logpost.to_params(p)[param] for p in history[low:high]], label=param);
        ax3.legend(loc='lower left', frameon=False)
        lines = ax4.plot(xs[:-1], np.diff(lnprobs[-min(lookback, len(lnprobs)):], axis=0)/lnprobs[-1])
        ax4.legend(lines, ['likelihood', 'prior'], loc='lower left', frameon=False)
        ax3.set_xlabel('steps')
        ax4.set_xlabel('steps')
        ax3.set_ylabel('param')
        ax4.set_ylabel('$\Delta \log$ PDF')

        display(fig1)
        display(fig2)

        fig3.clear()
        projector(logpost.hpmap(pbest), view=this_view, fig=3, **kwargs)
        display(fig3)

        display(probs)
        try:
            prob_diff = np.sum(np.diff(lnprobs[-2:], axis=0)/lnprobs[-1])
        except IndexError:
            prob_diff = np.inf
        return prob_diff

    history = []
    lnprobs = []
    params = logpost.dtype.names

    this_view = view

    fig1, _ = plt.subplots(2, len(epoch_starts), num=1, figsize=(16, 8))
    fig2, _ = plt.subplots(1, 2, num=2, figsize=(16, 4))
    fig3 = plt.figure(num=3)

    try:
        delta_p = np.inf
        while delta_p < 0 or delta_p > ftol or delta_p == 0.:
            #p, prob, _ = sampler.run_mcmc(p, nskip)
            p, prob, _ = sampler.run_mcmc(nskip, p)
            delta_p = cb(p, prob)
            print(delta_p)
    except KeyboardInterrupt:
        return pbests[-1]

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    return p

## DRY VIOLATION: copied from above.  Clean this up!
def differential_evolution(logpost, bounds, view='orth', lookback=5):
    pbests = []

    def cb(x, convergence=None):
        clear_output(wait=True)

        pbests.append(x)
        pbest = x

        fig1 = plt.figure(num=1)
        ax1, ax2 = fig1.get_axes()

        fig2 = plt.figure(num=2)
        ax3, ax4 = fig2.get_axes()

        fig3 = plt.figure(num=3)

        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()

        xlow, xhigh = min(logpost.times), max(logpost.times)

        probs = (logpost(pbest), logpost.log_prior(pbest))
        history.append(pbest)
        lnprobs.append(probs)

        ax1.errorbar(logpost.times, logpost.intensity, logpost.sigma_intensity, color='k', lw=1.5, capthick=0)
        for i in range(0, lookback):
            try:
                p = history[-(i+1)]
                lc = logpost.lightcurve(p)
                ax1.plot(logpost.times, lc, color='b', alpha=1-i*1./lookback)
                ax2.plot(logpost.times, (logpost.intensity - lc)/(logpost.error_scale(p)*logpost.sigma_intensity), color='b', alpha=1-i*1./lookback)
            except IndexError:
                pass
        ax2.plot((xlow, xhigh), (0, 0), color='k', ls='--', alpha=0.5)
        ax1.set_xlim(xlow, xhigh)
        ax2.set_xlim(xlow, xhigh)
        ax1.set_xlabel('time')
        ax1.set_ylabel('intensity')
        ax2.set_xlabel('time')
        ax2.set_ylabel('standardized residual')

        low, high = len(history)-min(lookback, len(history)), len(history)
        xs = np.arange(low, high)
        for param in params:
            ax3.plot(xs, [logpost.to_params(p)[param] for p in history[low:high]], label=param);
        ax3.legend()
        lines = ax4.plot(xs, lnprobs[-min(lookback, len(lnprobs)):])
        ax4.legend(lines, ['log(post)', 'log(prior)'])
        ax3.set_xlabel('steps')
        ax4.set_xlabel('steps')
        ax3.set_ylabel('param')
        ax4.set_ylabel('log PDF')

        display(fig1)
        display(fig2)

        fig3.clear()
        projector(logpost.hpmap(pbest), view=this_view, fig=3)
        display(fig3)

        display(probs)

    history = []
    lnprobs = []
    params = logpost.dtype.names

    this_view = view

    fig1, _ = plt.subplots(2, 1, num=1, figsize=(16, 8))
    fig2, _ = plt.subplots(1, 2, num=2, figsize=(16, 4))
    fig3 = plt.figure(num=3)

    try:
        res = so.differential_evolution(lambda x: -logpost(x), bounds, callback=cb)
        pbest = res.x
    except KeyboardInterrupt:
        return pbests[-1]

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    return pbest


