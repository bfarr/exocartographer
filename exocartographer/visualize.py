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

import exocartographer.gp_illumination_map as gim
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

def projector(map, map_min=None, map_max=None, view='orth', cmap='gray', flip='geo', title="", **kwargs):
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

    if map_min is None:
        map_min = np.min(map)
    if map_max is None:
        map_max = np.max(map)

    proj(map, cmap=cmap, flip=flip, min=map_min, max=map_max, title=title, **kwargs)


def illuminate(logpost, params, map, map_min=None, map_max=None, fignum=1):
    """Generate a movie showing the visibility illumination of the map in time."""
    maps = logpost.visibility_illumination_maps(np.concatenate((params, map)))
    if map_min is None:
        map_min = np.min(maps)
    if map_max is None:
        map_max = np.max(maps)

    fig = plt.figure(fignum)
    def f(i):
        fig.clf()
        projector(maps[i,:], map_min, map_max, fig=fignum)

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


def powell(logpost, p0, view='orth', lookback=5):
    global pbest
    global history
    global lnprobs
    global this_view

    def cb(x):
        clear_output(wait=True)
        global pbest
        global history
        global lnprobs

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

        probs = (logpost(pbest), logpost.log_prior(pbest), logpost.log_mapmarg_likelihood(pbest))
        history = np.append(history, np.atleast_2d(pbest), axis=0)
        lnprobs = np.append(lnprobs, np.atleast_2d(probs), axis=0)

        ax1.errorbar(logpost.times, logpost.intensity, logpost.sigma_intensity, color='k', lw=1.5, capthick=0)
        for i in range(0, lookback):
            try:
                p = history[-(i+1)]
                lc = logpost.lightcurve_map(np.concatenate((p, logpost.mbar(p))))
                ax1.plot(logpost.times, lc, color='b', alpha=1-i*1./lookback)
                ax2.errorbar(logpost.times, (logpost.intensity - lc)/logpost.sigma_intensity,
                             np.ones_like(logpost.sigma_intensity), color='b', lw=1.5, capthick=0, alpha=1-i*1./lookback)
            except IndexError:
                pass
        ax2.plot((xlow, xhigh), (0, 0), color='k', ls='--', alpha=0.5)
        ax1.set_xlim(xlow, xhigh)
        ax2.set_xlim(xlow, xhigh)

        low, high = history.shape[0]-min(lookback, history.shape[0]), history.shape[0]
        xs = np.arange(low, high)
        ax3.plot(xs, history[low:high]);
        ax4.plot(xs, lnprobs[-min(lookback, len(lnprobs)):])

        display(fig1)
        display(fig2)

        fig3.clear()
        projector(logpost.mbar(pbest), view=this_view, fig=3)
        display(fig3)

        display(probs)

    history = np.empty((0, p0.shape[0]))
    lnprobs = np.empty((0, 3))

    this_view = view

    fig1, _ = plt.subplots(2, 1, num=1, figsize=(16, 8))
    fig2, _ = plt.subplots(1, 2, num=2, figsize=(16, 4))
    fig3 = plt.figure(num=3)

    pbest = so.fmin_powell(lambda x: -logpost(x), p0, callback=cb)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    return pbest



