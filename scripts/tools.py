#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 20/07/2023

import os, sys
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import xarray as xr
import time

from sklearn.linear_model import LinearRegression

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import palettable

def hourly_to_daily(data):
    """
    Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
    Problem : the xarray tools to do that allows only accumulations between 0h and 23h.
    solution : shift time serie by 7h, compute 24h accumulations and shift back !
    """
    data['time'] = data.time-np.timedelta64(7, 'h')
    t1 = time.time()
    data = data.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
    t2 = time.time()
    print(f'Computing daily antilope took {(t2-t1)*1000}.ms')
    data['time'] = data.time+np.timedelta64(30, 'h')

    return data

def plot_scatter(reference, model, xlabel, ylabel, savename, savedir, color=None, addtext=None):
    ref = reference.flatten()
    mod = model.flatten()
    mask = np.where(~np.isnan(ref) & ~np.isnan(mod))
    #ref = ref[mask]
    #mod = mod[mask]
    bias = np.round(np.mean(mod - ref),3)
    rmse = np.round(np.sqrt(np.mean((mod-ref)**2)), 3)
    x = ref.reshape((-1,1))
    y = mod
    reg = LinearRegression().fit(x, y)
    z = reg.predict(x)
    r2 = np.round(reg.score(x, y), 3)

    fig,ax = plt.subplots()
    # TODO : plot points with ratio inversion in red
    if color is None:
        ax.scatter(ref, mod, marker='+')  # scatterplot ref vs estimation
    else:
        sc = ax.scatter(ref, mod, c=color, cmap='YlGnBu', marker='+')
        #sc = ax.scatter(x, y, c=color, cmap='YlGnBu', marker='+')
        cb = fig.colorbar(sc, label='Precipitation (mm)')

    # Add linear regression
    ax.plot(x, z, color='blue', linewidth=1, label=f'Slope={reg.coef_[0]:.3f}, Intercept={reg.intercept_:.3f}\nR²={r2:.4}, bias={bias}, rmse={rmse}')  # plot linear regression line

    if addtext is not None:
        addtext = addtext[mask]
        for idx,text in enumerate(addtext):
            ax.text(ref[idx], mod[idx], int(text))

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # Plot bissectrice and adjuste axes limits
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.plot(lims, [1, 1], color='k', linestyle='--', linewidth=0.5)
    ax.plot([1, 1], lims, color='k', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, savename), format='pdf')
    plt.close(fig)
