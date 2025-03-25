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

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def to_xarray(array, field, varname='rr'):
    output = xr.DataArray(
    name   = varname,
    data   = array,
    dims   = ["lat", "lon"],
    coords = dict(lon=field.lon, lat=field.lat),
    #attrs  = dict(description="Difference between each pixel cumul and the max of its neighbours"),
    )
    return output

def plot_scatter(ax, reference, model, color=None, addtext=None, xaxis=True, yaxis=True, legend=False, lims=[0, 45], cmap='YlGnBu'):
    from sklearn.linear_model import LinearRegression
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

    #fig,ax = plt.subplots()
    # TODO : plot points with ratio inversion in red
    if color is None:
        ax.scatter(ref, mod, marker='+')  # scatterplot ref vs estimation
    else:
        sc = ax.scatter(ref, mod, c=color, cmap=cmap, marker='+', s=80)
        #sc = ax.scatter(x, y, c=color, cmap='YlGnBu', marker='+')
        #cb = fig.colorbar(sc, label='Precipitation (mm)', size=16)

    # Add linear regression
    ax.plot(x, z, color='blue', linewidth=1.5, label=f'Slope={reg.coef_[0]:.3f}, Intercept={reg.intercept_:.3f}\nR²={r2:.4}, bias={bias}, rmse={rmse}')  # plot linear regression line

    if addtext is not None:
        addtext = addtext[mask]
        for idx,text in enumerate(addtext):
            ax.text(ref[idx], mod[idx], int(text))

    # Add x/y mean lines for article figures
    xmean = np.mean(x)
    ymean = np.mean(y)
    ax.plot([xmean, xmean], lims, color='red', linestyle='--', linewidth=1.5)
    ax.plot(lims, [ymean, ymean], color='red', linestyle='--', linewidth=1.5)

    # Plot bissectrice and adjuste axes limits
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=1.5)
    #ax.plot(lims, [1, 1], color='k', linestyle='--', linewidth=0.5)
    #ax.plot([1, 1], lims, color='k', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    if yaxis:
        ax.set_yticks(np.linspace(lims[0], lims[1], 5))
        ax.tick_params(
            axis='y',
            labelsize=14)
    else:
        # Remove y axislabel
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off
    if xaxis:
        ax.set_xticks(np.linspace(lims[0], lims[1], 5))
        ax.tick_params(
            axis='x',
            labelsize=14)
    else:
        # Remove x axislabel
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False,      # ticks along the bottom edge are off
            bottom=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

    if legend:
        ax.legend(fontsize=10)
    #fig.tight_layout()
    #fig.subplots_adjust(left=0.005, top=0.98, right=0.99, bottom=0.1)
    #fig.savefig(os.path.join(savedir, savename), format='pdf')
    #plt.close(fig)

    return sc
