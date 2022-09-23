#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os,sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # F401 unused import --> to ignore !

#import hvplot
#import hvplot.xarray
from pyproj import Proj, transform

# This script plot accumulated precipiation over the Grandes-Rousses domain (can
# be easily adapted to plot over different domain or the whole input domain)
# It interpolates a DEM grid on the grided precipitation data to plot 3D
# precipitation fields

print('USAGE : plot_radar_cumuls.py inputfile')
datadir = '/home/vernaym/These/DATA'
savedir = '/home/vernaym/These/figures'

datebegin = sys.argv[1]
dateend   = sys.argv[2]
print(f'Datebegin={datebegin}')
print(f'Dateend={dateend}')


# Domaine des Grandes Rousses
domaine = 'GrandesRousses'
extract_dom = dict(
    latmax = 45.240,
    latmin = 44.990,
    lonmin = 6.010,
    lonmax = 6.490,
)
latmin = extract_dom['latmin']
lonmin = extract_dom['lonmin']
latmax = extract_dom['latmax']
lonmax = extract_dom['lonmax']

landmarks = {
        "Alpe d'Huez" : dict(lon=6.070, lat=45.092, alt=1800, marker='o'),
        "Les 2 Alpes" : dict(lon=6.127, lat=45.013, alt=1800, marker='o'),
        "Lautaret"    : dict(lon=6.408, lat=45.038, alt=2058, marker='X'),
        "La Meije"    : dict(lon=6.311, lat=45.008, alt=3500, marker='^'),  # real alt = 3984
        "Pic Blanc"   : dict(lon=6.131, lat=45.128, alt=3000, marker='^'),  # real alt = 3333
    }

norm = plt.Normalize()
#colormap = plt.cm.gist_ncar
colormap = plt.cm.YlGnBu

cumul = dict()

def read_data(filename):
    filename = os.path.join(datadir, filename)
    ds = xr.open_dataset(filename)
    ds = ds.where((ds.lon>=lonmin) & (ds.lon<=lonmax) & (ds.lat>=latmin) & (ds.lat<=latmax), drop=True)
    cumul = ds.sum('time').rr
    return cumul

def add_landmarks(ax):
    for landmark, infos in landmarks.items():
        ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)

def finalize_fig(fig, im, savename):
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(os.path.join(savedir, savename), format='pdf')


def plot_arome():
    filename = f'arome_{datebegin}_{dateend}_{domaine}.nc'
    cumul = read_data(filename)
    fig, ax = plt.subplots(figsize=(13,6))
    im = cumul.plot(ax=ax, cmap=colormap)
    add_landmarks(ax)
    finalize_fig(fig, im, f'CUMULS_AROME_{datebegin}_{dateend}.pdf')

def plot_pearome(model):
    vmin = None
    vmax = None
    for member in range(1, 17):
        filename = f'{model}_{member:03d}_{datebegin}_{dateend}_{domaine}.nc'
        cumul[member] = read_data(filename)
        vmin = min(np.nanmin(cumul[member]), vmin) if vmin is not None else np.nanmin(cumul[member])
        vmax = max(np.nanmax(cumul[member]), vmax) if vmax is not None else np.nanmax(cumul[member])

    # Plot
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
    i = 0
    j = 0
    for member, field in cumul.items():
        im = field.plot(ax=ax[i,j], add_colorbar=False, vmin=0, vmax=vmax, cmap=colormap)
    #im = field.plot(ax=ax[i,j], add_colorbar=False, vmin=vmin, vmax=vmax, cmap=colormap)
        add_landmarks(ax[i,j])
        ax[i,j].set_aspect('equal')
        ax[i,j].axis('off')
        ax[i,j].set_title(f'Member {member:03d}')

        j = j + 1
        if j==4:
            j = 0
            i = i + 1

    cbar_ax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=f'Total precipitation between {datebegin} and {dateend} (mm)')
    if model == 'aspearome':
        outname = f'CUMULS_ASPEAROME_{datebegin}_{dateend}.pdf'
    elif model == 'stats':
        outname = f'CUMULS_Q50_{datebegin}_{dateend}.pdf'
    finalize_fig(fig, im, outname)

if __name__ == "__main__":

    plot_arome()
    plot_pearome('stats')
    plot_pearome('aspearome')
