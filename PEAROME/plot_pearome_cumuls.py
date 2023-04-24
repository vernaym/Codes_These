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

print('USAGE : plot_radar_cumuls.py [datebegin] [dateend]')
datadir = '/home/vernaym/These/DATA'
savedir = '/home/vernaym/These/figures'

if len(sys.argv) > 1:
    datebegin = sys.argv[1]
    dateend   = sys.argv[2]
else:
    datebegin = '2021102806'
    dateend  = '2022060206'
print(f'Datebegin={datebegin}')
print(f'Dateend={dateend}')

# Domaine des Grandes Rousses
domain = 'GrandesRousses'
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

norm = plt.Normalize(vmin=300, vmax=1200)
#norm = plt.Normalize()
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
    filename = f'arome_{datebegin}_{dateend}_{domain}.nc'
    cumul = read_data(filename)
    fig, ax = plt.subplots(figsize=(13,6))
    im = cumul.plot(ax=ax, cmap=colormap)
    add_landmarks(ax)
    finalize_fig(fig, im, f'CUMULS_AROME_{datebegin}_{dateend}.pdf')

def plot_pearome(model):
    vmin = None
    vmax = None
    for member in range(1, 17):
        filename = f'{model}_{member:03d}_{datebegin}_{dateend}_{domain}.nc'
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

def plot_mean_pearome_3D():
    filename = os.path.join(datadir, f'CUMUL_aspearome_mean_{datebegin}_{dateend}_alp.nc')
    if not os.path.exists(filename):
        filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_{datebegin}_{dateend}_alp_hourly.nc") for mb in range(1,17)]
        pearome = xr.open_mfdataset(filenames, combine='nested', concat_dim='member', chunks={'time': 24})  # Setting chunks is critical (read the doc !)
        ds = pearome.mean(dim='member').sum(dim='time')
        ds.to_netcdf(filename)
    else:
        ds = xr.open_dataset(filename)
    mnt = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_ALPES_WGS84_250m_bilinear.nc')  # Pour tracer sur toutes les Alpes
    #tmp = mnt.interp(lon=ds.lon, lat=ds.lat, method='nearest')  # Pour interpoller le MNT sur la grille PEAROME
    tmp = ds.interp(lon=mnt.lon, lat=mnt.lat)  # Pour interpoller la PEAROME sur le  MNT
    mnt['Z'] = mnt['Band1']
    mnt['Z'].values = np.nan_to_num(mnt['Z'].values)

    latmin = extract_dom['latmin']
    lonmin = extract_dom['lonmin']
    latmax = extract_dom['latmax']
    lonmax = extract_dom['lonmax']

    # On selectionne le sous domaine d'intéret avant de plotter...
    mnt = mnt.where((mnt.lon>=lonmin) & (mnt.lon<=lonmax) & (mnt.lat>=latmin) & (mnt.lat<=latmax), drop=True)
    X = mnt['lon'].values
    Y = mnt['lat'].values
    X, Y = np.meshgrid(X, Y)
    Z = mnt['Z']

    # define pixel colors
    #radar = ds.where((ds.longitude>=lonmin) & (ds.longitude<=lonmax) & (ds.latitude>=latmin) & (ds.latitude<=latmax), drop=True).rr_cumul
    rr = tmp.where((tmp.lon>=lonmin) & (tmp.lon<=lonmax) & (tmp.lat>=latmin) & (tmp.lat<=latmax), drop=True).rr
    #colors = plt.cm.coolwarm(norm(np.nan_to_num(radar.values)))
    colors = plt.cm.YlGnBu(norm(np.nan_to_num(rr.values)))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(elev=50., azim=135)  # Set point of view
    surf = ax.plot_surface(X=X, Y=Y, Z=Z, linewidth=0, antialiased=False, facecolors=colors)
    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    ax.set_xlabel('Longitude', labelpad=20)
    ax.set_ylabel('Latitude', labelpad=20)
    ax.set_zlabel('Elevation (m)')
    ax.set_zlim(0., 3500.)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGnBu), ax=ax, shrink=0.75, aspect=8, label=f'PEAROME mean cumulated precipitation \n between {datebegin} and {dateend} (mm)')
    plt.tight_layout()
    plt.savefig(f'{savedir}/CUMUL3D_ASPEAROME_{datebegin}_{dateend}.pdf', format='pdf')


if __name__ == "__main__":

#    plot_mean_pearome_3D()
    plot_arome()
#    plot_pearome('stats')
#    plot_pearome('aspearome')
