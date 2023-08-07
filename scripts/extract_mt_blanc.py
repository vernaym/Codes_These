#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 31/03/2023

import os, sys
import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import palettable


def plot_and_save(field, name, cmap=plt.cm.Greys, vmin=None, vmax=None, scores=None):
    fig, ax = plt.subplots()
    ax = plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax, scores=scores)
    #fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf', layout='tight')
    fig.tight_layout()
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf')
    field.to_netcdf(os.path.join(savedir, f'{name}.nc'))  # WARNING : does not work if nctoolkit is installed

def plot_field(fig, ax, field, cmap=plt.cm.Greys, vmin=None, vmax=None, scores=None, colorbar=True):

    if vmin is None:
        vmin = np.nanmin(field)
    if vmax is None:
        vmax = np.nanmax(field)
    if cmap == 'custom':
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkviolet", "green", "orange", "red"], 5)
        thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]  # TODO : vérier la coéhrence des seuils entre les figures
        #thresholds = [0., 0.5, 0.90, 1.1, 1.5, 10]
        norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)
        cml = field.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
    else:
        cml = field.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)

    if scores is not None:
        if field.name == 'ratio':
            # Plot scores with same cmap since it is the same information
            sc = add_scores(scores, ax, mycmap=cmap, vmin=vmin, vmax=vmax)
        else:
            sc = add_scores(scores, ax)

    if colorbar:
        cb = fig.colorbar(cml)
        cb.set_label(field.name, fontsize=24)
        cb.ax.tick_params(labelsize=20)

    return ax

datadir = '/home/vernaym/workdir/ASSIMILATION/mask/alp/'
savedir = f'/home/vernaym/workdir/ASSIMILATION/mask/MontBlanc'
# Coordonnées du Mont Blanc :
lat = 45.83
lon = 6.86

extract_lat = [l/100 for l in range(int(lat*100)-15, int(lat*100)+16, 1)]
extract_lon = [l/100 for l in range(int(lon*100)-15, int(lon*100)+16, 1)]

fic_error = os.path.join(datadir, 'Observation_error_0.2_alp.nc')
fic_ratio = os.path.join(datadir, 'Estimated_ratio_alp_0.2.nc')
error = xr.open_dataarray(fic_error)
ratio = xr.open_dataarray(fic_ratio)

reduced_error = error.sel({'lat':extract_lat, 'lon':extract_lon})
reduced_ratio = ratio.sel({'lat':extract_lat, 'lon':extract_lon})

plot_and_save(reduced_error, 'Observation_error_MontBlanc', vmin=1, vmax=16, cmap=plt.cm.YlOrBr)
plot_and_save(reduced_ratio, 'Estimated_ratio_MontBlanc', vmin=0.2, vmax=1.8, cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap)

