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

import make_mask

if len(sys.argv) > 1:
    domain = sys.argv[1]
else:
    domain = 'MontBlanc'

domain_coords = dict(
        #GrandesRousses = dict(latmax=45.240, latmin=44.990, lonmin=6.010, lonmax = 6.490),
        GrandesRousses = dict(latmax=45.4, latmin=44.9, lonmin=5.8, lonmax = 6.6),
        NorthernAlps   = dict(lonmin=6.0, lonmax=6.9, latmin=45.6, latmax=46.35),
        CentralAlps    = dict(lonmin=5.6, lonmax=7.0, latmin=45.0, latmax=45.6),
        SouthernAlps   = dict(lonmin=5.7, lonmax=7.0, latmin=44.2, latmax=45.0),
        HauteSavoie    = dict(lonmin=6.45, lonmax=6.95, latmin=45.67, latmax=46.35),
        MontBlanc      = dict(lonmin=6.45, lonmax=7.1, latmin=45.65, latmax=46.1),
        Savoie         = dict(lonmin=6.0, lonmax=7.2, latmin=45.1, latmax=45.9),
        Isere          = dict(lonmin=5.54, lonmax=6.19, latmin=44.89, latmax=45.16),
        Brianconnais   = dict(lonmin=6.48, lonmax=6.95, latmin=44.67, latmax=44.95),
        HautesAlpes    = dict(lonmin=6.1, lonmax=7.1, latmin=44.4, latmax=45.2),
        AlpesSud       = dict(lonmin=6.56, lonmax=6.92, latmin=44.18, latmax=44.49),
        alp            = dict(latmax=46.450, latmin=44.100, lonmin=5.400, lonmax=7.200),
)

figsize = dict(
        alp            = (14,16),
        GrandesRousses = (15,7),
        HauteSavoie    = (12,12),
        HautesAlpes    = (16,10),
        MontBlanc      = (15,10),
        Savoie         = (16,8),
        Isere          = (16,8),
)

#datadir = f'/home/vernaym/workdir/ASSIMILATION/mask/alp/nivometeo'
#savedir = f'/home/vernaym/workdir/ASSIMILATION/mask/{domain}/nivometeo'
datadir = f'/home/vernaym/workdir/ASSIMILATION/mask/alp'
savedir = f'/home/vernaym/workdir/ASSIMILATION/mask/{domain}'

# Coordonnées du Mont Blanc :
#lat = 45.83
#lon = 6.86
#extract_lat = [l/100 for l in range(int(lat*100)-15, int(lat*100)+16, 1)]
#extract_lon = [l/100 for l in range(int(lon*100)-15, int(lon*100)+16, 1)]
extract_lat = np.round(np.arange(domain_coords[domain]['latmin'], domain_coords[domain]['latmax'], 0.01, dtype=float), 2)
extract_lon = np.round(np.arange(domain_coords[domain]['lonmin'], domain_coords[domain]['lonmax'], 0.01, dtype=float), 2)

def plot_and_save(field, name, cmap=plt.cm.Greys, vmin=None, vmax=None, scores=None, subdir=''):
    fig, ax = plt.subplots(figsize=figsize[domain])
    #ax = plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax, scores=scores)
    ax = make_mask.plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax, scores=scores)
    #fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf', layout='tight')
    fig.tight_layout()
    fig.savefig(os.path.join(savedir, subdir, f'{name}.pdf'), format='pdf')
    field.to_netcdf(os.path.join(savedir, subdir, f'{name}.nc').encode('utf-8'))  # WARNING : encode ncessary if name contains a formatted flot

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


if __name__ == "__main__":

    d0 = 0.15
    d0 = 0.25
    for subdir in ['', 'nivometeo']:
        if not os.path.exists(os.path.join(savedir, subdir)):
            os.makedirs(os.path.join(savedir, subdir))
        #fic_error = os.path.join(datadir, subdir, f'Observation_error_{d0}_alp.nc')

        # Extract AROME precipitation accumulation
#        fic = os.path.join('/home/vernaym/These/DATA', f'CUMUL_AROME.nc')
#        cumul = xr.open_dataset(fic)
#        reduced_cumul = cumul.sel({'lat':np.intersect1d(extract_lat, cumul.lat), 'lon':np.intersect1d(extract_lon, cumul.lon)})
#        make_mask.plot(reduced_cumul, '2021103000', '2022060200', categories=False, biascorrection=False, scores=False)

#        Extract ANTILOPE precipitation accumulation
        fic = os.path.join('/home/vernaym/These/DATA', f'CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc')
        cumul = xr.open_dataset(fic)
        reduced_cumul = cumul.sel({'lat':np.intersect1d(extract_lat, cumul.lat), 'lon':np.intersect1d(extract_lon, cumul.lon)})
        make_mask.plot(reduced_cumul, '2021103000', '2022060200', categories=False, biascorrection=False, scores=True, dom=domain)

        fic_error = os.path.join(datadir, subdir, f'Observation_uncertainty.nc')
        fic_ratio = os.path.join(datadir, subdir, f'Estimated_ratio.nc')
        error = xr.open_dataarray(fic_error)
        ratio = xr.open_dataarray(fic_ratio)
        reduced_error = error.sel({'lat':np.intersect1d(extract_lat, error.lat), 'lon':np.intersect1d(extract_lon, error.lon)})
        reduced_ratio = ratio.sel({'lat':np.intersect1d(extract_lat, ratio.lat), 'lon':np.intersect1d(extract_lon, ratio.lon)})
        plot_and_save(reduced_error, f'Observation_error', vmin=1, vmax=10, cmap=plt.cm.YlOrBr, subdir=subdir)
        plot_and_save(reduced_ratio, f'Estimated_ratio', vmin=0.5, vmax=1.5, cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, subdir=subdir)

