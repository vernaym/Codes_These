#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 31/03/2023

import os, sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import pandas as pd
import xarray as xr

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import palettable

from scipy.stats import norm, gamma
from scipy import signal
import random
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix, diags
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LinearRegression

from These.radar import Preprocessing_ANTILOPE
import make_mask
import scores

if len(sys.argv) > 1:
    domain = sys.argv[1]
else:
    domain = 'MontBlanc'

domain_coords = dict(
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
        HautesAlpes    = (16,9),
        MontBlanc      = (16,10),
        Savoie         = (16,8),
        Isere          = (16,8),
)

savedir = '/home/vernaym/These/figures/illustration/methode_correction'

extract_lat = np.round(np.arange(domain_coords[domain]['latmin'], domain_coords[domain]['latmax'], 0.01, dtype=float), 2)
extract_lon = np.round(np.arange(domain_coords[domain]['lonmin'], domain_coords[domain]['lonmax'], 0.01, dtype=float), 2)
# Coordonnées du Mont Blanc :
lat = 45.83
lon = 6.87

def plot(field, name, cmap=plt.cm.YlGnBu, vmin=None, vmax=None, scores=None):
    if vmin is None:
        vmin = np.min(field)
    if vmax is None:
        vmax = np.max(field)
    fig, ax = plt.subplots(figsize=figsize[domain])
    make_mask.plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax, scores=scores)
    circle = plt.Circle((lon, lat), 0.14, color='red', fill=False, linewidth=3)
    ax.plot(lon, lat, color='red', marker='+', markersize=10)
    ax.add_artist(circle)
    #plt.Circle((lon, lat), 0.15, color='k', fill=False, linewidth=2)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf')
    plt.close(fig)

def plot_super_ensemble(ensemble, weights, initial_obs, new_obs):
        #mu, std, pond, product, label=None, ax=None, reference=None, initial_obs=None):
    ensemble = ensemble.flatten()
    weights = weights.flatten()
    ensemble = ensemble[weights>0]
    weights = weights[weights>0]
    W = np.sum(weights)

    mean = np.sum(weights*ensemble)/W
    std = np.sqrt(np.sum((weights*ensemble-mean)**2)/W)

    fig, ax = plt.subplots()
    ax.hist(ensemble, density=True, bins=np.arange(np.floor(np.nanmin(ensemble))-1, np.ceil(np.nanmax(ensemble)) + 1, 1), weights=weights/W, label='Neighborhood distribution', alpha=0.75, color='silver')
    plot_distribution(ax, mean, std, color='k', vmin=np.min(ensemble), vmax=np.max(ensemble), label='Theoretical neighborhood distribution')
    ax.plot(initial_obs, 0.0003, marker='v', color='red', label='Original value', linestyle='')
    ax.plot(new_obs, 0.0003, marker='v', color='blue', label='Corrected value', linestyle='')
    ax.plot(np.mean(ensemble), 0.0003, marker='v', color='dimgrey', label='Non-weighted mean', linestyle='')
    ax.plot(mean, 0.0003, marker='v', color='k', label='Weighted mean', linestyle='')

    ax.legend()
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Probability')
    fig.savefig(os.path.join(savedir, 'Distribution.pdf'), format='pdf')

def plot_distribution(ax, mean, sd, vmin=0, vmax=1000, ensemble=None, label=None, color=None, distribution='norm', linewidth=1):
    x = np.linspace(vmin, vmax, 10000)
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']

    if distribution == 'norm':
        ax.plot(x, norm.pdf(x, loc=mean, scale=np.sqrt(sd)), color=color, label=label, linestyle='--', linewidth=linewidth)
        if ensemble is not None:
            #ax.bar(ensemble, norm.pdf(ensemble, loc=mean, scale=sd), 'r-', color=color, label='members', linewidth=linewidth)
            ax.bar(ensemble, norm.pdf(ensemble, loc=mean, scale=np.sqrt(sd)), width=0.1, color=color)
    elif distribution == 'gamma':
        #k = mean**2/sd
        #theta = sd/mean
        #on veut que mu soit le mode de la distribution gamma (< à la moyenne)
        theta = (np.sqrt(mean**2+4*sd)-mean)/2
        k     = 4*sd/(np.sqrt(mean**2+4*sd)-mean)**2
        ax.plot(x, gamma.pdf(x, k, scale=theta), color=color, label=label, linestyle='--', linewidth=0.5)
    elif distribution == 'EGP':
        pass

    return ax

#    obsweight = weights[point]
#    #ax.bar(ensemble, weights/np.sum(weights))
#    #ax.plot(obs, obsweight, marker='+', color='orange', label='Initial Observation')
#    ax.bar(mu, 2, width=0.005, color='k')
#
#    # Plot the actual distribution used for the assimilation :
#    plot_distribution(ax, mu, std, distribution='norm', linewidth=1, label=f'{product} distribution', color='k')  # mu est la valeur du pixel
##        plot_distribution(ax, obs, std, distribution='norm', linewidth=1, label='Observation distribution')  # mu est la valeur du pixel
#
#    # To test a new method (the goal is that it gives the same distribution as the red one in the final version) :
##        mean = np.sum(weights*ensemble)/np.sum(weights)
##        ax.bar(mean, 2.54, width=0.005, color='k')
##        plot_distribution(ax, mean, std, distribution='norm', linewidth=1, color='k')  # mu est la valeur du pixel
#
#    ax.plot(obs, obsweight, marker='.', color='k', label=f'Initial {product}', linestyle='', markersize=15)
#
#    if reference is not None:
#        ax.bar(np.sqrt(reference), 2, width=0.1, color='red', label='Reference Observation')
#
#    if initial_obs is not None:
#        ax.bar(np.sqrt(initial_obs[point]), 2, width=0.1, color='blue', label='Initial Observation')
#
##        newobs = (obs*obsweight + mean * np.nanmean(weights[weights>0])) / (obsweight+np.nanmean(weights[weights>0]))
##        #sd   = np.sum(weights*(ensemble-obs)**2)/np.sum(weights)
##        sd   = np.sum(weights*(ensemble-mean)**2)/np.sum(weights)
##        plot_distribution(ax, newobs, sd, distribution='norm', linewidth=1, color='blue', label='Observation distribution')
#
#    if now:
#        # Set figure boundaries
#        ax.set_ylim(bottom=0, top=1)
#        #ax.set_xlim(left=4, right=np.nanmax(ensemble)+std)
#        #ax.set_xlim(left=0, right=3)
#        ax.set_xlabel('R^1/2 (mm^1/2)')
#        ax.set_ylabel('Weight')
#        ax.legend()
##            if not os.path.exists(f'{self.date_str}/distributions'):
##                os.makedirs(f'{self.date_str}/distributions')


if __name__ == "__main__":

    # Extract and plot initial field
    filename = f'CUMUL_ANTILOPE_{domain}.nc'
    if os.path.exists(os.path.join(savedir, filename)):
        field = xr.open_dataarray(os.path.join(savedir, filename))
    else:
        tmp = '/home/vernaym/These/DATA/CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc'
        field = xr.open_dataarray(tmp)
        field = field.sel({'lat':np.intersect1d(extract_lat, field.lat), 'lon':np.intersect1d(extract_lon, field.lon)})
        field.to_netcdf(os.path.join(savedir, filename))
    vmax = np.nanmax(field.data)
    field = field.rename('Precipitation (mm)')
    plot(field, 'initial_field', vmax=vmax)

    # Extract correlation window
    coords=[(lon,lat) for lat in field.lat.data for lon in field.lon.data]
    codist = Preprocessing_ANTILOPE.codistances(coords)
    lons, lats = np.meshgrid(field.lon, field.lat)
    point = np.where((lons.flatten()==lon) & (lats.flatten()==lat))[0][0]
    correlations = codist.getrow(point).toarray()[0].reshape((len(field.lat), len(field.lon)))
    local = correlations.copy()
    local[local>0] = 1
    correlations = make_mask.to_xarray(correlations, field)
    correlations = correlations.rename('Inverse Distance Weighting')
    plot(correlations, 'correlations', cmap=plt.cm.Greys)
    window = field.copy()
    window.data = window.data * local
    plot(window, 'window', vmax=vmax)

    # Extract observation uncertainty / confidence
    fic_error = os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', f'Observation_uncertainty_0.15_alp.nc')
    error = xr.open_dataarray(fic_error)
    error = error.sel({'lat':np.intersect1d(extract_lat, error.lat), 'lon':np.intersect1d(extract_lon, error.lon)})
    plot(error, 'error_field', cmap=plt.cm.YlOrBr, vmin=1, vmax=30)
    local_error = error.copy()
    local_error.data = local * local_error.data
    local_error = local_error.rename('Estimated uncertainty')
    plot(local_error, 'local_error', cmap=plt.cm.YlOrBr)

    confidence = 1 / error
    local_confidence = confidence.copy()
    local_confidence.data = local_confidence.data * local
    local_confidence = local_confidence.rename('Estimated confidence')
    plot(local_confidence, 'confidence', cmap=plt.cm.Greens)

    # Compute weights
    weights = correlations * confidence
    weights = weights.rename('Weight')
    plot(weights, 'weights', cmap=plt.cm.Blues)

    # Field correction
    pond = codist.dot(diags(1/error.data.flatten(), 0))
    #tmp = pond.getrow(point).toarray()[0].reshape((len(field.lat), len(field.lon)))
    #tmp = make_mask.to_xarray(tmp.reshape((len(field.lat), len(field.lon))), field)
    #plot(tmp,'tmp')
    new, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(field.data, pond)
    correctedfield = make_mask.to_xarray(new.reshape((len(field.lat), len(field.lon))), field).rename('Precipitation (mm)')
    plot(correctedfield, 'dynamic_correction_only', vmax=vmax)

    original = field.sel({'lat':lat, 'lon':lon}).data
    corrected = correctedfield.sel({'lat':lat, 'lon':lon}).data
    # The distribution from the raw field is more interesting for method understanding
    plot_super_ensemble(window.data, weights.data, original, corrected)

    # Extract observation uncertainty / confidence
    fic_ratio = os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', f'Estimated_ratio_alp_0.15.nc')
    ratio = xr.open_dataarray(fic_ratio)
    ratio = ratio.sel({'lat':np.intersect1d(extract_lat, ratio.lat), 'lon':np.intersect1d(extract_lon, ratio.lon)})
    plot(ratio, 'ratio_field', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap)
    #local_ratio = ratio.copy()
    #local_error.data = local * local_error.data
    #plot(local_error, 'local_error', cmap=plt.cm.YlOrBr)

    # Debiasing
    deb = field / ratio.data
    plot(deb, 'debiased_field', vmax=vmax)
    window = deb.copy()
    window.data = window.data * local
    plot(window, 'window_after_debiasing', vmax=vmax)

    #plot_super_ensemble(window.data, weights.data)

    # Debiasing + field correction
    new, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(deb.data, pond, plot=True)
    correctedfield = make_mask.to_xarray(new.reshape((len(field.lat), len(field.lon))), field).rename('Precipitation (mm)')
    plot(correctedfield, 'debiasing+dynamic_correction', vmax=vmax)

    # smoothing
    smooth = field.copy()
    smooth.data = uniform_filter(deb, size=15)
    plot(smooth, 'smoothed_field', vmax=vmax)

    # Difference
    diff = (correctedfield - smooth).rename('Precipitation difference (mm)')
    dmax = np.nanmax(diff)
    plot(diff, 'diff', cmap='RdBu_r', vmin=-dmax, vmax=dmax)

    # Add relief mean vertical gradient (Not implemented in experiments)
    fic_gradient = os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', f'model_gradient.nc')
    gradient = xr.open_dataarray(fic_gradient)
    gradient = gradient.sel({'lat':np.intersect1d(extract_lat, gradient.lat), 'lon':np.intersect1d(extract_lon, gradient.lon)})
    final_field = correctedfield * gradient.data
    plot(final_field, 'debiasing+dynamic_correction+model_gradient', vmax=vmax)








