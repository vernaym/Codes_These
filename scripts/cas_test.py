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

plot = False
if len(sys.argv) > 1:
    domain = sys.argv[1]
    if len(sys.argv)>2:
        plot = True
else:
    domain = 'MontBlanc'

d0 = 0.15
d0 = 0.25

figsize = dict(
        alp            = (25,30),
        GrandesRousses = (15,7),
        HauteSavoie    = (12,12),
        HautesAlpes    = (24,15),
        MontBlanc      = (17,10),
        Savoie         = (16,8),
        Isere          = (16,8),
)

savedir = f'/home/vernaym/These/figures/illustration/{domain}'
datadir = f'/home/vernaym/workdir/ASSIMILATION/mask/{domain}'

#Np = 15  # Domain size
Np = 31  # Domain size
ld = 0.07  # Correlation length
#ld = 0.05  # Correlation length

def diagonal_field():
    """Generation of an idealised precipitation field"""
    # Generation of a fake ANTILOPE precipitation field
    #field = diags([1, 3, 5, 7, 10, 10, 10, 7, 5, 3, 1], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,], shape=(Np, Np)).toarray()
    #field[3,3] = 2  # Underestimation of the precipitation in the center of the domain (==ridge)
    field = diags([10-abs(x) for x in range(-Np//2, Np//2+2, 1)], range(-Np//2, Np//2+2, 1), shape=(Np, Np)).toarray()
    field[Np//2, Np//2] = 2  # Underestimation of the precipitation in the center of the domain (==ridge)

    return field

def random_field(nlon, nlat):
    """
    Method to generate a random precipitation field over a given domain (nlon, nlat)
    The field is the sum of :
    - a large scale field defined by a Gaussian kernel with a radius up to the domain size and intensity
      draw from uniform distribution
    - several small scale gaussian fields with max intensities drawn from decreasing exponential laws
    """

    # 1. Draw large scale field
    X   = np.random.randint(0, nlon)  # Position of the center of the kernel
    Y   = np.random.randint(0, nlat)  # Position of the center of the kernel
    std0 = max(nlon, nlat)//2  # Size of the kernel
    K0   = np.random.randint(0, 20)  # Intensity of the kernel
    field = gaussian_field(X, Y, std0, nlon, nlat)  # Generation of a field containing the gaussian kernel
    field = K0*field  # Add intensity

    #Nk = np.random.randint(0, 100)  # Number of additional kernels
    #Nk = np.random.randint(0, 5)  # Number of additional kernels
    Nk = np.random.randint(0, 10)  # Number of additional kernels
    for i in range(Nk):
        X   = np.random.randint(0, nlon)  # Position of the center of the kernel
        Y   = np.random.randint(0, nlat)  # Position of the center of the kernel
        std = np.random.randint(0, std0)  # Size of the kernel
        K   = np.random.randint(0, 20)  # Intensity of the kernel
        field = field + K * gaussian_field(X, Y, std, nlon, nlat)  # Add small scale kernel to the field

    random_ratio = np.random.randint(0, 20, size=np.shape(field))/10.  # generation of random ratio field between 0 and 2
    random_ratio = uniform_filter(random_ratio, size=10)

    field = field * random_ratio

    #field = field - 10  # Try to increase the number of non precipitation situations
    field[field<0] = 0

    return np.flip(field, axis=0)

def isolated_storm():

    # Add gaussian structure centered over the Mont-Blanc
    k1d = signal.gaussian(Np, std=2).reshape(Np, 1)
    field = np.outer(k1d, k1d) * 30

    return np.flip(field, axis=0)

def perturbed_ratio(ratio):
    """
    Perturbation of the "real" ANTILOPE vs reality ratio to add noise and match the estimated vs observed ratio scatter plot slope
    This accounts for the ratio estimation method errors where there is no obvious ANTILOPE spatial pattern.
    --> to apply only once (climatological noise)
    """
    pile_ou_face = np.random.randint(0, 1)
    pile_ou_face = 1

    #fact = ratio.data ** 2
    fact = ratio.data

    if pile_ou_face == 1:
        #ratio.data[(ratio.data>0.8) & (ratio.data<1.2)] = 1
        fact[(fact>0.7) & (fact<1.3)] = 1  # Filter out areas that the method almost certainly identify as bad

    increase = np.random.normal(30, scale=20)  # Draw random percentage of increase in bad areas
    #increase = 30  #  increase error in bad areas of 50%
    k = 100 / increase
    fact = fact - (1-fact) / k  # Increase error in bad areas

    perturb = np.random.randint(0, 20, size=np.shape(ratio.data))/10. - 1  # generation of perturbations between -1 and 1
    perturb = uniform_filter(perturb, size=5)  # Do not perturb the structure of the ratio field too much
    #perturb = uniform_filter(perturb, size=5)  # Do not perturb the structure of the ratio field too much
    #perturb = uniform_filter(perturb, size=5)
    #fact = uniform_filter(ratio.data, size=5)
    #ratio.data = ratio.data + (1+np.exp(-np.abs(1-fact)**2/1))*perturb
    tmp = fact + perturb
    tmp[tmp<=0] = -tmp[tmp<=0]+0.01
    #tmp = np.sqrt(tmp)
    ratio.data =  tmp
    plot_field(ratio, 'real_ratio.pdf', label='Ratio', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=0.4, vmax=1.6, add_circle=False)
    return ratio

def perturb_field(field, ratio):
    """
    Perturbation of the idealised climatological ANTILOPE vs reality ratio to account for daily variability.
    Here the perturbations are amplified for pixels with high climatolocical biases and a small random noise is added.
    --> to apply at each new "event"
    """
    perturb = np.random.randint(0, 20, size=np.shape(field))/10. - 1  # generation of perturbations between -1 and 1
    #perturb = np.random.randint(0, 200, size=np.shape(field))/10. - 10  # generation of perturbations between -10 and 10
    #perturb[perturb<0] = -perturb[perturb<0]
    #perturb[perturb==0] = 1
    #perturb = np.random.randint(1, 100, size=(Np, Np))/10.
    perturb = uniform_filter(perturb, size=5)
    #perturb = uniform_filter(perturb, size=10)
    #perturb = np.flip(perturb, axis=0)

    if plot:
        plot_field(perturb, 'perturb.pdf', label='Ratio', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=-1, vmax=1, add_circle=False)

    noise = np.random.randint(0, 10, size=np.shape(field))/10. - 0.5  # noise between -0.5 and 0.5
    noise = uniform_filter(noise, size=10)
    #new_ratio = ratio + (1+np.abs(1-ratio))*perturb + noise
    new_ratio = ratio * (1+perturb) + noise
    #new_ratio = ratio + np.abs(1-ratio)*perturb + noise
    new_ratio = uniform_filter(new_ratio, size=2)  # Increasing the window increases the mean negative bias over the Hautes Alpes

    #bias = np.random.randint(0, 4)/10. - 0.2  # Uniform bias of +/-20% over the domain
    #new_ratio = new_ratio + bias

    new_ratio[new_ratio<=0] = -new_ratio[new_ratio<=0]+0.01
    #new_ratio[new_ratio==0] = 0.01

    if plot:
        plot_field(new_ratio, 'daily_ratio.pdf', label='ratio', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=0.4, vmax=1.6, add_circle=False)

    perturbed_field =field*new_ratio
    #noise = np.random.randint(0, 10, size=np.shape(field)) - 5  # Add noise between -5mm and 5 mm
    #noise = uniform_filter(noise, size=5)
    #perturbed_field = perturbed_field + noise
    perturbed_field[perturbed_field<0.5] = 0  # Fake "missed precipitation"
    return perturbed_field


def simple_error_field():
    ratio =  np.ones((Np, Np))
    ratio[(Np-1)//2, (Np-1)//2] = 0.2
    if plot:
        plot_field(ratio, 'ratio_simple.pdf', label='Ratio', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=0.4, vmax=1.6, add_circle=False)

    error = np.ones((Np, Np))
    error[(Np-1)//2, (Np-1)//2] = 10
    #plt.imshow(field)
    #plt.show()
    if plot:
        plot_field(error, 'error_simple.pdf', label='Error (mm)', cmap='YlOrBr', vmin=1, vmax=10, add_circle=False)

    return ratio, error

def gaussian_field(X, Y, std, nlon, nlat):
    """Generation of a Gaussian Kernel of size 'std' centered on point (X,Y) of a (nlon, nlat) domain"""

    N = max(nlat, nlon)*2  # Size of the kernel
    k1d = signal.gaussian(N, std=std).reshape(N, 1)  # 1D Gaussian of size N
    kernel = np.outer(k1d, k1d)  # 2D gaussian of size (N,N)

    A = np.zeros((nlon, nlat))  # Initialisation of output field
    A[X, Y] = 1    # Center of gaussian kernel
    #row, col = np.where(A == 1)
    xmin = N//2-X if X<N//2 else 0
    xmax = N//2+nlon-X if X+N//2>nlon else N
    dx = xmax - xmin
    ymin = N//2-Y if Y<N//2 else 0
    ymax = N//2+nlat-Y if Y+N//2>nlat else N
    dy = ymax -ymin
    lonmin = max(X-N//2-xmin, 0)
    lonmax = min(lonmin+dx, nlon)
    latmin = max(Y-N//2-ymin, 0)
    latmax = min(latmin+dy, nlat)
    A[lonmin:lonmax, latmin:latmax] = kernel[xmin:xmax, ymin:ymax]

    #return np.flip(A, axis=0)
    return A

def plot_mean_and_dispersion(ensemble, vmin=None, vmax=None, cmap=plt.cm.YlGnBu):
    """
    Dispersion = sqrt(sum((Xi-Xmean)(Xi-Xmean)')/(N-1))
    """
#    if vmin is None:
#        vmin = np.nanmin(ensemble)
#    if vmax is None:
#        vmax = np.nanmax(ensemble)

    mean = np.mean(np.array(ensemble), axis=0)
    ensemble_mean = make_mask.to_xarray(mean, real_ratio, varname='Precipitation')
    plot_field(mean, f'Ensemble_mean.pdf', label='Precipitation (mm)', cmap=cmap, vmin=0, vmax=vmax)

    N = len(ensemble)
    dispersion = np.sqrt(np.sum(np.array([(member-mean)**2 for member in ensemble]), axis=0)/(N-1))
    dispersion = make_mask.to_xarray(dispersion, real_ratio, varname='Dispersion')
    plot_field(dispersion, f'Ensemble_dispersion.pdf', label='Precipitation (mm)', cmap=cmap)

def plot_field(field, filename, label='Precipitation (mm)', cmap='YlGnBu', vmin=None, vmax=None, add_circle=True):

    if vmin is None:
        vmin = np.nanmin(field)
    if vmax is None:
        vmax = np.nanmax(field)

    #if isinstance(field, np.ndarray):
    if not isinstance(field, xr.core.dataarray.DataArray):
        field = make_mask.to_xarray(field, real_ratio)
    nlat = len(field.lat)
    nlon = len(field.lon)
    fig, ax = plt.subplots(figsize=figsize[domain])
    make_mask.plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if not filename.endswith('pdf'):
        filename = f'{filename}.pdf'
    fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)
    fig.savefig(os.path.join(savedir, filename), format='pdf')
    plt.close(fig)
#
#    fig,ax = plt.subplots()
#    fd = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax)
#    cbar = fig.colorbar(fd)
#    cbar.ax.tick_params(labelsize=16)
#    cbar.set_label(label=label, fontsize=18)
#
#    if add_circle:
#        #circle = plt.Circle((Np//2, Np//2), ld*100*3, color='k', fill=False, linewidth=2)  # max distance
#        circle = plt.Circle((Np//2, Np//2), ld*100*2, color='k', fill=False, linewidth=2)  # max distance
#        #circle = plt.Circle((Np//2, Np//2), ld*100, color='k', fill=False, linewidth=2)  # Correlation length
#        ax.add_artist(circle)
#    #ax.scatter(Np//2, Np//2)  # Add dot over the Mont-Blanc
#
#    ax.set_xticks([])
#    ax.set_yticks([])
#    fig.tight_layout()
#    fig.savefig(os.path.join(savedir, filename), format='pdf')

def plot_ensemble_field(field, ax, vmin, vmax, title=None, cmap=plt.cm.YlGnBu):

    im = field.plot(ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_aspect('equal')
    ax.axis('off')
    if title is not None:
        ax.set_title(title)

    return im

def finalize_fig(figure, imm, label, outname):
    figure.tight_layout()
    figure.subplots_adjust(right=0.85)
    cbar_ax = figure.add_axes([0.87, 0.05, 0.03, 0.9])
    cb = figure.colorbar(imm, cax=cbar_ax)
    cb.ax.tick_params(labelsize=20)
    cb.set_label(label, size=24)
    figure.savefig(os.path.join(savedir, outname), format='pdf')
    plt.close(figure)

def compare(reference, model):
    diff = model-reference
    ratio = model / reference
    ratio[np.isinf(ratio)] = np.nan
    ref = reference.flatten()
    mod = model.flatten()
    x = ref.reshape((-1,1))
    y = mod
    reg = LinearRegression().fit(x, y)
    z = reg.predict(x)
    r2 = np.round(reg.score(x, y), 3)
    slope = np.round(reg.coef_[0], 3)

    return r2, diff, ratio, slope

def plot_scatter(reference, model, savename):
    ref = reference.flatten()
    mod = model.flatten()
    bias = np.round(np.mean(mod - ref),3)
    rmse = np.round(np.sqrt(np.mean((mod-ref)**2)), 3)
    x = ref.reshape((-1,1))
    y = mod
    reg = LinearRegression().fit(x, y)
    z = reg.predict(x)
    r2 = np.round(reg.score(x, y), 3)

    fig,ax = plt.subplots()
    # TODO : plot points with ratio inversion in red
    ax.scatter(ref, mod, marker='+')  # scatterplot ref vs estimation

    ax.plot(x, z, color='blue', linewidth=1, label=f'Slope={reg.coef_[0]:.3f}, Intercept={reg.intercept_:.3f}\nR²={r2:.4}, bias={bias}, rmse={rmse}')  # plot linear regression line
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
    ax.set_ylabel('Estimated value')
    ax.set_xlabel('Real value')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, savename), format='pdf')
    plt.close(fig)

def ensemble_evaluation(observation, rawfield, smoothfield, correctedfield, dynamiccorrection, ensemble, smoothfield5=None, smoothfield10=None):

    obse = observation.stack(points=["date", "lat", "lon"]).data
    ens = ensemble.stack(points=["date", "lat", "lon"]).transpose().data
    rw = rawfield.stack(points=["date", "lat", "lon"]).data
    sm = smoothfield.stack(points=["date", "lat", "lon"]).data
    dy = dynamiccorrection.stack(points=["date", "lat", "lon"]).data
    cr = correctedfield.stack(points=["date", "lat", "lon"]).data

    labels = dict(ens='Analysis ensemble', rw='Raw ANTILOPE', sm="Smoothed debiased ANTILOPE field", dy="Dynamic correction method alone", cr="Dynamic correction method with quantile-quantile adjustment")

    if smoothfield5 is not None:
        sm5 = smoothfield5.stack(points=["date", "lat", "lon"]).data
    if smoothfield10 is not None:
        sm10 = smoothfield10.stack(points=["date", "lat", "lon"]).data

    # 1. Brier score over all dates and pixels for different thresholds
    #brier = dict(ens=list(), rw = list(), sm=list(), cr=list(), sm5=list(), sm10=list())
    brier = dict(rw = list(), sm=list(), dy=list(), cr=list(), ens=list())
    if smoothfield5 is not None:
        brier['sm5'] = list()
    if smoothfield10 is not None:
        brier['sm10'] = list()
    #thresholds = [x/10 for x in range(1,10)] + [x for x in range(1, 51)]
    thresholds = [x for x in range(1, 51)]  # Ignore small precipitation problems for now
    for threshold in thresholds:
        brier['rw'].append(scores.brier(rw, obse, threshold=threshold))
        brier['sm'].append(scores.brier(sm, obse, threshold=threshold))
        brier['dy'].append(scores.brier(dy, obse, threshold=threshold))
        brier['cr'].append(scores.brier(cr, obse, threshold=threshold))
        brier['ens'].append(scores.brier(ens, obse, threshold=threshold))
        if smoothfield5 is not None:
            brier['sm5'].append(scores.brier(sm5, obse, threshold=threshold))
        if smoothfield10 is not None:
            brier['sm10'].append(scores.brier(sm10, obse, threshold=threshold))

    fig, ax = plt.subplots()
    for key, value in brier.items():
        if len(value) > 0:
            ax.plot(thresholds, value, label=labels[key])
            #ax.semilogx(thresholds, value, label=labels[key])  # Ignore small precipitation problems for now
    ax.legend()
    plt.tight_layout()
    ax.set_ylim(bottom=0, top=0.25)
    ax.set_xlabel('Threshold (mm)')
    ax.set_ylabel('Brier score')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, "Brier.pdf"), format='pdf')
    plt.close(fig)

    fig, ax = plt.subplots()
    next(ax._get_lines.prop_cycler)['color']  # Drop first color corresponding to raw product not show on the BSS
    for key, value in brier.items():
        if key != 'rw':
            ax.plot(thresholds, 1 - np.array(brier[key])/np.array(brier['rw']), label=labels[key])
            #ax.semilogx(thresholds, 1 - np.array(brier[key])/np.array(brier['rw']), label=labels[key])  # Ignore small precipitation problems for now
#    ax.plot(range(1, 51), 1 - np.array(brier['sm'])/np.array(brier['rw']), label='smooth')
#    ax.plot(range(1, 51), 1 - np.array(brier['cr'])/np.array(brier['rw']), label='correction')
#    ax.plot(range(1, 51), 1 - np.array(brier['ens'])/np.array(brier['rw']), label='ensemble')
#    if smoothfield5 is not None:
#        ax.plot(range(1, 51), 1 - np.array(brier['sm5'])/np.array(brier['rw']), label='smooth')
#    if smoothfield10 is not None:
#        ax.plot(range(1, 51), 1 - np.array(brier['sm10'])/np.array(brier['rw']), label='smooth')
    ax.axhline(0, color='k')
    ax.legend()
    ax.set_ylim(bottom=-1, top=1)
    ax.set_xlabel('Threshold (mm)')
    ax.set_ylabel('Brier Skill Score')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, "Brier_Skill_Score.pdf"), format='pdf')
    plt.close(fig)

    # 2. CRPS
    crps1 = scores.CRPS(ens, obse)
    crps1 = crps1.reshape(Ndates, nlat, nlon)
    crps2 = scores.CRPS(rw, obse)
    crps2 = crps2.reshape(Ndates, nlat, nlon)
    #crps3 = scores.CRPS(sm, obse)
    crps3 = scores.CRPS(dy, obse)
    crps3 = crps3.reshape(Ndates, nlat, nlon)
    crps4 = scores.CRPS(cr, obse)
    crps4 = crps4.reshape(Ndates, nlat, nlon)
    vmax = max(np.max(np.mean(crps1, axis=0)), np.max(np.mean(crps3, axis=0)), np.max(np.mean(crps4, axis=0)))
    plot_field(np.mean(crps1, axis=0), f'CRPS_ensemble.pdf', label='CRPS (mm)', cmap=plt.cm.Reds, vmin=0, vmax=vmax)
    plot_field(np.mean(crps2, axis=0), f'CRPS_raw.pdf', label='CRPS (mm)', cmap=plt.cm.Reds, vmin=0, vmax=vmax)
    plot_field(np.mean(crps3, axis=0), f'CRPS_dyn.pdf', label='CRPS (mm)', cmap=plt.cm.Reds, vmin=0, vmax=vmax)
    plot_field(np.mean(crps4, axis=0), f'CRPS_correction.pdf', label='CRPS (mm)', cmap=plt.cm.Reds, vmin=0, vmax=vmax)

    # 3. Spread-skill relationship
    # Long et inutile
#    N, Ne = np.shape(ens)
#    mean = np.mean(ens, axis=1)
#    disp = np.sqrt(np.sum((ens.transpose()-mean)**2, axis=0)/Ne)
#    err  = np.abs(mean-obse)
#    plot_scatter(err, disp, "Spread-skill_relationship.pdf")

    # 4. rank histogram
    fig, ax = plt.subplots()
    scores.rank_histogram(ens, obse, ax)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, "Rank_histogram.pdf"), format='pdf')
    plt.close(fig)

    # 5. Obs out of ensemble frequency
    # TODO : faire varier le threshold
    freq_error_raw = list()
    freq_error_smooth = list()
    freq_error_dyn = list()
    freq_error_correction = list()
    for threshold in np.arange(0.1, 0.61, 0.1):
        freq_error_raw.append(scores.error_frequency(rw, obse, threshold=threshold))
        #print('Raw error >20% frequency : ', freq_error_raw)
        freq_error_smooth.append(scores.error_frequency(sm, obse, treshold=threshold))
        #print('Smooth error >20% frequency : ', freq_error_smooth)
        freq_error_dyn.append(scores.error_frequency(dy, obse, treshold=threshold))
        freq_error_correction.append(scores.error_frequency(cr, obse, treshold=threshold))
        #print('Correction error >20% frequency : ', freq_error_correction)
    freq_error_ensemble = scores.error_frequency(ens, obse)
    fig, ax = plt.subplots()
    ax.plot(np.arange(10, 61, 10), np.array(freq_error_raw), label=labels['rw'])
    ax.plot(np.arange(10, 61, 10), np.array(freq_error_smooth), label=labels['sm'])
    ax.plot(np.arange(10, 61, 10), np.array(freq_error_dyn), label=labels['dy'])
    ax.plot(np.arange(10, 61, 10), np.array(freq_error_correction), label=labels['cr'])
    #ax.axhline(freq_error_ensemble, color='k', label=labels['ens'])
    ax.axhline(freq_error_ensemble, color=next(ax._get_lines.prop_cycler)['color'], label=labels['ens'])
    ax.set_xlabel('Error threshold (%)')
    ax.set_ylabel('Frequency of error above threshold (%)\nFrequency of observation outside the ensemble (%)')
    ax.legend(fontsize=8)
    #print('Obs outside analysis ensemble frequency : ', freq_error_ensemble)
    #ax2 = ax.twinx()
    #ax2.axhline(freq_error_ensemble, color='k')
    #ax2.set_ylim(ax.get_ylim())
    #ax2.set_ylabel("Frequency of observation outside the ensemble")
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, "error_frequency.pdf"), format='pdf')
    plt.close(fig)

    # 6. ROC
    fig, ax = plt.subplots()
    scores.ROC(rw, obse, 'raw', ax)
    scores.ROC(sm, obse, 'smooth15', ax)
    if smoothfield5 is not None:
        scores.ROC(sm5, obse, 'smooth5', ax)
    if smoothfield10 is not None:
        scores.ROC(sm10, obse, 'smooth10', ax)
    scores.ROC(dy, obse, 'dynamic', ax)
    scores.ROC(cr, obse, 'correction', ax)
    scores.ROC(ens, obse, 'analysis', ax)
    ax.set_xlabel('False alarm rate')
    ax.set_ylabel('Sucess rate')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, "ROC.pdf"), format='pdf')
    plt.close(fig)

    # 7. Reliability diagram
    fig, ax = plt.subplots()
    scores.reliability_diagram(ens, obse, 'Analysis', ax, threshold=10)
    fig.savefig(os.path.join(savedir, "Reliability_diagram.pdf"), format='pdf')
    plt.close(fig)


if __name__ == "__main__":

    # TODO : loop over realities

    real_ratio = np.flip(xr.open_dataarray(os.path.join(datadir, 'nivometeo', f'Estimated_ratio_{domain}_{d0}.nc')), axis=0)  # Reference ratio estimated with nivometeo observations only
    real_ratio = perturbed_ratio(real_ratio)  # Climatological perturbations of the ratio field to account for the ratio estimation method's errros

    # Compute spatial correlations
    #coords = [(lon/100., lat/100.) for lat in range(nlon) for lon in range(nlat)]
    coords = [(lon,lat) for lat in real_ratio.lat.data for lon in real_ratio.lon.data]
    codist = Preprocessing_ANTILOPE.codistances(coords, ld=ld)
    #cd     = codist.toarray()
    #plot_field(codist.getrow(Np**2//2).toarray()[0].reshape((Np,Np)), "codist.pdf", label='Codistances', vmin=0, vmax=1, cmap='Greens', add_circle=True)  # Weights for central pixel correction

    # Read ratio/error fields to evaluate
    estimated_ratio = np.flip(xr.open_dataarray(os.path.join(datadir, f'Estimated_ratio_{domain}_{d0}.nc')), axis=0)  # Ratio estimated with automatic observations that we want to evaluate
    error = np.flip(xr.open_dataarray(os.path.join(datadir, f'Observation_error_{d0}_{domain}.nc')).data, axis=0)

    plot_scatter(real_ratio.data, estimated_ratio.data, f"ratios_scatterplot.pdf")
    r1 = real_ratio.data.flatten()
    r2 = estimated_ratio.data.flatten()
    mask = np.random.randint(0, len(r1), 60)
    plot_scatter(r1[mask], r2[mask], f"ratios_scatterplot_sample.pdf")

    nlon = len(real_ratio.lon)
    nlat = len(real_ratio.lat)

    #r2 = dict(raw=list(), debiasing=list(), smoothing5=list(), smoothing10=list(), smoothing15=list(), dyn=list(), full=list())
    r2 = dict(raw=list(), debiasing=list(), smoothing15=list(), dyn=list(), full=list())
    err = dict(raw=list(), debiasing=list(), smoothing5=list(), smoothing10=list(), smoothing15=list(), dyn=list(), full=list())
    rat = dict(raw=list(), debiasing=list(), smoothing5=list(), smoothing10=list(), smoothing15=list(), dyn=list(), full=list())
    slp = dict(raw=list(), debiasing=list(), smoothing5=list(), smoothing10=list(), smoothing15=list(), dyn=list(), full=list())
    estimated_error = list()
    real_error      = list()
    real_error_dyn  = list()
    smooth_error    = list()

    if plot: Ndates = 1
    else : Ndates = 100
    #else : Ndates=2
    obs = xr.DataArray(dims=["date", "lat", "lon"], coords={'lon':real_ratio.lon, 'lat':real_ratio.lat, 'date':range(Ndates)})
    raw = xr.DataArray(dims=["date", "lat", "lon"], coords={'lon':real_ratio.lon, 'lat':real_ratio.lat, 'date':range(Ndates)})
    smo5 = xr.DataArray(dims=["date", "lat", "lon"], coords={'lon':real_ratio.lon, 'lat':real_ratio.lat, 'date':range(Ndates)})
    smo10 = xr.DataArray(dims=["date", "lat", "lon"], coords={'lon':real_ratio.lon, 'lat':real_ratio.lat, 'date':range(Ndates)})
    smo15 = xr.DataArray(dims=["date", "lat", "lon"], coords={'lon':real_ratio.lon, 'lat':real_ratio.lat, 'date':range(Ndates)})
    cor = xr.DataArray(dims=["date", "lat", "lon"], coords={'lon':real_ratio.lon, 'lat':real_ratio.lat, 'date':range(Ndates)})
    cor2 = xr.DataArray(dims=["date", "lat", "lon"], coords={'lon':real_ratio.lon, 'lat':real_ratio.lat, 'date':range(Ndates)})
    analysis = xr.DataArray(dims=["date", "member", "lat", "lon"], coords={'lon':real_ratio.lon, 'lat':real_ratio.lat, 'date':range(Ndates), 'member':range(16)})
    for date in range(Ndates):
    #for date in range(2):

        real_field = random_field(nlat, nlon)  # Randomly generated reference precipitation field
        obs.data[date] = real_field
        #field = isolated_storm()

        perturbed_field = perturb_field(real_field.data, real_ratio.data)  # Pertubation of the reference field to simulate an ANTILOPE field
        raw.data[date] = perturbed_field

        # Add error ponderation
        # TODO : reporter la formulation retenue dans l'expérience avec données réelles
        #pond = codist.dot(diags(np.exp(-(error-1)).flatten(), 0))  # error is in [1, inf[.
        #pond = codist.dot(diags(np.exp(-error)).flatten(), 0))  # error is in [1, inf[.
        pond = codist.dot(diags(1/error.flatten(), 0))  # error is in [1, inf[.  # Best formulation with new observation error formula

        # De-biasing only
        db = perturbed_field / estimated_ratio.data
        #smooth5 = uniform_filter(db, size=5)
        #smooth10 = uniform_filter(db, size=10)
        smooth15 = uniform_filter(db, size=15)
        #smo5.data[date] = smooth5
        #smo10.data[date] = smooth10
        smo15.data[date] = smooth15
        smootherr = db - smooth15

        # De-biasing + Dynamic correction only
        dyn, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(db, pond)
        dyn = dyn.reshape((nlat, nlon))
        cor2.data[date] = dyn
        sd1 = sd.reshape((nlat, nlon))
        #plot_field(np.sqrt(sd.reshape((nlat, nlon))), f'dynamic_error_without_debiasing.pdf', label='Error (mm)', cmap=plt.cm.Reds)

        # De-biasing + Dynamic correction + qq adjustment
        dd, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(db, pond, qq_adjustment=True)
        dd = dd.reshape((nlat, nlon))
        cor.data[date] = dd
        sd2 = sd.reshape((nlat, nlon))
        sd = sd2  # TODO : confirmer ce choix
        #sd = np.sqrt(sd*dd)  # Overdispersion !
        #plot_field(np.sqrt(sd), f'dynamic_error_with_debiasing.pdf', label='Error (mm)', cmap=plt.cm.Reds)


        vmax = max(np.max(real_field), np.max(perturbed_field), np.max(dyn), np.max(dd))*1.1

        #######################################################################
        # TMP : Try to re-increase extreme values
        # --> should be done for each neighborhood : unfeasible !!
#        ref = db.flatten()
#        mod = dd.flatten()
#        x = ref.reshape((-1,1))
#        y = mod
#        reg = LinearRegression().fit(x, y)
#        z = reg.predict(x)
#        r2 = np.round(reg.score(x, y), 3)
#        slope = reg.coef_[0]
#        intercep = reg.intercept_
#        toto = intercep + slope * db
#        plot_field(toto.reshape((nlat, nlon)), f'test.pdf', vmin=0, vmax=vmax)
#        plot_scatter(db, dd, f"dd_vs_db_scatterplot.pdf")
#        plot_scatter(db, toto, f"toto_vs_db_scatterplot.pdf")
#        plot_scatter(real_field, toto, f"toto_vs_real_scatterplot.pdf")
        #######################################################################

        if plot:
            fig,ax = plt.subplots(nrows=4, ncols=4, figsize=figsize[domain])
            i = 0
            j = 0
        ensemble = list()
        for member in range(16):
            #ana = Preprocessing_ANTILOPE.random_draw(dd, np.sqrt(sd))
            #ana = Preprocessing_ANTILOPE.random_draw(dd, sd+error.data)
            ana = Preprocessing_ANTILOPE.random_draw(dd, sd, distribution='gamma')  # Good ODG with real error
            #ana = Preprocessing_ANTILOPE.random_draw(dd, sd, distribution='normal')  # Good ODG with real error
            analysis.data[date, member] = ana
            ensemble.append(ana)
            ana = make_mask.to_xarray(ana, real_ratio, varname='Precipitation')
            if plot:
                im = plot_ensemble_field(ana, ax[i,j], 0, vmax)
                #im  = make_mask.plot_field(fig, ax[i,j], ana, cmap='YlGnBu', vmin=0, vmax=vmax)
                ax[i,j].set_title(None)
                j = j + 1
                if j==4:
                    j = 0
                    i = i + 1

        if plot:
            finalize_fig(fig, im, label='24-hour precipitation (mm)', outname=f'Analysis_ensemble.pdf')
            plot_mean_and_dispersion(ensemble, vmax=vmax)

        #Dynamic correction + de-biasing  ==> does not work at all !
        #qq = dyn/ratio

        if plot:

            plot_scatter(real_field, perturbed_field, f"Initial_perturbations_scatterplot.pdf")
            plot_scatter(real_field, dyn, f"dynamic_correction_ld{ld}_scatterplot.pdf")
            plot_scatter(real_field, db, f"debiasing_scatterplot.pdf")
            #plot_scatter(real_field, smooth5, f"smooth5_scatterplot.pdf")
            #plot_scatter(real_field, smooth10, f"smooth10_scatterplot.pdf")
            plot_scatter(real_field, smooth15, f"smooth15_scatterplot.pdf")
            plot_scatter(real_field, dd, f"debiasing+dynamic_correction_ld{ld}_scatterplot.pdf")
            plot_scatter(np.abs(dd-real_field), sd, f"error_scatterplot.pdf")

            # Transform np arrays into xarray Dataarrays
            real_field = make_mask.to_xarray(real_field, real_ratio, varname='Precipitation')
            perturbed_field = make_mask.to_xarray(perturbed_field, real_ratio, varname='Precipitation')
            dyn = make_mask.to_xarray(dyn, real_ratio, varname='Precipitation')
            db = make_mask.to_xarray(db, real_ratio, varname='Precipitation')
            dd = make_mask.to_xarray(dd, real_ratio, varname='Precipitation')

            # Plot fields
            #vmax = max(np.max(real_field), np.max(perturbed_field), np.max(dyn), np.max(db), np.max(dd))
            plot_field(real_field, f'real_field.pdf', vmin=0, vmax=vmax)
            plot_field(perturbed_field, f'fake_antilope_field.pdf', vmin=0, vmax=vmax)
            plot_field(dyn, f'dynamic_correction_ld{ld}.pdf', vmin=0, vmax=vmax)
            plot_field(db, f'debiasing.pdf', vmin=0, vmax=vmax)
            #plot_field(smooth5, f'Smoothed5_debiased_field.pdf', vmin=0, vmax=vmax)
            #plot_field(smooth10, f'Smoothed10_debiased_field.pdf', vmin=0, vmax=vmax)
            plot_field(smooth15, f'Smoothed15_debiased_field.pdf', vmin=0, vmax=vmax)
            plot_field(dd, f'debiasing+dynamic_correction_ld{ld}.pdf', vmin=0, vmax=vmax)
            #plot_field(qq, f'dynamic_correction_ld{ld}+debiasing.pdf', vmin=0, vmax=30)

            # Plot errors
            vmax = max(np.max(np.abs(dd.data-real_field.data)), np.max(sd1), np.max(sd2), np.max(smootherr))
            plot_field(smootherr, f'Estimated_error_smooth.pdf', label='Error (mm)', cmap=plt.cm.Reds, vmin=0, vmax=vmax)
            plot_field(sd1, f'Estimated_error_dyn.pdf', label='Error (mm)', cmap=plt.cm.Reds, vmin=0, vmax=vmax)
            plot_field(sd2, f'Estimated_error_full.pdf', label='Error (mm)', cmap=plt.cm.Reds, vmin=0, vmax=vmax)
            plot_field(dd-real_field, f'diff_full_correction-real_field.pdf', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=-vmax, vmax=vmax)
            plot_field(np.abs(dyn-real_field), f'real_error_dynamic_correction.pdf', cmap='Reds', vmin=0, vmax=vmax)
            plot_field(np.abs(dd-real_field), f'real_error_full_correction.pdf', cmap='Reds', vmin=0, vmax=vmax)
            vmax = np.nanmax(np.abs(dd.data/real_field.data)-1)
            #plot_field(dd/real_field, f'ratio_full_correction-real_field.pdf', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=1-vmax, vmax=1+vmax)

            simu  = analysis.data[0].reshape(16, nlat*nlon).transpose()
            obse = obs.data[0].flatten()

            brier = scores.brier(simu, obse, threshold=10)

        else:

            estimated_error.append(sd)
            real_error.append(np.abs(dd-real_field))
            real_error_dyn.append(np.abs(dyn-real_field))
            smooth_error.append(smootherr)
            a,b,c,d = compare(real_field, perturbed_field)
            r2['raw'].append(a)
            err['raw'].append(b)
            rat['raw'].append(c)
            slp['raw'].append(d)
            a,b,c,d = compare(real_field, db)
            r2['debiasing'].append(a)
            err['debiasing'].append(b)
            rat['debiasing'].append(c)
            slp['debiasing'].append(d)
#            a,b,c,d = compare(real_field, smooth5)
#            r2['smoothing5'].append(a)
#            err['smoothing5'].append(b)
#            rat['smoothing5'].append(c)
#            slp['smoothing5'].append(d)
#            a,b,c,d = compare(real_field, smooth10)
#            r2['smoothing10'].append(a)
#            err['smoothing10'].append(b)
#            rat['smoothing10'].append(c)
#            slp['smoothing10'].append(d)
            a,b,c,d = compare(real_field, smooth15)
            r2['smoothing15'].append(a)
            err['smoothing15'].append(b)
            rat['smoothing15'].append(c)
            slp['smoothing15'].append(d)
            a,b,c,d = compare(real_field, dyn)
            r2['dyn'].append(a)
            err['dyn'].append(b)
            rat['dyn'].append(c)
            slp['dyn'].append(d)
            a,b,c,d = compare(real_field, dd)
            r2['full'].append(a)
            err['full'].append(b)
            rat['full'].append(c)
            slp['full'].append(d)

    if not plot:
        #ensemble_evaluation(obs, raw, smo15, cor, analysis, smo5, smo10)
        ensemble_evaluation(obs, raw, smo15, cor, cor2, analysis)

        #bias = dict(raw=list(), debiasing=list(), smoothing5=list(), smoothing10=list(), smoothing15=list(), dyn=list(), full=list())
        #rmse = dict(raw=list(), debiasing=list(), smoothing=list(), smoothing10=list(), smoothing15=list(), dyn=list(), full=list())
        #ratio  = dict(raw=list(), debiasing=list(), smoothing=list(), smoothing10=list(), smoothing15=list(), dyn=list(), full=list())
        bias = dict(raw=list(), debiasing=list(), smoothing15=list(), dyn=list(), full=list())
        rmse = dict(raw=list(), debiasing=list(), smoothing15=list(), dyn=list(), full=list())
        ratio  = dict(raw=list(), debiasing=list(), smoothing15=list(), dyn=list(), full=list())
        for product in r2.keys():
            bias[product] = np.nanmean(np.array(err[product]), axis=0)
            rmse[product] = np.sqrt(np.nanmean(np.array(err[product])**2, axis=0))
            ratio[product]  = np.nanmean(np.array(rat[product]), axis=0)

        #bmin = min([np.min(arr) for arr in bias.values()])
        bb = bias.copy()
        bb.pop('debiasing')  # The debiasing method genreates unrealistically high biases
        bb.pop('raw')
        bmax = max([np.max(np.abs(arr)) for arr in bb.values()])
        emin = min([np.min(arr) for arr in rmse.values()])
        emax = max([np.max(arr) for arr in rmse.values()])
        #rmin = min([np.max(arr) for arr in rat.values()])
        rmax = max([np.max(np.abs(arr)) for arr in ratio.values()])
        for product in r2.keys():
            plot_field(bias[product], f'mean_bias_{product}.pdf', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=-bmax, vmax=bmax)
            plot_field(rmse[product], f'rmse_{product}.pdf', cmap='Reds', vmin=0, vmax=emax)
            plot_field(ratio[product], f'mean_ratio_{product}.pdf', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=0.4, vmax=1.6)
            if product != 'raw':
                plot_scatter(bias[product], bias['raw'], f"bias_{product}-raw_scatterplot.pdf")
                plot_scatter(rmse[product], rmse['raw'], f"rmse_{product}-raw_scatterplot.pdf")
                plot_scatter(ratio[product], ratio['raw'], f"ratio_{product}-raw_scatterplot.pdf")
                #plot_field(bias[product]-bias['raw'], f'diff_bias_{product}-raw.pdf', cmap='RdBu_r')
                #plot_field(rmse[product]-rmse['raw'], f'diff_rmse_{product}-raw.pdf', cmap='RdBu_r')
                #plot_field(ratio[product]-ratio['raw'], f'diff_ratio_{product}-raw.pdf', cmap='RdBu_r')

        for product in r2.keys():
            print(f'Mean R2 for product {product} = ', np.nanmean(np.array(r2[product])))
            print(f'Mean slope for product {product} = ', np.nanmean(np.array(slp[product])))
            if not product == 'raw':
                # TODO : improve representation
                plot_scatter(np.array(r2[product]), np.array(r2['raw']), f"R2_{product}_vs_raw_scatterplot.pdf")

        full_error_mean = np.mean(np.array(estimated_error), axis=0)
        real_error_mean = np.mean(np.array(real_error), axis=0)
        smooth_error_mean = np.mean(np.array(smooth_error), axis=0)
        vmax = max(np.nanmax(full_error_mean), np.nanmax(real_error_mean), np.nanmax(smooth_error_mean))
        plot_field(full_error_mean, f'mean_error_full.pdf', cmap=plt.cm.Reds, vmin=0, vmax=vmax)
        plot_field(real_error_mean, f'mean_error_real.pdf', cmap=plt.cm.Reds, vmin=0, vmax=vmax)
        plot_field(smooth_error_mean, f'mean_error_smooth.pdf', cmap=plt.cm.Reds, vmin=0, vmax=vmax)

        mean_full_error = np.mean(np.array(estimated_error), axis=(1,2))
        mean_estimated_error = np.mean(np.array(real_error), axis=(1,2))
        mean_smooth_error = np.mean(np.array(smooth_error), axis=(1,2))
        plot_scatter(mean_estimated_error, mean_full_error, f"estimated_error_full-real_scatterplot.pdf")
        plot_scatter(mean_estimated_error, mean_smooth_error, f"estimated_error_smooth-real_scatterplot.pdf")



    #TODO :
    # - Draw ~80 pixels to plot the ratio scatterplot on the same sample size and match correlations
    # - Improve plot_field formats (same figsizes as in "extract_domain")
    # - Plot estimated error (compare ODG with "real" error)
    # - Add random sampling
    # - Compute RS spread skill

    # DONE :
    # - Extend domain --> OK
    # - Add more diversity in initial field (position, magnitude and spread of the gaussian kernel) --> OK
    # - Perturbations using ratio field estimated with nivometeo observations --> OK
    # - Simulation on the Alps domain --> [OK]
    # - Statistics over ~100 situations  --> 0K
    # - Plot bias/rmse fields (--> ODG ?) --> Uniformiser les échelles entre les différents produits --> OK
    # - PLot "improvment fields" (ex : "product bias" vs "raw bias") --> OK




