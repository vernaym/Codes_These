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

plot = False
if len(sys.argv) > 1:
    domain = sys.argv[1]
    if len(sys.argv)>2:
        plot = True
else:
    domain = 'MontBlanc'

figsize = dict(
        alp            = (14,16),
        GrandesRousses = (15,7),
        HauteSavoie    = (12,12),
        HautesAlpes    = (16,10),
        MontBlanc      = (15,10),
        Savoie         = (16,8),
        Isere          = (16,8),
)

savedir = f'/home/vernaym/These/figures/illustration/{domain}'
datadir = f'/home/vernaym/workdir/ASSIMILATION/mask/{domain}'

#Np = 15  # Domain size
Np = 31  # Domain size
ld = 0.07  # Correlation length

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
    random_ratio = uniform_filter(random_ratio, size=5)

    field = field * random_ratio

    field = field -3
    field[field<0] = 0

    return np.flip(field, axis=0)

def isolated_storm():

    # Add gaussian structure centered over the Mont-Blanc
    k1d = signal.gaussian(Np, std=2).reshape(Np, 1)
    field = np.outer(k1d, k1d) * 30

    return np.flip(field, axis=0)

def perturbed_ratio(ratio):
    """
    Perturbation of the "real" ANTILOPE vs reality ratio to add noise around "non-biased" pixels.
    This accounts for the ratio estimation method errors where there is no obvious ANTILOPE spatial pattern.
    --> to apply only once (climatological noise)
    """
    perturb = np.random.randint(0, 20, size=np.shape(ratio.data))/10. - 1  # generation of perturbations between -1 and 1
    perturb = uniform_filter(perturb, size=10)  # Do not perturb the structure of the ratio field too much
    #perturb = uniform_filter(perturb, size=5)
    #fact = uniform_filter(ratio.data, size=5)
    fact = ratio.data
    ratio.data = ratio.data + (1+np.exp(-np.abs(1-fact)**2/1))*perturb
    #ratio.data = ratio.data + 1/(1+np.abs(fact-1))*perturb
    #ratio.data = ratio.data + 1/(1+np.abs(fact-1))*perturb
    ratio.data[ratio.data<=0] = -ratio.data[ratio.data<=0]+0.01
    #ratio.data = uniform_filter(ratio.data, size=5)  # Increasing the window increases the mean negative bias over the Hautes Alpes
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
    perturbed_field[perturbed_field<0.1] = 0  # Fake "missed precipitation"
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
    Dispersion = sqrt(sum((Xi-Xmean)(Xi-Xmean)'))
    """
#    if vmin is None:
#        vmin = np.nanmin(ensemble)
#    if vmax is None:
#        vmax = np.nanmax(ensemble)

    mean = np.mean(np.array(ensemble), axis=0)
    ensemble_mean = make_mask.to_xarray(mean, real_ratio, varname='Precipitation')
    plot_field(mean, f'Ensemble_mean.pdf', label='Precipitation (mm)', cmap=cmap, vmin=0, vmax=vmax)

    N = len(ensemble)
    dispersion = np.sqrt(np.sum(np.array([(member-mean)**1 for member in ensemble]), axis=0)/N)
    dispersion = make_mask.to_xarray(dispersion, real_ratio, varname='Dispersion')
    plot_field(dispersion, f'Ensemble_dispersion.pdf', label='Precipitation (mm)', cmap=cmap)

def plot_field(field, filename, label='Precipitation (mm)', cmap='YlGnBu', vmin=None, vmax=None, add_circle=True):

    if vmin is None:
        vmin = np.nanmin(field)
    if vmax is None:
        vmax = np.nanmax(field)

    #if not isinstance(field, xr.core.dataarray.DataArray):
    if isinstance(field, np.ndarray):
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
    r2 = np.round(reg.score(x, y), 2)

    return r2, diff, ratio


def plot_scatter(reference, model, savename):
    ref = reference.flatten()
    mod = model.flatten()
    bias = np.round(np.mean(mod - ref),2)
    rmse = np.round(np.sqrt(np.mean((mod-ref)**2)), 2)
    x = ref.reshape((-1,1))
    y = mod
    reg = LinearRegression().fit(x, y)
    z = reg.predict(x)
    r2 = np.round(reg.score(x, y), 2)

    fig,ax = plt.subplots()
    # TODO : plot points with ratio inversion in red
    ax.scatter(ref, mod, marker='+')  # scatterplot ref vs estimation

    ax.plot(x, z, color='blue', linewidth=1, label=f'R²={r2:.4}, bias={bias}, rmse={rmse}')  # plot linear regression line
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
    ax.legend(fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, savename), format='pdf')


if __name__ == "__main__":

    real_ratio = np.flip(xr.open_dataarray(os.path.join(datadir, 'nivometeo', f'Estimated_ratio_{domain}_0.15.nc')), axis=0)  # Reference ratio estimated with nivometeo observations only
    real_ratio = perturbed_ratio(real_ratio)  # Climatological perturbations of the ratio field to account for the ratio estimation method's errros

    # Compute spatial correlations
    #coords = [(lon/100., lat/100.) for lat in range(nlon) for lon in range(nlat)]
    coords = [(lon,lat) for lat in real_ratio.lat.data for lon in real_ratio.lon.data]
    codist = Preprocessing_ANTILOPE.codistances(coords, ld=ld)
    #cd     = codist.toarray()
    #plot_field(codist.getrow(Np**2//2).toarray()[0].reshape((Np,Np)), "codist.pdf", label='Codistances', vmin=0, vmax=1, cmap='Greens', add_circle=True)  # Weights for central pixel correction

    # Read ratio/error fields to evaluate
    estimated_ratio = np.flip(xr.open_dataarray(os.path.join(datadir, f'Estimated_ratio_{domain}_0.15.nc')), axis=0)  # Ratio estimated with automatic observations that we want to evaluate
    error = np.flip(xr.open_dataarray(os.path.join(datadir, f'Observation_error_0.15_{domain}.nc')).data, axis=0)

    plot_scatter(real_ratio.data, estimated_ratio.data, f"ratios_scatterplot.pdf")
    r1 = real_ratio.data.flatten()
    r2 = estimated_ratio.data.flatten()
    mask = np.random.randint(0, len(r1), 60)
    plot_scatter(r1[mask], r2[mask], f"ratios_scatterplot_sample.pdf")

    nlon = len(real_ratio.lon)
    nlat = len(real_ratio.lat)

    r2 = dict(raw=list(), debiasing=list(), dyn=list(), full=list())
    err = dict(raw=list(), debiasing=list(), dyn=list(), full=list())
    rat = dict(raw=list(), debiasing=list(), dyn=list(), full=list())

    if plot: N = 1
    else : N=100
    for i in range(N):
    #for i in range(2):

        real_field = random_field(nlat, nlon)  # Randomly generated reference precipitation field
        #field = isolated_storm()

        perturbed_field = perturb_field(real_field.data, real_ratio.data)  # Pertubation of the reference field to simulate an ANTILOPE field

        # Add error ponderation
        # TODO : reporter la formulation retenue dans l'expérience avec données réelles
        pond = codist.dot(diags(np.exp(-(error-1)).flatten(), 0))  # error is in [1, inf[.

        # De-biasing only
        db = perturbed_field / estimated_ratio.data

        # Dynamic correction only
        dyn, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(perturbed_field, pond)
        dyn = dyn.reshape((nlat, nlon))
        #plot_field(np.sqrt(sd.reshape((nlat, nlon))), f'dynamic_error_without_debiasing.pdf', label='Error (mm)', cmap=plt.cm.Reds)

        # De-biasing + Dynamic correction
        dd, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(perturbed_field.data/estimated_ratio.data, pond)
        dd = dd.reshape((nlat, nlon))
        sd = sd.reshape((nlat, nlon))
        sd = uniform_filter(sd, size=10)  # TODO : TMP !!!
        #plot_field(np.sqrt(sd), f'dynamic_error_with_debiasing.pdf', label='Error (mm)', cmap=plt.cm.Reds)
        plot_field(sd, f'dynamic_error_with_debiasing.pdf', label='Error (mm)', cmap=plt.cm.Reds)

        vmax = max(np.max(real_field), np.max(perturbed_field), np.max(dyn), np.max(dd))
        if plot:
            fig,ax = plt.subplots(nrows=4, ncols=4, figsize=figsize[domain])
            i = 0
            j = 0
        analysis = list()
        for member in range(16):
            #ana = Preprocessing_ANTILOPE.random_draw(dd, np.sqrt(sd))
            #ana = Preprocessing_ANTILOPE.random_draw(dd, sd)
            ana = Preprocessing_ANTILOPE.random_draw(dd, sd+error.data)
            analysis.append(ana)
            ana = make_mask.to_xarray(ana, real_ratio, varname='Precipitation')
            if plot:
                im = plot_ensemble_field(ana, ax[i,j], 0, vmax)
                #im  = make_mask.plot_field(fig, ax[i,j], ana, cmap='YlGnBu', vmin=0, vmax=vmax)
                ax[i,j].set_title(None)
                j = j + 1
                if j==4:
                    j = 0
                    i = i + 1
        finalize_fig(fig, im, label='24-hour precipitation (mm)', outname=f'Analysis_ensemble.pdf')

        if plot:
            plot_mean_and_dispersion(analysis, vmax=vmax)

        #Dynamic correction + de-biasing  ==> does not work at all !
        #qq = dyn/ratio

        if plot:

            plot_scatter(real_field, perturbed_field, f"Initial_perturbations_scatterplot.pdf")
            plot_scatter(real_field, dyn, f"dynamic_correction_ld{ld}_scatterplot.pdf")
            plot_scatter(real_field, db, f"debiasing_scatterplot.pdf")
            plot_scatter(real_field, dd, f"debiasing+dynamic_correction_ld{ld}_scatterplot.pdf")

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
            plot_field(dd, f'debiasing+dynamic_correction_ld{ld}.pdf', vmin=0, vmax=vmax)
            #plot_field(qq, f'dynamic_correction_ld{ld}+debiasing.pdf', vmin=0, vmax=30)

            # Plot errors
            vmax = np.max(np.abs(dd.data-real_field.data))
            plot_field(dd-real_field, f'diff_full_correction-real_field.pdf', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=-vmax, vmax=vmax)
            plot_field(np.abs(dd-real_field), f'real_error_full_correction.pdf', cmap='Reds', vmin=0, vmax=vmax)
            vmax = np.nanmax(np.abs(dd.data/real_field.data)-1)
            plot_field(dd/real_field, f'ratio_full_correction-real_field.pdf', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=1-vmax, vmax=1+vmax)

        else:

            a,b,c = compare(real_field, perturbed_field)
            r2['raw'].append(a)
            err['raw'].append(b)
            rat['raw'].append(c)
            a,b,c = compare(real_field, db)
            r2['debiasing'].append(a)
            err['debiasing'].append(b)
            rat['debiasing'].append(c)
            a,b,c = compare(real_field, dyn)
            r2['dyn'].append(a)
            err['dyn'].append(b)
            rat['dyn'].append(c)
            a,b,c = compare(real_field, dd)
            r2['full'].append(a)
            err['full'].append(b)
            rat['full'].append(c)

    if not plot:

        bias = dict(raw=list(), debiasing=list(), dyn=list(), full=list())
        rmse = dict(raw=list(), debiasing=list(), dyn=list(), full=list())
        ratio  = dict(raw=list(), debiasing=list(), dyn=list(), full=list())
        for product in r2.keys():
            bias[product] = np.nanmean(np.array(err[product]), axis=0)
            rmse[product] = np.sqrt(np.nanmean(np.array(err[product])**2, axis=0))
            ratio[product]  = np.nanmean(np.array(rat[product]), axis=0)

        #bmin = min([np.min(arr) for arr in bias.values()])
        bb = bias.copy()
        bb.pop('debiasing')
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
            print(f'Mean R2 for product {product} = ', np.mean(np.array(r2[product])))
            if not product == 'raw':
                # TODO : improve representation
                plot_scatter(np.array(r2[product]), np.array(r2['raw']), f"RS_{product}_vs_raw_scatterplot.pdf")


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




