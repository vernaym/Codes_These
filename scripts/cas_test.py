#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 31/03/2023

import os, sys
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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

if len(sys.argv) > 1:
    domain = sys.argv[1]
else:
    domain = 'MontBlanc'

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
    Nk = np.random.randint(0, 5)  # Number of additional kernels
    for i in range(Nk):
        X   = np.random.randint(0, nlon)  # Position of the center of the kernel
        Y   = np.random.randint(0, nlat)  # Position of the center of the kernel
        std = np.random.randint(0, std0)  # Size of the kernel
        K   = np.random.randint(0, 20)  # Intensity of the kernel
        field = field + K * gaussian_field(X, Y, std, nlon, nlat)  # Add small scale kernel to the field

    #field = uniform_filter(field, size=3)
    field[field<3] = 0

    return np.flip(field, axis=0)

def isolated_storm():

    # Add gaussian structure centered over the Mont-Blanc
    k1d = signal.gaussian(Np, std=2).reshape(Np, 1)
    field = np.outer(k1d, k1d) * 30

    return np.flip(field, axis=0)

def perturbed_field(field, ratio):
    perturb = np.random.randint(0, 100, size=np.shape(field))/10. - 5  # generation of perturbations between -5 and 5
    #perturb = np.random.randint(0, 200, size=np.shape(field))/10. - 10  # generation of perturbations between -10 and 10
    #perturb[perturb<0] = -perturb[perturb<0]
    #perturb[perturb==0] = 1
    #perturb = np.random.randint(1, 100, size=(Np, Np))/10.
    perturb = uniform_filter(perturb, size=5)
    #perturb = uniform_filter(perturb, size=10)
    #perturb = np.flip(perturb, axis=0)

    plot_field(perturb, 'perturb.pdf', label='Ratio', cmap='RdBu_r', vmin=-2, vmax=2, add_circle=False)

    noise = np.random.randint(0, 40, size=np.shape(field))/100. - 0.2  # noise between -0.2 and 0.2
    new_ratio = ratio + np.abs(1-ratio)*perturb + noise
    #new_ratio = ratio + np.abs(1-ratio)*perturb
    new_ratio[new_ratio<0] = -new_ratio[new_ratio<0]
    new_ratio[new_ratio==0] = 0.1
    new_ratio = uniform_filter(new_ratio, size=5)

    plot_field(new_ratio, 'new_ratio.pdf', label='ratio', cmap='RdBu_r', vmin=0.2, vmax=1.8, add_circle=False)

    perturbed_field =field*new_ratio
    perturbed_field[perturbed_field<3] = 0  # Fake "missed precipitation"
    return perturbed_field


def simple_error_field():
    ratio =  np.ones((Np, Np))
    ratio[(Np-1)//2, (Np-1)//2] = 0.2
    plot_field(ratio, 'ratio_simple.pdf', label='Ratio', cmap='RdBu_r', vmin=0.2, vmax=1.8, add_circle=False)

    error = np.ones((Np, Np))
    error[(Np-1)//2, (Np-1)//2] = 10
    #plt.imshow(field)
    #plt.show()
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

def plot_field(field, filename, label='Precipitation (mm)', cmap='YlGnBu', vmin=None, vmax=None, add_circle=True):

    #if not isinstance(field, xr.core.dataarray.DataArray):
    if isinstance(field, np.ndarray):
        field = make_mask.to_xarray(field, real_ratio)
    fig, ax = plt.subplots()
    make_mask.plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if not filename.endswith('pdf'):
        filename = f'{filename}.pdf'
    fig.savefig(os.path.join(savedir, filename), format='pdf')
#    if vmin is None:
#        vmin = np.nanmin(field)
#    if vmax is None:
#        vmax = np.nanmax(field)
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

def plot_scatter(reference, model, savename):
    ref = reference.flatten()
    mod = model.flatten()
    bias = np.round(np.mean(mod - ref),2)
    rmse = np.round(np.sqrt(np.mean((mod-ref)**2)), 2)
    fig,ax = plt.subplots()
    ax.scatter(ref, mod)  # scatterplot ref vs estimation

    x = ref.reshape((-1,1))
    y = mod
    reg = LinearRegression().fit(x, y)
    z = reg.predict(x)
    r2 = np.round(reg.score(x, y), 2)

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

    nlon = len(real_ratio.lon)
    nlat = len(real_ratio.lat)
    real_field = random_field(nlat, nlon)  # Randomly generated reference precipitation field
    #field = isolated_storm()

    perturbed_field = perturbed_field(real_field.data, real_ratio.data)  # Pertubation of the reference field to simulate an ANTILOPE field

    # Compute spatial correlations
    #coords = [(lon/100., lat/100.) for lat in range(nlon) for lon in range(nlat)]
    coords = [(lon,lat) for lat in real_ratio.lat.data for lon in real_ratio.lon.data]
    codist = Preprocessing_ANTILOPE.codistances(coords, ld=ld)
    #cd     = codist.toarray()
    #plot_field(codist.getrow(Np**2//2).toarray()[0].reshape((Np,Np)), "codist.pdf", label='Codistances', vmin=0, vmax=1, cmap='Greens', add_circle=True)  # Weights for central pixel correction

    # Read ratio/error fields to evaluate
#    ratio = np.flip(xr.open_dataarray(os.path.join(datadir, 'Estimated_ratio_MontBlanc.nc')).data, axis=0)
#    plot_field(ratio, f'ratio.pdf', cmap='RdBu_r', vmin=0.2, vmax=1.8, add_circle=False)
#    error = np.flip(xr.open_dataarray(os.path.join(datadir, 'Observation_error_MontBlanc.nc')).data, axis=0)
    estimated_ratio = np.flip(xr.open_dataarray(os.path.join(datadir, f'Estimated_ratio_{domain}_0.15.nc')), axis=0)  # Ratio estimated with automatic observations that we want to evaluate
    #plot_field(estimated_ratio, f'estimated_ratio.pdf', cmap='RdBu_r', vmin=0.2, vmax=1.8, add_circle=False)
    error = np.flip(xr.open_dataarray(os.path.join(datadir, f'Observation_error_0.15_{domain}.nc')).data, axis=0)

    # Add error ponderation
    # TODO : reporter la formulation retenue dans l'expérience avec données réelles
    pond = codist.dot(diags(np.exp(-(error-1)).flatten(), 0))  # error is in [1, inf[.
    #pond = codist.dot(diags(np.exp(-error).flatten(), 0))  # error is in [1, inf[
    #pond = codist.dot(diags((1/error).flatten(), 0))  # error is in [1, inf[
    #plot_field(pond.diagonal().reshape((Np,Np)), "pond.pdf", label='Confidence', vmin=0, vmax=1, cmap='Greens')  # exp(-(error-1))
    #plot_field(pond.getrow(Np**2//2).toarray()[0].reshape((Np,Np)), "weights_MontBlanc.pdf", label='Weights for Mont-Blanc correction', vmin=0, vmax=1, cmap='Greens', add_circle=True)  # Weights for central pixel correction

    # De-biasing only
    db = perturbed_field / estimated_ratio.data

    # Dynamic correction only
    dyn, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(perturbed_field, pond)
    dyn = dyn.reshape((nlat, nlon))
    #plot_field(np.sqrt(sd.reshape((nlat, nlon))), f'dynamic_error_without_debiasing.pdf', label='Error (mm)', cmap=plt.cm.Reds)

    # De-biasing + Dynamic correction
    dd, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(perturbed_field.data/estimated_ratio.data, pond)
    dd = dd.reshape((nlat, nlon))
    #plot_field(np.sqrt(sd.reshape((nlat, nlon))), f'dynamic_error_with_debiasing.pdf', label='Error (mm)', cmap=plt.cm.Reds)

    #Dynamic correction + de-biasing  ==> does not work at all !
    #qq = dyn/ratio

    plot_scatter(real_field, perturbed_field, f"Initial_perturbations_scatterplot.pdf")
    plot_scatter(real_field, dyn, f"dynamic_correction_ld{ld}_scatterplot.pdf")
    plot_scatter(real_field, db, f"debiasing_scatterplot.pdf")
    plot_scatter(real_field, dd, f"debiasing+dynamic_correction_ld{ld}_scatterplot.pdf")

    real_field = make_mask.to_xarray(real_field, real_ratio, varname='Precipitation')
    perturbed_field = make_mask.to_xarray(perturbed_field, real_ratio, varname='Precipitation')
    dyn = make_mask.to_xarray(dyn, real_ratio, varname='Precipitation')
    db = make_mask.to_xarray(db, real_ratio, varname='Precipitation')
    dd = make_mask.to_xarray(dd, real_ratio, varname='Precipitation')

    vmax = max(np.max(real_field), np.max(perturbed_field), np.max(dyn), np.max(db), np.max(dd))
    plot_field(real_field, f'real_field.pdf', vmin=0, vmax=vmax)
    plot_field(perturbed_field, f'fake_antilope_field.pdf', vmin=0, vmax=vmax)
    plot_field(dyn, f'dynamic_correction_ld{ld}.pdf', vmin=0, vmax=vmax)
    plot_field(db, f'debiasing.pdf', vmin=0, vmax=vmax)
    plot_field(dd, f'debiasing+dynamic_correction_ld{ld}.pdf', vmin=0, vmax=vmax)
    #plot_field(qq, f'dynamic_correction_ld{ld}+debiasing.pdf', vmin=0, vmax=30)


    #TODO :
    # - Extend domain
    # - Add more diversity in initial field (position, magnitude and spread of the gaussian kernel)
    # - Perturbations using ratio field estimated with nivometeo observations
    # - Simulation on Alps domain
    # - Plot bias/rmse fields (--> ODG ?)
    # - Add random sampling
    # - Compute RS spread skill




