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

from These.radar import Preprocessing_ANTILOPE

savedir = '/home/vernaym/These/figures/illustration'
datadir = f'/home/vernaym/workdir/ASSIMILATION/mask/MontBlanc'

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

def random_field():
    # TODO  draw random values
    field = np.random.randint(0, 20, size=(Np, Np))
    field = uniform_filter(field, size=5)
    #field = np.zeros((Np,Np))

    # Add gaussian structure centered in the top left corner of the domain
    #k1d = signal.gaussian(2*Np, std=15).reshape(2*Np, 1)
    k1d = signal.gaussian(2*Np, std=10).reshape(2*Np, 1)
    kernel = np.outer(k1d, k1d)
    #A = np.zeros((Np, Np))
    #A[0:(Np//2)+1, 0:(Np//2)+1] = kernel[Np:, Np:]
    A = kernel[Np:, Np:]

    field = field + np.flip(A, axis=0)*15
    #field = field + np.flip(A, axis=0)*20
    #perturb = np.random.randint(0, 20, size=(Np, Np))/10.-1  # perturbations between -1 and 1
    #perturb = uniform_filter(perturb, size=5)
    #field = field * (1+perturb)

    return np.flip(field, axis=0)

def isolated_storm():

    # Add gaussian structure centered over the Mont-Blanc
    k1d = signal.gaussian(Np, std=2).reshape(Np, 1)
    field = np.outer(k1d, k1d) * 30

    return np.flip(field, axis=0)

def perturbed_field(field, ratio):
    #perturb = np.random.randint(0, 100, size=(Np, Np))/10. - 5  # generation of perturbations between -5 and 5
    perturb = np.random.randint(0, 200, size=(Np, Np))/10. - 10  # generation of perturbations between -10 and 10
    #perturb[perturb<0] = -perturb[perturb<0]
    #perturb[perturb==0] = 1
    #perturb = np.random.randint(1, 100, size=(Np, Np))/10.
    perturb = uniform_filter(perturb, size=5)
    perturb = np.flip(perturb, axis=0)

    plot_field(perturb, 'perturb.pdf', label='Ratio', cmap='RdBu_r', vmin=-2, vmax=2, add_circle=False)

    new_ratio = ratio + np.abs(1-ratio)*perturb
    new_ratio[new_ratio<0] = -new_ratio[new_ratio<0]
    new_ratio[new_ratio==0] = 0.1
    new_ratio = uniform_filter(new_ratio, size=3)

    plot_field(new_ratio, 'new_ratio.pdf', label='ratio', cmap='RdBu_r', vmin=0.2, vmax=1.8, add_circle=False)

    perturbed_field =field*new_ratio
    perturbed_field[perturbed_field<3] = 0
    return perturbed_field


def simple_error_field():
    ratio =  np.ones((Np, Np))
    ratio[(Np-1)//2, (Np-1)//2] = 0.2
    plot_field(ratio, 'ratio_simple.pdf', label='Ratio', cmap='RdBu', vmin=0.2, vmax=1.8, add_circle=False)

    error = np.ones((Np, Np))
    error[(Np-1)//2, (Np-1)//2] = 10
    #plt.imshow(field)
    #plt.show()
    plot_field(error, 'error_simple.pdf', label='Error (mm)', cmap='YlOrBr', vmin=1, vmax=10, add_circle=False)

    return ratio, error

def gaussian_field():
    """Generation of a Gaussian Kernel centered on point (X,Y)"""

    k1d = signal.gaussian(Np, std=5).reshape(Np, 1)
    kernel = np.outer(k1d, k1d)

    A = np.zeros((Np, Np))
    A[Np//2-(Np//2):Np//2+(Np//2)+1, Np//2-(Np//2):Np//2+(Np//2)+1] = kernel

    A = A*10
    A[(Np-1)//2, (Np-1)//2] = 2

    return A

def plot_field(field, filename, label='Precipitation (mm)', cmap='YlGnBu', vmin=0, vmax=1, add_circle=True):
    fig,ax = plt.subplots()
    fd = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(fd)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=label, fontsize=18)

    if add_circle:
        #circle = plt.Circle((Np//2, Np//2), ld*100*3, color='k', fill=False, linewidth=2)  # max distance
        circle = plt.Circle((Np//2, Np//2), ld*100*2, color='k', fill=False, linewidth=2)  # max distance
        #circle = plt.Circle((Np//2, Np//2), ld*100, color='k', fill=False, linewidth=2)  # Correlation length
        ax.add_artist(circle)
    #ax.scatter(Np//2, Np//2)  # Add dot over the Mont-Blanc

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(savedir, filename), format='pdf')


if __name__ == "__main__":

#field = random_field()
    field = isolated_storm()
    plot_field(field, f'real_field.pdf', vmin=0, vmax=30, add_circle=False)

    ratio = np.flip(xr.open_dataarray(os.path.join(datadir, 'Estimated_ratio_MontBlanc.nc')).data, axis=0)
    plot_field(ratio, f'ratio.pdf', vmin=0.2, vmax=1.8, add_circle=False)
    error = np.flip(xr.open_dataarray(os.path.join(datadir, 'Observation_error_MontBlanc.nc')).data, axis=0)

#field = perturbed_field(field, ratio)
    plot_field(field, f'fake_antilope_field.pdf', vmin=0, vmax=30, add_circle=False)

# Compute spatial correlations
    coords = [(lon/100., lat/100.) for lat in range(Np) for lon in range(Np)]
    codist = Preprocessing_ANTILOPE.codistances(coords, ld=ld)
#cd     = codist.toarray()
    plot_field(codist.getrow(Np**2//2).toarray()[0].reshape((Np,Np)), "codist.pdf", label='Codistances', vmin=0, vmax=1, cmap='Greens', add_circle=True)  # Weights for central pixel correction

# Add error ponderation
# TODO : choisir la bonne formulation
#pond = codist.dot(diags(np.exp(-(error-1)).flatten(), 0))  # error is in [1, inf[
    pond = codist.dot(diags((1/error).flatten(), 0))  # error is in [1, inf[
    plot_field(pond.diagonal().reshape((Np,Np)), "pond.pdf", label='Confidence', vmin=0, vmax=1, cmap='Greens')  # exp(-(error-1))
    plot_field(pond.getrow(Np**2//2).toarray()[0].reshape((Np,Np)), "weights_MontBlanc.pdf", label='Weights for Mont-Blanc correction', vmin=0, vmax=1, cmap='Greens', add_circle=True)  # Weights for central pixel correction

# De-biasing only
    db = field/ratio

# Dynamic correction only
    dyn = Preprocessing_ANTILOPE.dynamic_correction(field, pond)
    dyn = dyn.reshape((Np, Np))

# De-biasing + Dynamic correction
    dd = Preprocessing_ANTILOPE.dynamic_correction(field/ratio, pond)
    dd = dd.reshape((Np, Np))

#Dynamic correction + de-biasing
    qq = dyn/ratio

#plt.imshow(new)
#plt.show()
    savedir = '/home/vernaym/These/figures/illustration'

    plot_field(dyn, f'dynamic_correction_ld{ld}.pdf', vmin=0, vmax=30)
    plot_field(db, f'debiasing_ld{ld}.pdf', vmin=0, vmax=30)
    plot_field(dd, f'debiasing+dynamic_correction_ld{ld}.pdf', vmin=0, vmax=30)
    plot_field(qq, f'dynamic_correction_ld{ld}+debiasing.pdf', vmin=0, vmax=30)



