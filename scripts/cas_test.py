#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 31/03/2023

import os, sys
import numpy as np
import pandas as pd

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

from These.radar import Preprocessing_ANTILOPE

savedir = '/home/vernaym/These/figures/illustration'

Np = 7  # Domain size

def field_generator():
    """Generation of an idealised precipitation field"""
    # Generation of a fake ANTILOPE precipitation field
    field = diags([1, 3, 5, 7, 10, 10, 10, 7, 5, 3, 1], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,], shape=(Np, Np)).toarray()
    field[3,3] = 2  # Underestimation of the precipitation in the center of the domain (==ridge)

    ratio =  np.ones((Np, Np))
    ratio[(Np-1)//2, (Np-1)//2] = 0.2

    error = np.ones((Np, Np))
    error[(Np-1)//2, (Np-1)//2] = 10
    #plt.imshow(field)
    #plt.show()

    return field, ratio, error

def fake_field():
    obs = field_generator(12, 12)
    stdobs=diags(obs.flatten()/5+1)
    ebauche = field_generator(6, 6)
    fig,ax = plt.subplots()
    im = ax.imshow(obs, cmap='viridis')
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(savedir, 'obs.png'))
    fig,ax = plt.subplots()
    im = ax.imshow(ebauche, cmap='viridis')
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(savedir, 'ebauche.png'))

    #coords=[(lon,lat) for lat in range(16) for lon in range(16)]
    coords=[(lon,lat) for lat in range(Np) for lon in range(Np)]
    pond = codistances(coords)

    super_ensemble = pond.copy()
    super_ensemble[super_ensemble.nonzero()] = 1  # Position of pixels to inclue in the spread computation = L
    # Computation of the likelyhood of each pixel of the super-ensemble
    O = super_ensemble.dot(diags(obs.flatten()))
    M = super_ensemble.dot(diags(ebauche.flatten()))
    S = super_ensemble.dot(diags(1/stdobs.diagonal()))
    A = (O-M).multiply(S)
    A = (M.transpose()-O).multiply(S)
    #A = (M-O.transpose()).multiply(S.transpose())
    likelyhood = -A.multiply(A)
    np.exp(likelyhood.data, out=likelyhood.data)  # likelyhood = exp(-((O-M).S)**2)

#    pond = pond.multiply(likelyhood)  # distance and likelyhood ponderation
    pond = likelyhood  # likelyhood only ponderation

    #weight = likelyhood.sum(axis=1).getA1()
    weight = pond.sum(axis=1).getA1()

    #replacement_strategy = 'mean'
    replacement_strategy = 'max_weight'

    newfield = get_parameters(ebauche, pond, weight=weight, super_ensemble=super_ensemble, replacement_strategy=replacement_strategy)


    new_ensemble = newfield.reshape(Np, Np)
#    if replacement_strategy == 'max_weight':
#        # 1. take all new values (weighted average between the original value and the average of
#        # the local super-ensemble weighted by the average weight
#        # --> cela a tendance à lisser le champs en diminuant/augmenatant les valeurs extremes !!
#        new_ensemble = newfield.reshape(Np, Np)
#    elif replacement_strategy == 'only_zeros':
#        # 3. Change only values when the original value is 0 and the local weighted average is >0
#        mean[np.where(ebauche > 0)] = ebauche[np.where(ebauche > 0)]
#        new_ensemble = mean.reshape(Np, Np)
#    elif replacement_strategy == 'mean':
#        # 4. Change values by the mean local average weighted by the observation likelyhood
#        new_ensemble = mean.reshape(Np, Np)

    fig,ax = plt.subplots()
    im = plt.imshow(new_ensemble, cmap='viridis')
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(savedir, 'new_ebauche.png'))
    import pdb
    pdb.set_trace()

def get_parameters(field, pond, weight=None, super_ensemble=None, replacement_strategy='keep'):

    initial_field = field.flatten()
    X = diags(field.flatten(), 0)

    # 1. Calcul de la moyenne pondérée par la distance ET l'erreur statique
    if weight is None:
        weight = pond.sum(axis=1).getA1()  # The sum of the weights (axis=1 <==> sum over rows)

    mean = pond.dot(X).sum(axis=1).getA1()  # getA1 transforms the 1*N matrix object into a 1D np.array
    mean = mean / weight

    # Get maximum weight of each line of the pond matrix
    idx = pond.argmax(axis=0).A1  # get the index of the maximum value of each line

#    TODO : on veut mapper initial_field[i] avec initial_field[idx] où idx est l'indice du maximim de likelyhood de la ligne i

    newfield = initial_field[idx]

    return newfield

def get_std(data, mean, pond, weight=None, super_ensemble=None):

    if weight is None:
        weight = pond.sum(axis=1).getA1()  # The sum of the weights (axis=1 <==> sum over rows)
    if super_ensemble is None:
        super_ensemble = pond.copy()  # WARNING : make a copy or pond will change when super_ensemble changes
        super_ensemble[super_ensemble.nonzero()] = 1  # Position of pixels to inclue in the spread computation

    se_mean = diags(mean, 0).dot(super_ensemble)  # matrix with mean[i] at each non-zero element of line i of super_ensemble
    X = super_ensemble.dot(data)-se_mean  # # M.diag(obs)-diag(e).M
    sd = X.multiply(X).multiply(pond).sum(axis=1).getA1()
    sd = sd / weight
    sd = np.nan_to_num(sd)

    return sd


def violin(raw, pf, enkf, obs):
    """ References :
    https://stackoverflow.com/questions/64646449/how-to-create-asymmetric-violin-plot-in-python-using-matplotlib
    https://seaborn.pydata.org/generated/seaborn.violinplot.html
    """
    fig, ax = plt.subplots()
    data = pd.DataFrame({'raw':raw, 'pf':pf, 'enkf':enkf})
    data = data.melt()
    data['dummy'] = 0
    #sns.violinplot(data=data, split=True, y='value', hue='variable', x='dummy', inner="stick", palette=['sandybrown', 'skyblue', 'green'])
    #sns.violinplot([raw], positions=[0], show_boxplot=False, side='left', ax=ax, plot_opts={'violin_fc':'C0'})
    sns.violinplot(positions=[0], data=raw, split=True, ax=ax, side='left', scale="count", scale_hue=False, saturation=0.75, inner=None, color='orange')
    sns.violinplot(positions=[0], data=pf, split=True, ax=ax, side='right', scale="count", scale_hue=False, saturation=0.75, inner=None, color='blue')
    sns.violinplot(positions=[0], data=enkf, split=True, ax=ax, side='right', scale="count", scale_hue=False, saturation=0.75, inner=None, color='green')
    #sns.violinplot(positions=[0], data=enkf, split=True, ax=ax, scale="count", scale_hue=False, saturation=0.75, inner=None, side='right')

#    violinplot([raw], positions=[0], side='left', ax=ax, plot_opts={'violin_fc':'C0'})
#    violinplot([pf], positions=[0], show_boxplot=False, side='right', ax=ax, plot_opts={'violin_fc':'C1'})
#    violinplot([enkf], positions=[0], show_boxplot=False, side='right', ax=ax, plot_opts={'violin_fc':'C1'})
    plt.plot(obs, marker='_', markersize=30, markeredgewidth=3, color='red')
    fig.savefig(f'violin.pdf', formatout='pdf',  bbox_inches='tight')

    #sns.violinplot(data=data, y='24 hour precipitation (mm)', split=True, hue='Simulation')

def inverse_sample_function(dist, N, x_min=0, x_max=60, n=1e5, **kwargs):
    """
    Method to draw a sample of 'N' points from a user-defined distribution 'dist'
    from : https://stackoverflow.com/questions/21100716/fast-arbitrary-distribution-random-sampling-inverse-transform-sampling
    """
    x = np.linspace(x_min, x_max, int(n))
    cumulative = np.cumsum(methods[dist](x, **kwargs))
    cumulative -= cumulative.min()
    f = interp1d(cumulative/cumulative.max(), x)

    return f(np.random.random(N))

def EGP_distribution(x, P0=0, k=5, sigma=1, ksi=0.5, **kw):
    """ Definition of the extended Pareto distribution in Taillardat 2020 or Papastathopoulos 2013"""
    EGP = P0 + (1 - P0) * (1 - (1 + ksi * x / sigma)**(-1 / ksi))**k
    return EGP

def plot_distribution(ax, mean, sd, label=None, color='k', distribution='norm', linewidth=0.5):
    x = np.linspace(0, 60, 10000)
    if distribution == 'norm':
        ax.plot(x, norm.pdf(x, loc=mean, scale=sd), 'r-', color=color, label=label, linewidth=linewidth)
        if color == 'red':  # Plot observation
            ax.bar(mean, norm.pdf(mean, loc=mean, scale=sd), color=color, width=0.1)
    elif distribution == 'gamma':
        #k = mean**2/sd
        #theta = sd/mean
        #on veut que mu soit le mode de la distribution gamma (< à la moyenne)
        theta = (np.sqrt(mean**2+4*sd)-mean)/2
        k     = 4*sd/(np.sqrt(mean**2+4*sd)-mean)**2
        ax.plot(x, gamma.pdf(x, k, scale=theta), 'r-', color=color, label=label, linestyle='--', linewidth=0.5)
    elif distribution == 'EGP':
        pass

    return ax


field, ratio, error = field_generator()

# Compute spatial correlations
coords = [(lon/100., lat/100.) for lat in range(Np) for lon in range(Np)]
codist = Preprocessing_ANTILOPE.codistances(coords)

# Add error ponderation
pond = codist.dot(diags(np.exp(-error).flatten(), 0))

# Compute new field
new = Preprocessing_ANTILOPE.dynamic_correction(field, pond)
new = new.reshape((Np, Np))

import pdb
pdb.set_trace()

