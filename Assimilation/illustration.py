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

savedir = '/home/vernaym/These/figures/illustration'

Np = 16  # Domain size

# TODO : Set up an idealised 2D experiment with an isolated storm to see the effect of background correction 
# when the model is able to reproduce the phenomenon but at a different location
#https://stackoverflow.com/questions/66580517/creating-a-gaussian-2d-array-with-mean-1-at-specificed-location
def field_generator(X, Y):
    """Generation of a Gaussian Kernel centered on point (X,Y)"""
    N = 7   # kernel size
#    N = 1   # kernel size
    k1d = signal.gaussian(N, std=1).reshape(N, 1)
    kernel = np.outer(k1d, k1d)
#    plt.imshow(kernel)
#    plt.show()

    #A = np.zeros((16, 16))
    A = np.zeros((Np, Np))
    A[X, Y] = 1    # random
#    plt.imshow(A)
#    plt.show()

    row, col = np.where(A == 1)
    #A[max(row[0]-(N//2), 0):min(row[0]+(N//2)+1, 15), max(col[0]-(N//2),0):min(col[0]+(N//2)+1,15)] = kernel
    A[row[0]-(N//2):row[0]+(N//2)+1, col[0]-(N//2):col[0]+(N//2)+1] = kernel
#    plt.imshow(A)
#    plt.show()

    return A

def codistances(coords):
    """
    Solution pour le calcul des inter-distances trouvée sur : https://stackoverflow.com/questions/35296935/python-calculate-lots-of-distances-quickly
    """
    tree = cKDTree(coords)
    dist = tree.sparse_distance_matrix(tree, 30, p=2, output_type='coo_matrix')
    dist = csr_matrix(dist)
    #TODO : utiliser une gaussienne plutot qu'une exponentielle décroissante
    dist[dist.nonzero()] = -dist[dist.nonzero()]
    np.exp(dist.data, out=dist.data )
    return dist

def illustration_modification_ebauche():
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

def resample(weights, Ne):
    """ Ne is the number of members to draw for the new enesmble """
    step = 1/Ne
    # Sort particules on [0,1[ according to their weight
    cumulated_weights = np.cumsum(weights, axis=0)
    selected_particles = list()
    # Random draw between [0, 1/Ne[
    rdm = random.uniform(0, step)
    while rdm <= 1:
        # Select particle indicies in wich rdm falls
        selected_particles.append(int(np.apply_along_axis(lambda a: a.searchsorted(rdm), axis=0, arr=cumulated_weights)))  # add index value (int)
        # Go 1 step forward and start again
        rdm += step
    return selected_particles

def plot(mu, std, N, obs, obs_std, vmin, vmax, distribution='norm', pf=True, enkf=True):

    fig,(ax0,ax1) = plt.subplots(2,1, gridspec_kw={'height_ratios': [8, 1]})

    # 0. Draw ensemble
    #TODO : draw ensemble from different distributions to see the differences
    if distribution == 'norm':
        ensemble = np.random.normal(loc=mu, scale=std, size=N)
    elif distribution == 'gamma':
#        k = mu**2/std
#        theta = std/mu
        #on veut que mu soit le mode de la distribution gamma (< à la moyenne)
        theta = (np.sqrt(mu**2+4*std)-mu)/2
        k     = 4*std/(np.sqrt(mu**2+4*std)-mu)**2
        #ensemble = np.random.gamma(k, scale=theta, size=N)
        ensemble = gamma.rvs(k, scale=theta, size=N)

    ensemble[ensemble<0]=0
    ensemble = np.sort(ensemble)

    # 1. Plot background
    #ax0.plot(ensemble, norm.pdf(ensemble, loc=mu, scale=std), label=f'Background (mean={mu}mm, std={std}mm)', color='k')
    if distribution == 'norm':
        ax0.plot(ensemble, norm.pdf(ensemble, loc=mu, scale=std), color='k')
    elif distribution == 'gamma':
        ax0.plot(ensemble, gamma.pdf(ensemble, k, scale=theta), color='k', linestyle='--', label='Gamma', linewidth=0.5)
        ax0.plot(np.NaN, np.NaN, color='k', label='Norm', linewidth=0.5)  # To add a legend entry without plotting anything
    ax0.hist(ensemble,density=True,bins=100, color='k', alpha=0.4, label='Background')

    # Draw values to illustrate members displacements
    if obs-obs_std+1>np.min(ensemble):
        drawmask = [np.where(ensemble==ensemble[(ensemble>obs-obs_std)&(ensemble<obs-obs_std+1)][0])[0][0], np.where(ensemble==ensemble[(ensemble>obs+2*std)&(ensemble<obs+2*std+1)][0])[0][0]]
        #drawmask = [np.where(ensemble==ensemble[(ensemble>mu-(std+obs_std))&(ensemble<mu-(std+obs_std)+1)][0])[0][0], np.where(ensemble==ensemble[(ensemble>mu+std)&(ensemble<mu+std+1)][0])[0][0]]
    else:
        drawmask = [np.where(ensemble==np.min(ensemble))[0][0], np.where(ensemble==ensemble[(ensemble>mu+std)&(ensemble<mu+std+0.1)][0])[0][0]]

    drawbackground = ensemble[drawmask]
    #drawbackground = np.array([ensemble[(ensemble>10)&(ensemble<11)][0], ensemble[(ensemble>29)&(ensemble<30)][0]])
    #ax1.plot(drawbackground, [2]*len(drawbackground), linestyle='', marker='.', color='k', markersize=10)

    # 2. Draw observation
    #ax0,ax1 = plot_distribution(ax0, ax1, obs, obs_std, data=[obs], color='red', marker='.', vmin=vmin, vmax=vmax, label='Observation distribution')
    #ax0,ax1 = plot_distribution(ax0, ax1, obs, obs_std, data=[obs], color='red', marker='.', vmin=vmin, vmax=vmax, label='Observation distribution')
    #ax0 = plot_normal_distribution(ax0, obs, obs_std, f'Observation (Y={obs}mm, std={obs_std}mm)', color='red')
    ax0 = plot_distribution(ax0, obs, obs_std, f'Observation', color='red', distribution='norm', linewidth=1)
    if distribution == 'gamma':
        ax0 = plot_distribution(ax0, obs, obs_std, color='red', distribution='gamma', linewidth=1)
    ax1.plot(obs, [3], color='red', marker='|', markersize=100)
    ax1.plot([obs-obs_std, obs+obs_std], [1.5, 1.5], marker='|', markersize=10, color='red', linewidth=1, linestyle='--')

    # 3. EnKF analysis
    if enkf:
        enkf = ensemble + std/(std+obs_std)*(obs-ensemble)
        enkf_mean = np.mean(enkf)
        enkf_std = np.sqrt(1/N*np.sum((enkf-enkf_mean)**2))
        #color = next(ax0._get_lines.prop_cycler)['color']
        color = 'blue'
        ax0.hist(enkf,density=True,bins=100, alpha=0.4, label='EnKF', color=color)
        ax0 = plot_distribution(ax0, enkf_mean, enkf_std, color=color, distribution='norm')
        if distribution == 'gamma':
            ax0 = plot_distribution(ax0, enkf_mean, enkf_std, color=color, distribution='gamma')
        drawenkf = drawbackground + std/(std+obs_std)*(obs-drawbackground)
        #ax1.plot(drawenkf, [2]*len(drawenkf), linestyle='', marker='.', color='blue', markersize=10)
        ax1.scatter(drawenkf, [2]*len(drawenkf), s=50, facecolors=color, edgecolors=None)
        enkf = True

    # 4. PF analysis
    if pf:
        if distribution == 'norm':
            weights = norm.pdf(ensemble, loc=obs, scale=obs_std)
        elif distribution == 'gamma':
            theta = (np.sqrt(obs**2+4*obs_std)-obs)/2
            k     = 4*obs_std/(np.sqrt(obs**2+4*obs_std)-obs)**2
            weights = gamma.pdf(ensemble, k, scale=theta)
        weights = weights / np.sum(weights)  # Normalisation
        selection = resample(weights, N)
        pf = ensemble[selection]
        pf_mean = np.mean(pf)
        pf_std  = np.sqrt(1/N*np.sum((pf-pf_mean)**2))
#        color = next(ax0._get_lines.prop_cycler)['color']
#        color = next(ax0._get_lines.prop_cycler)['color']  # call next twice to avoid orange color
        color = 'green'
        ax0.hist(pf, density=True,bins=100, alpha=0.4, label='PF', color=color)
        ax0 = plot_distribution(ax0, pf_mean, pf_std, color=color, distribution='norm')
        if distribution == 'gamma':
            ax0 = plot_distribution(ax0, pf_mean, pf_std, color=color, distribution='gamma')
        #drawpf = pf[drawmask]  # To plot analysis for the full ensemble
        # To plot PF analysis for a 2 members enemble :
        if distribution == 'norm':
            weights=norm.pdf(drawbackground, loc=obs, scale=obs_std)
        elif distribution == 'gamma':
            #k = pf_mean**2/pf_std
            #theta = pf_std/pf_mean
            #on veut que mu soit le mode de la distribution gamma (< à la moyenne)
            theta = (np.sqrt(pf_mean**2+4*pf_std)-pf_mean)/2
            k     = 4*pf_std/(np.sqrt(pf_mean**2+4*pf_std)-pf_mean)**2
            weights=gamma.pdf(drawbackground, k, scale=theta)
        weights = weights / np.sum(weights)  # Normalisation
        selection = resample(weights, 2)
        drawpf = drawbackground[selection]
        ax1.scatter(drawpf,[1]*len(drawpf), s=50, facecolors=color, edgecolors=None)
        pf = True

    # Plot displacment arrows
    for i in range(len(drawbackground)):
        #ax1.arrow(drawbackground[i], 1, drawenkf[i]-drawbackground[i], 0, head_width=0.05, head_length=0.1, fc='k', ec='k', linewidth=0.5)
        if enkf:
            ax1.scatter(drawbackground, [2]*len(drawbackground), s=50, facecolors='none', edgecolors='k')
            ax1.arrow(drawbackground[i], 2, drawenkf[i]-drawbackground[i], 0, head_width=0.5, head_length=1, linewidth=0.5, linestyle=':', length_includes_head=True, color='k')
        if pf:
            ax1.scatter(drawbackground, [1]*len(drawbackground), s=50, facecolors='none', edgecolors='k')
            ax1.arrow(drawbackground[i], 1, drawpf[i]-drawbackground[i], 0, head_width=0.5, head_length=1, linewidth=0.5, linestyle=':', length_includes_head=True, color='k')

    ax0.get_xaxis().set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.set_xlim(left=vmin, right=vmax)
#    ax0.set_ylim(top=min(0.7, max(?))  TODO : définir l'échelle dynamiquement
    ax0.set_ylim(top=0.15)
    ax0.legend(fontsize=10)

    ax1.get_yaxis().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_xlabel('Precipitation (mm)', fontsize=10)
    ax1.set_xlim(left=vmin, right=vmax)
    ax1.set_ylim(bottom=0, top=3)

    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"illustration_assimilation_{distribution}_{mu}_{std}_{obs}_{obs_std}.pdf"), format='pdf')

#    violin(ensemble, pf, enkf, obs)

#illustration_modification_ebauche()

methods = {'EGP_distribution':EGP_distribution}  # To call function fro string (see EGP_distribution)

# Definition of the background distribution statistics:
N = 1000000  # Ensemble size
#N = 1000  # To run PF faster
# TODO : géréer le cas de rr=0

# To plot EGP
#P0 = 0
#k = 5
#sigma = 1
#ksi = 0.5
#draw = inverse_sample_function('EGP_distribution', N, x_min=0, x_max=60, n=1e5, P0=0, k=k, sigma=sigma, ksi=ksi)
##x = np.random.uniform(0, 100, 1000000)
##draw = EGP_distribution(x, P0=P0, k=k, sigma=sigma, ksi=ksi)
##plt.plot(np.sort(draw))
#plt.hist(draw, density=True, bins=100, label='EGP')
#plt.show()
#import pdb
#pdb.set_trace()


mu  = 20  # ensemble mean
#mu  = 10  # ensemble mean
std = 10  # ensemble dispersion / std
#std = 8  # ensemble dispersion / std
#std = 5  # ensemble dispersion / std
obs = 10
obs_std = 5
vmin = 0
vmax = 40
plot(mu, std, N, obs, obs_std, vmin, vmax, pf=False, enkf=True)
#plot(mu, std, N, obs, obs_std, vmin, vmax, distribution='gamma', pf=True, enkf=False)
#plot(mu, std, N, obs, obs_std, vmin, vmax, distribution='gamma')
