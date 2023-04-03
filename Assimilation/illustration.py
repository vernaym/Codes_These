#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 31/03/2023

import os, sys
import numpy as np

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gamma
import random
from scipy.interpolate import interp1d


def violin(raw, assim, obs, xpid, num_poste, date, ref=None):
    """ References :
    https://stackoverflow.com/questions/64646449/how-to-create-asymmetric-violin-plot-in-python-using-matplotlib
    https://seaborn.pydata.org/generated/seaborn.violinplot.html
    """
    fig, ax = plt.subplots()
    data = pd.DataFrame({'raw':raw, 'assim':assim})
    data = data.melt()
    data['dummy'] = 0
    sns.violinplot(data=data, split=True, y='value', hue='variable', x='dummy', inner="stick", palette=['sandybrown', 'skyblue'])
    plt.plot(obs, marker='_', markersize=30, markeredgewidth=3, color='red')
    if ref is not None:
        plt.plot(ref, marker='_', markersize=30, markeredgewidth=3, color='dark')
    fig.savefig(f'{savedir}/assim_{xpid}_{num_poste}_{date}.pdf', formatout='pdf',  bbox_inches='tight')

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
        ax.plot(x, norm.pdf(x, loc=mean, scale=sd), 'r-', lw=1, color=color, label=label, linewidth=linewidth)
    elif distribution == 'gamma':
        #k = mean**2/sd
        #theta = sd/mean
        #on veut que mu soit le mode de la distribution gamma (< à la moyenne)
        theta = (np.sqrt(mean**2+4*sd)-mean)/2
        k     = 4*sd/(np.sqrt(mean**2+4*sd)-mean)**2
        ax.plot(x, gamma.pdf(x, k, scale=theta), 'r-', lw=1, color=color, label=label, linestyle='--', linewidth=0.5)
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

def plot(mu, std, N, obs, obs_std, vmin, vmax, distribution='norm'):

    fig,(ax0,ax1) = plt.subplots(2,1, gridspec_kw={'height_ratios': [8, 1]})
    # Draw ensemble
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

    #ax0.plot(ensemble, norm.pdf(ensemble, loc=mu, scale=std), label=f'Background (mean={mu}mm, std={std}mm)', color='k')
    if distribution == 'norm':
        ax0.plot(ensemble, norm.pdf(ensemble, loc=mu, scale=std), color='k')
    elif distribution == 'gamma':
        ax0.plot(ensemble, gamma.pdf(ensemble, k, scale=theta), color='k', linestyle='--', label='Gamma', linewidth=0.5)
        ax0.plot(np.NaN, np.NaN, color='k', label='Norm', linewidth=0.5)  # To add a legend entry without plotting anything
    ax0.hist(ensemble,density=True,bins=100, color='k', alpha=0.5, label='Background')
    if obs-obs_std+1>0:
        drawmask = [np.where(ensemble==ensemble[(ensemble>obs-obs_std)&(ensemble<obs-obs_std+1)][0])[0][0], np.where(ensemble==ensemble[(ensemble>obs+2*std)&(ensemble<obs+2*std+1)][0])[0][0]]
    else:
        drawmask = [np.where(ensemble==ensemble[(ensemble>=0)&(ensemble<=0.01)][0])[0][0], np.where(ensemble==ensemble[(ensemble>obs+std)&(ensemble<obs+std+0.1)][0])[0][0]]

    drawbackground = ensemble[drawmask]
    #drawbackground = np.array([ensemble[(ensemble>10)&(ensemble<11)][0], ensemble[(ensemble>29)&(ensemble<30)][0]])
    #ax1.plot(drawbackground, [2]*len(drawbackground), linestyle='', marker='.', color='k', markersize=10)

    # Draw observation
    #ax0,ax1 = plot_distribution(ax0, ax1, obs, obs_std, data=[obs], color='red', marker='.', vmin=vmin, vmax=vmax, label='Observation distribution')
    #ax0,ax1 = plot_distribution(ax0, ax1, obs, obs_std, data=[obs], color='red', marker='.', vmin=vmin, vmax=vmax, label='Observation distribution')
    #ax0 = plot_normal_distribution(ax0, obs, obs_std, f'Observation (Y={obs}mm, std={obs_std}mm)', color='red')
    ax0 = plot_distribution(ax0, obs, obs_std, f'Observation', color='red', distribution='norm', linewidth=1)
    if distribution == 'gamma':
        ax0 = plot_distribution(ax0, obs, obs_std, color='red', distribution='gamma', linewidth=1)
    ax1.plot(obs, [3], color='red', marker='|', markersize=100)
    ax1.plot([obs-std, obs+std], [1.5, 1.5], marker='|', markersize=10, color='red', linewidth=1, linestyle='--')

    # EnKF analysis
    enkf = ensemble + std/(std+obs_std)*(obs-ensemble)
    enkf_mean = np.mean(enkf)
    enkf_std = np.sqrt(1/N*np.sum((enkf-enkf_mean)**2))
    color = next(ax0._get_lines.prop_cycler)['color']
    ax0.hist(enkf,density=True,bins=100, alpha=0.5, label='EnKF', color=color)
    ax0 = plot_distribution(ax0, enkf_mean, enkf_std, color=color, distribution='norm')
    if distribution == 'gamma':
        ax0 = plot_distribution(ax0, enkf_mean, enkf_std, color=color, distribution='gamma')
    drawenkf = drawbackground + std/(std+obs_std)*(obs-drawbackground)
    #ax1.plot(drawenkf, [2]*len(drawenkf), linestyle='', marker='.', color='blue', markersize=10)
    ax1.scatter(drawenkf, [2]*len(drawenkf), s=50, facecolors=color, edgecolors=None)

    # PF analysis
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
    color = next(ax0._get_lines.prop_cycler)['color']
    color = next(ax0._get_lines.prop_cycler)['color']  # call next twice to avoid orange color
    ax0.hist(pf, density=True,bins=100, alpha=0.5, label='PF', color=color)
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

    # Draw background now to be visible
    ax1.scatter(drawbackground, [2]*len(drawbackground), s=50, facecolors='none', edgecolors='k')
    ax1.scatter(drawbackground, [1]*len(drawbackground), s=50, facecolors='none', edgecolors='k')

    # Plot displacment arrows
    for i in range(len(drawbackground)):
        #ax1.arrow(drawbackground[i], 1, drawenkf[i]-drawbackground[i], 0, head_width=0.05, head_length=0.1, fc='k', ec='k', linewidth=0.5)
        ax1.arrow(drawbackground[i], 2, drawenkf[i]-drawbackground[i], 0, head_width=0.5, head_length=1, linewidth=0.5, linestyle=':', length_includes_head=True, color='k')
        ax1.arrow(drawbackground[i], 1, drawpf[i]-drawbackground[i], 0, head_width=0.5, head_length=1, linewidth=0.5, linestyle=':', length_includes_head=True, color='k')

    ax0.get_xaxis().set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.set_xlim(left=vmin, right=vmax)
    ax0.set_ylim(top=0.7)
    ax0.legend(fontsize=10)

    ax1.get_yaxis().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_xlabel('Precipitation (mm)', fontsize=10)
    ax1.set_xlim(left=vmin, right=vmax)
    ax1.set_ylim(bottom=0, top=3)

    plt.tight_layout()
    fig.savefig(f"illustration_assimilation_{distribution}_{mu}_{std}_{obs}_{obs_std}.pdf", format='pdf')


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


mu  = 0  # ensemble mean
#mu  = 10  # ensemble mean
std = 5  # ensemble dispersion / std
#std = 5  # ensemble dispersion / std
obs = 0
obs_std = 2
vmin = 0
vmax = 20
vmax = 60
vmax = 5
plot(mu, std, N, obs, obs_std, vmin, vmax)
plot(mu, std, N, obs, obs_std, vmin, vmax, distribution='gamma')
