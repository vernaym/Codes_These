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
from scipy.stats import norm
from scipy.special import gamma
import random


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

def plot_normal_distribution(ax, mu, sd, label=None, color='k'):
    x = np.linspace(0, 60, 10000)
    ax.plot(x, norm.pdf(x, loc=mu, scale=sd), 'r-', lw=1, color=color, label=label)

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

fig,(ax0,ax1) = plt.subplots(2,1, gridspec_kw={'height_ratios': [8, 1]})

# Definition of the background distribution statistics:
N = 1000000  # Ensemble size
#N = 1000  # To run PF faster
mu  = 30  # mean = 30mm
std = 10  # Standard deviation=10mm
vmin = 0
vmax = 60

# Draw ensemble
#TODO : draw ensemble from different distributions to see the differences
ensemble = np.random.normal(loc=mu, scale=std, size=N)
ensemble[ensemble<0]=0
ensemble = np.sort(ensemble)

#ax0.plot(ensemble, norm.pdf(ensemble, loc=mu, scale=std), label=f'Background (mean={mu}mm, std={std}mm)', color='k')
ax0.plot(ensemble, norm.pdf(ensemble, loc=mu, scale=std), color='k')
ax0.hist(ensemble,density=True,bins=100, color='k', alpha=0.5, label='Background')
drawmask = [np.where(ensemble==ensemble[(ensemble>10)&(ensemble<11)][0])[0][0], np.where(ensemble==ensemble[(ensemble>29)&(ensemble<30)][0])[0][0]]
drawbackground = ensemble[drawmask]
#drawbackground = np.array([ensemble[(ensemble>10)&(ensemble<11)][0], ensemble[(ensemble>29)&(ensemble<30)][0]])
#ax1.plot(drawbackground, [2]*len(drawbackground), linestyle='', marker='.', color='k', markersize=10)

# Draw observation
obs = 25
obs_std = 5
#ax0,ax1 = plot_distribution(ax0, ax1, obs, obs_std, data=[obs], color='red', marker='.', vmin=vmin, vmax=vmax, label='Observation distribution')
#ax0,ax1 = plot_distribution(ax0, ax1, obs, obs_std, data=[obs], color='red', marker='.', vmin=vmin, vmax=vmax, label='Observation distribution')
#ax0 = plot_normal_distribution(ax0, obs, obs_std, f'Observation (Y={obs}mm, std={obs_std}mm)', color='red')
ax0 = plot_normal_distribution(ax0, obs, obs_std, f'Observation', color='red')
ax1.plot(obs, [3], color='red', marker='|', markersize=100)

# EnKF analysis
enkf = ensemble + std/(std+obs_std)*(obs-ensemble)
enkf_mean = np.mean(enkf)
enkf_std = np.sqrt(1/N*np.sum((enkf-enkf_mean)**2))
color = next(ax0._get_lines.prop_cycler)['color']
ax0.hist(enkf,density=True,bins=100, alpha=0.5, label='EnKF', color=color)
ax0 = plot_normal_distribution(ax0, enkf_mean, enkf_std, color=color)
drawenkf = drawbackground + std/(std+obs_std)*(obs-drawbackground)
#ax1.plot(drawenkf, [2]*len(drawenkf), linestyle='', marker='.', color='blue', markersize=10)
ax1.scatter(drawenkf, [2]*len(drawenkf), s=50, facecolors=color, edgecolors=None)

# PF analysis
#if likelyhood == 'normal':
#    draw = self.normal_dist(x, mu, sigma)
#elif likelyhood == 'gamma':
#    draw = self.gamma_dist(x, mu, sigma)
weights = norm.pdf(ensemble, loc=obs, scale=obs_std)
weights = weights / np.sum(weights)  # Normalisation
selection = resample(weights, N)
pf = ensemble[selection]
color = next(ax0._get_lines.prop_cycler)['color']
color = next(ax0._get_lines.prop_cycler)['color']  # call next twice to avoid orange color
ax0.hist(pf, density=True,bins=100, alpha=0.5, label='PF', color=color)
#drawpf = pf[drawmask]  # To plot analysis for the full ensemble
# To plot PF analysis for a 2 members enemble :
weights=norm.pdf(drawbackground, loc=obs, scale=obs_std)
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



#ax0,ax1 = plot_distribution(ax0, ax1, enkf_mean, enkf_std, data=enkf, color='green', marker='x', vmin=vmin, vmax=vmax, label='enkf distribution')

#ax0.axis('off')
ax0.get_xaxis().set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_xlim(left=vmin, right=vmax)
ax0.legend(fontsize=10)

ax1.get_yaxis().set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Precipitation (mm)', fontsize=10)
ax1.set_xlim(left=vmin, right=vmax)
ax1.set_ylim(bottom=0, top=3)

plt.tight_layout()
fig.savefig(f"illustration_assimilation.pdf", format='pdf')
