#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
import glob
import time
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import rankdata
import CRPS.CRPS as pscore

from These.scripts import scores, tools

import argparse

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print('USAGE : evaluation_produit_existants.py [onlysnow]')

savedir = '/home/vernaym/These/figures/evaluation/produits_existants'

subdir = ''
if len(sys.argv) > 1:
    subdir = sys.argv[1]

fig, axes = plt.subplots(1, 2, figsize=(12,6))
for i,threshold in enumerate([0, 10]):

    if threshold == 0:
        suffix = ''
    else:
        suffix = f'_{threshold}'

    ax = axes[i]

    labels = list()

    filename = os.path.join('/home/vernaym/workdir/evaluation_PANTHERE/2018_2019', subdir, f'scores_2018120106_2019043006_alpes{suffix}.csv')
    panthere = pd.read_csv(filename, sep=';')
    #labels.append(scores.add_label(ax.violinplot(panthere.freq_error, positions=[1], showmeans=True), 'PANTHERE'))
    labels.append(scores.add_label(ax.violinplot(panthere.rmse, positions=[1], showmeans=True), 'PANTHERE'))

    filename = os.path.join('/home/vernaym/workdir/evaluation_ANTILOPE/krigeage_pluvios_antilope/krigeage_sans_obs_clim', subdir, f'scores_2018120106_2019043006_alpes{suffix}.csv')
    krigeage = pd.read_csv(filename, sep=';')
    #labels.append(scores.add_label(ax.violinplot(krigeage.freq_error, positions=[2], showmeans=True), 'KRIGING'))
    labels.append(scores.add_label(ax.violinplot(krigeage.rmse, positions=[2], showmeans=True), 'KRIGING'))

    filename = os.path.join('/home/vernaym/workdir/evaluation_ANTILOPE/ANTILOPEQ/2018-2019', subdir, f'scores_2018110106_2019043006_alpes{suffix}.csv')
    antilope = pd.read_csv(filename, sep=';')
    #labels.append(scores.add_label(ax.violinplot(antilope.freq_error, positions=[3], showmeans=True), 'ANTILOPE'))
    labels.append(scores.add_label(ax.violinplot(antilope.rmse, positions=[3], showmeans=True), 'ANTILOPE'))

    filename = os.path.join('/home/vernaym/workdir/evaluation_PEAROME/aspearome', subdir, f'scores_2021110106_2022043006_alpes{suffix}.csv')
    pearome = pd.read_csv(filename, sep=';')
    #labels.append(scores.add_label(ax.violinplot(antilope.freq_error, positions=[3], showmeans=True), 'ANTILOPE'))
    labels.append(scores.add_label(ax.violinplot(pearome.rmse, positions=[4], showmeans=True), 'AS-PEAROME'))

    if suffix in ['', '_1']:
        ax.legend(*zip(*labels), fontsize=10, loc='upper left')
        #ax.set_ylabel('Frequency of error below 20%')
        ax.set_ylabel('Root mean square error (mm)')
        if subdir == 'onlysnow':
            ax.set_title('a) All solid precipitation events')
        else:
            ax.set_title('a) All precipitation events')
    else:
        # Remove y axislabel
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off
        if subdir == 'onlysnow':
            ax.set_title('b) Solid precipitation events above 10mm / 24h')
        else:
            ax.set_title('b) Precipitation events above 10mm / 24h')

    ax.set_ylim(bottom=0, top=50)
    # Remove xaxis labels
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

plt.tight_layout()
#fig.savefig(os.path.join(savedir, f"freq_error{suffix}{subdir}.pdf"), format='pdf')
fig.savefig(os.path.join(savedir, f"RMSE{subdir}.pdf"), format='pdf')
plt.close(fig)
