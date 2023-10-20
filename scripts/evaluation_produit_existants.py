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

savedir = '/home/vernaym/These/figures/evaluation/produits_existants'

suffix = ''
if len(sys.argv) > 1:
    suffix = f'_{sys.argv[1]}'

subdir = 'onlysnow'
#subdir = ''

labels = list()
fig, ax = plt.subplots()

filename = os.path.join('/home/vernaym/workdir/evaluation_PANTHERE/2018_2019', subdir, f'scores_2018120106_2019043006_alpes{suffix}.csv')
panthere = pd.read_csv(filename, sep=';')
labels.append(scores.add_label(plt.violinplot(panthere.freq_error, positions=[1], showmeans=True), 'PANTHERE'))

filename = os.path.join('/home/vernaym/workdir/evaluation_ANTILOPE/krigeage_pluvios_antilope/krigeage_sans_obs_clim', subdir, f'scores_2018120106_2019043006_alpes{suffix}.csv')
krigeage = pd.read_csv(filename, sep=';')
labels.append(scores.add_label(plt.violinplot(krigeage.freq_error, positions=[2], showmeans=True), 'KRIGEAGE'))

filename = os.path.join('/home/vernaym/workdir/evaluation_ANTILOPE/ANTILOPEQ/2018-2019', subdir, f'scores_2018110106_2019043006_alpes{suffix}.csv')
antilope = pd.read_csv(filename, sep=';')
labels.append(scores.add_label(plt.violinplot(antilope.freq_error, positions=[3], showmeans=True), 'ANTILOPE'))

ax.legend(*zip(*labels), fontsize=10, loc='upper left')
ax.set_ylabel('Frequency of error below 20%')
plt.tight_layout()
fig.savefig(os.path.join(savedir, f"freq_error{suffix}{subdir}.pdf"), format='pdf')
plt.close(fig)
