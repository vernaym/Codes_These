#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 05/11/2022

import os, sys
from datetime import datetime,timedelta
import numpy as np
import xarray as xr

import argparse

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Annotation

###########################################
# USAGE : p plot_field.py $filename [cumul]
###########################################


filename = sys.argv[1]
field = xr.open_dataarray(filename)

cumul = False
figname = filename.replace('.nc', '.pdf')
if len(sys.argv)>2:
    cumul=True
    figname = f"CUMUL_{figname}"
    if 'member' in field.dims:
        field = field.sum(dim=['time']).mean(dim=['member'])
    else:
        field = field.sum(dim=['time'])


fig, ax = plt.subplots(figsize=(12,6))
#field.plot(ax=ax, cbar_kwargs={"label":'Total precipitation between {0:s} and {1:s} (mm)'.format(args.datebegin.strftime('%Y%m%d'), args.dateend.strftime('%Y%m%d'))}, cmap=plt.cm.coolwarm)
#field.plot(ax=ax, cmap=plt.cm.Greys)
field.plot(ax=ax, cmap=plt.cm.YlGnBu)

#add_landmarks()
#add_radar_positions(ax)
#add_scores()
plt.show()
fig.savefig(figname, format='pdf', bbox_inches='tight')




