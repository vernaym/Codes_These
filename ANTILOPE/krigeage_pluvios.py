#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os,sys
import datetime
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from snowtools.plots.maps import cartopy

from pykrige.ok import OrdinaryKriging


gridx = np.arange(5.2, 7.9, 0.01)
gridy = np.arange(43.9, 46.5, 0.01)

def parse_command_line():
    description = "Evaluation of RADAR products (ANTILOPE or PANTHERE) using nivo-météo network observations"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    args = parser.parse_args()

    args.datebegin = get_date(args.datebegin)
    if args.dateend:
        args.dateend = get_date(args.dateend)
    else:
        args.dateend = args.datebegin + timedelta(hours=24)

    return args

def get_date(a_string):
    try:
        date = datetime.strptime(a_string, '%Y%m%d').replace(hour=6, minute=0, second=0, microsecond=0)
    except ValueError:
        try:
            date = datetime.strptime(a_string, '%y%m%d').replace(hour=6, minute=0, second=0, microsecond=0)
        except ValueError:
            try:
                date = datetime.strptime(a_string, '%Y%m%d%H').replace(minute=0, second=0, microsecond=0)
            except ValueError:
                try:
                    date = datetime.strptime(a_string, '%y%m%d%H').replace(minute=0, second=0, microsecond=0)
                except ValueError:
                    print('The start date provided is not in a good format (YYYYMMDDHH or YYMMDDHH or YYYYMMDDHHMM or YYMMDD or YYYYMMDD)')
                    raise
    finally:
        return date 

def date_range(start, end, dt=24):
    start = start.replace(hour=6)
    dates = list()
    while start <= end:
        dates.append(start)
        start += timedelta(hours=dt)
    
    return dates

def goto(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)


if __name__ == "__main__":
    args = parse_command_line()


    extract_period = date_range(args.datebegin, args.dateend)
    fic = "obs_quotidiennes_RR_reseau_poste_full.data"
    pluvios = pd.read_csv(fic, sep=';', parse_dates=['dat'], dtype={'num_poste':int, 'poste':str, 'lat':float, 'lon':float, 'alti':int, 'rr':float, 'reseau_poste':int}, na_values=['--'])
    pluvios = pluvios.loc[~pluvios['rr'].isna()]
    for rundate in extract_period:
        print(rundate)
        startdate = rundate - timedelta(hours=24)
        # Extract hourly precipitation of the last 23-hours
        tmp = pluvios.loc[pluvios['dat']<=rundate].loc[pluvios['dat']>startdate]
        nval = tmp.groupby(['num_poste']).dat.count()
        y = tmp.groupby(['num_poste']).lat.mean()[nval==24]
        x = tmp.groupby(['num_poste']).lon.mean()[nval==24]
        rr   = tmp.groupby(['num_poste']).rr.sum()[nval==24]

        OK = OrdinaryKriging(x.values, y.values, rr.values, variogram_model='gaussian')
        z, ss = OK.execute('grid', gridx, gridy)

#        plt.imshow(z, interpolation='none', cmap='jet')
#        plt.show()
#        import pdb
#        pdb.set_trace()

        fig = cartopy.Map_alpes()
        lon, lat = np.meshgrid(gridx, gridy)
        cf = plt.contourf(lon, lat, z, levels=100, cmap='YlGnBu')
        plt.colorbar(cf, label='24h precipitation (mm)')
        fig.init_massifs()
        fig.addpoints(x, y, labels=np.around(rr,1))
        plt.tight_layout()
        fig.save('map_{0:s}.svg'.format(rundate.strftime('%Y%m%d%H')), formatout='svg')
        fig.close()



