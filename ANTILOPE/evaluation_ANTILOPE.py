#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os
import datetime
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import argparse

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

##############################################################################################
# Ce script sert permet de comparer les données ANTILOPE extraites de la BDAP par le script
# "Extraction_ANTILOPE_bdap.py aux observations des postes du réseau nivométéo correspondants 
# extraites par le script "extract_obs_nivometeo.py" de snowtools dans le but d'évaluer le 
# biais d'ANTILOPE avec l'altitude.
##############################################################################################

ANTILOPE_data = 'ANTILOPE_2015120106_2020031506.csv'
nivometeo_data = 'obs_nivometeo_daily_RR_20151201_20200315.csv'

def parse_command_line():
    description = "BDAP extraction of ANTILOPE data."
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-d', '--rundate', help='Rundate for operational executions, format YYMMDDHH')
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-w', '--workdir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/workdir/extraction_antilope')

    args = parser.parse_args()

    args.datebegin = get_date(args.datebegin)
    if args.dateend:
        args.dateend = get_date(args.dateend)
    else:
        args.dateend = args.datebegin + timedelta(hours=24)

    return args

def nivologyseason(date):
    """Return the nivology season of a current date"""
    if date.month < 8:
        season_begin = datetime(date.year - 1, 8, 1)
        season_end   = datetime(date.year, 7, 31)
    else:
        season_begin = datetime(date.year, 8, 1)
        season_end   = datetime(date.year + 1, 7, 31)

    return season_begin.strftime('%y') + season_end.strftime('%y')

def get_date(a_string):
    try:
        date = datetime.strptime(a_string, '%Y%m%d%H').replace(minute=0, second=0, microsecond=0)
    except ValueError:
        try:
            date = datetime.strptime(a_string, '%y%m%d%H').replace(minute=0, second=0, microsecond=0)
        except ValueError:
            try:
                date = datetime.strptime(a_string, '%Y%m%d%H%M').replace(second=0, microsecond=0)
            except ValueError:
                try:
                    date = datetime.strptime(a_string, '%Y%m%d').replace(hour=6, minute=0, second=0, microsecond=0)
                except ValueError:
                    try:
                        date = datetime.strptime(a_string, '%y%m%d').replace(hour=6, minute=0, second=0, microsecond=0)
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

def read_nivometeo_coords(domain):
    metadata = pd.read_csv('postes_nivometeo.csv', sep=';')
    latmax, latmin, lonmin, lonmax = coords[domain]
    subdata = metadata[metadata['lat'].isin(range(int(latmin), int(latmax))) & metadata['lon'].isin(range(int(lonmin), int(lonmax)))]
    return dict(zip(np.array(subdata['num_poste']), zip(np.array(subdata['lat']), np.array(subdata['lon']))))

def predict(x):
   return slope * x + intercept       

def raw_scatterplot(rr_nivometeo, rr_antilope, elevations, datebegin, dateend):
    fig = plt.figure(figsize=(10,6))
    #plt.scatter(rr_nivometeo.to_numpy(), rr_antilope.to_numpy(), s=nb_values.to_numpy(), marker='o')
    sc = plt.scatter(rr_nivometeo, rr_antilope, marker='D', s=10, c=elevations)
    plt.colorbar(sc, label='Elevation')
    xpoints = ypoints = plt.xlim()
    plt.grid(visible=True, linestyle=':', linewidth=0.5)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.plot(xpoints, ypoints, linestyle='--', color='grey', lw=1, scalex=False, scaley=False)
    plt.ylabel('Total ANTILOPE precipitation estimate (mm)', fontsize=12)
    plt.xlabel('Total rain-gauges observed precipitation (mm)', fontsize=12)

    # Regression linéaire
    reg = LinearRegression().fit(rr_nivometeo.reshape((-1, 1)), rr_antilope)
    y = reg.predict(rr_nivometeo.reshape((-1,1)))
    r2 = r2_score(rr_nivometeo.reshape((-1, 1)), rr_antilope)
    plt.plot(rr_nivometeo, y, color="red", linewidth=1)
    plt.text(3500, 2100, f'R²={r2:.4}', fontsize=18, color='red')

    plt.tight_layout()
    #fig.savefig('raw_scatterplot_{0:s}_{1:s}.pdf'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='pdf')
    fig.savefig('raw_scatterplot_{0:s}_{1:s}.png'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='png')

def elevation_scatterplot(workdf, datebegin, dateend):

    fig = plt.figure(figsize=(12,9))

    elevation_thresholds = [1450, 1725, 2000, 4000] # Based on the main "jumps"
    elevation_thresholds = [1530, 1739, 1967, 4000] # Based on the quantiles
    elevation_thresholds = [np.quantile(workdf['elevation'], q) for q in [0.25, 0.5, 0.75, 1]]
    colors = ['darkblue', 'cyan', 'gold', 'red']

    workdf = workdf.sort_values('elevation')

    for i, threshold in enumerate(elevation_thresholds):
        index_names = workdf[workdf['elevation'] < threshold].index
        tmp = workdf.loc[workdf.index.isin(index_names)]
        workdf.drop(index_names, inplace = True)
        x = tmp['rr_nivometeo'].to_numpy()
        y = tmp['rr_antilope'].to_numpy()
        if i == 0:
            label = '< {0:d} m'.format(int(threshold)) 
        elif i == 3:
            label = '> {0:d} m'.format(int(elevation_thresholds[i-1]))
        else:
            label = '[{0:d} m - {1:d} m]'.format(int(elevation_thresholds[i-1]), int(elevation_thresholds[i]))
        sc = plt.scatter(x, y, marker='D', s=10, c=colors[i], label=label)
        # Regression linéaire
        reg = LinearRegression().fit(x.reshape((-1, 1)), y)
        y = reg.predict(x.reshape((-1,1)))
        plt.plot(x, y, color=colors[i], linewidth=1)

    plt.legend()
    xpoints = ypoints = plt.xlim()
    plt.grid(visible=True, linestyle=':', linewidth=0.5)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.plot(xpoints, ypoints, linestyle='--', color='grey', lw=1, scalex=False, scaley=False)
    plt.ylabel('Total ANTILOPE precipitation estimate (mm)', fontsize=12)
    plt.xlabel('Total rain-gauges observed precipitation (mm)', fontsize=12)


    #plt.tight_layout()
    #fig.savefig('scatterplot_{0:s}_{1:s}.pdf'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='pdf')
    fig.savefig('elevation_scatterplot_{0:s}_{1:s}.png'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='png')

if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend)
    #antilope = pd.read_csv(ANTILOPE_data, sep=';', index_col='date')
    #nivometeo = pd.read_csv(nivometeo_data, sep=';', index_col='dat')
    antilope = pd.read_csv(ANTILOPE_data, sep=';', parse_dates=['date'])
    nivometeo = pd.read_csv(nivometeo_data, sep=';', parse_dates=['dat'])
    antilope['date'] = antilope['date'].dt.date
    nivometeo['date'] = nivometeo['dat'].dt.date
    #antilope = antilope.set_index('date')
    #nivometeo = nivometeo.set_index('dat')
    nivometeo["num_poste"] = nivometeo["Q.num_poste"]
    df = pd.merge(antilope, nivometeo, on=["date", "num_poste"])
    df = df.loc[df["date"]>=datetime.date(args.datebegin)].loc[df["date"]<=datetime.date(args.dateend)]
    df = df.rename(columns={'poste_nivo.alti':'alti'})
    df = df.loc[~df['rr'].isna()].loc[~df['rr_antilope'].isna()][['date', 'num_poste', 'rr_antilope', 'rr', 'alti']]
    df['diff'] = df['rr_antilope'] - df['rr']
    df['diff'].describe()
    rr_nivometeo = df.groupby(['num_poste']).rr.sum()
    rr_antilope  = df.groupby(['num_poste']).rr_antilope.sum()
    nb_values    = df.groupby(['num_poste']).date.count()
    elevations   = df.groupby(['num_poste']).alti.mean()

    #raw_scatterplot(rr_nivometeo.to_numpy(), rr_antilope.to_numpy(), elevations.to_numpy(), args.datebegin, args.dateend)
    workdict = {'elevation':elevations, 'rr_nivometeo':rr_nivometeo, 'rr_antilope':rr_antilope, 'nb_values':nb_values}
    workdf   = pd.DataFrame(workdict)
    elevation_scatterplot(workdf, args.datebegin, args.dateend)


