#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
from datetime import datetime,timedelta
import pandas as pd  # Version 0.25.3
import numpy as np
import xarray as xr
#import copy

import argparse

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
#import seaborn as sns

#from sklearn.linear_model import LinearRegression, RANSACRegressor
#from sklearn.datasets import make_regression
#from sklearn.metrics import mean_squared_error, r2_score

#from scipy.stats import gaussian_kde

import statistics

#from snowtools.plots.maps import cartopy


##############################################################################################
##############################################################################################

datadir = '/home/vernaym/These/DATA'

def parse_command_line():
    description = "Evaluation of RADAR products (ANTILOPE or PANTHERE) using nivo-météo network observations"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-d', '--domain', nargs='+', help='Domain of the file', choices=['alp', 'pyr', 'cor', 'GrandesRousses'], default='GrandesRousses')
    parser.add_argument('-w', '--workdir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/workdir/extraction_antilope')
    parser.add_argument('-m', '--massif', help='PLot for a specific massif', default=None, type=int)
    parser.add_argument('-t', '--threshold', default=None, help='Threshold of precipitation (mm) to apply in the data to consider', type=int)
    parser.add_argument('-p', '--product', default='antilopejp1', help='Product to deal with', choices=['antilope', 'antilopejp1', 'panthere', 'kriging'])
    parser.add_argument('-l', '--lpn', action='store_true', help='Take into account the rain-snow limit')

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

# Definition of gamma distribution
def gamma_shape_PDF(x, k=3, theta=1):
    import math
    if isinstance(x, np.ndarray):
        return list(map(lambda t: t**(k-1)*np.exp(-t/theta)/(theta**k*math.gamma(k)) if t > 0 else None, x))
    else:
        return x**(k-1)*np.exp(-x/theta)/(theta**k*math.gamma(k)) if x > 0 else None

# Definition of SCGD
def SCGD_shape_PDF(x, k=1., theta=1., delta=0., plot_distribution=False, plot_parameters=False):
    if not (isinstance(x, (list, np.ndarray))):
        x = np.array([x])
    # On n'impose pas de condition sur delta : on peut vouloir translater la distribution vers la droite ou vers la gauche
#    if delta > 0:
#        # WARNING : dans Schuerer and Hammil 2015 and Nusu 2019 delat est définit comme > 0, ce qui entraine un
#        # décallage vers la droite plutot que vers la gauche...
#        raise ValueError('Error : delta parameter must be <= 0')
    dy = 0.01

    # WARNING : la distribution obtenue sera discontinue en 0 et potentiellement accordera trop / trop peu de
    # poids aux precipitations nulles
    y = np.arange(delta, theta*(k-1)+delta+3*k*theta**2, dy)  # WARNING : 'y' doit ABSOLUMENT couvrir toute la distribution
                                    # de gamma sinon la probabilité en 0 est artificiellement surestimée !
    gamma = np.array(list(map(
        lambda t: gamma_shape_PDF(t-delta, k=k, theta=theta) if t > delta else 0, y)))
    scgd0 = 1 - sum(gamma[y>0]*dy)  # Calcul de la probabilité résiduelle en 0

    if plot_distribution:
        fig,ax = plt.subplots()
        color = next(ax._get_lines.prop_cycler)['color']
        # Plot over a smaller range for better lisibility
        ymin = theta*(k-1)+delta-k*theta**2
        ymax = theta*(k-1)+delta+1.5*k*theta**2
        ax.plot(y[y<ymax], gamma[y<ymax], linestyle=':', color=color)
        ax.plot(y[(y>0) & (y<ymax)], gamma[(y>0) & (y<ymax)],
                label=f'SCGD (k={k:0.2}, theta={theta:0.2}, delta={delta:0.2}, mu={mu1:0.2}, bias={mu0:0.2}, sigma={sigma:0.2})', color=color)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=Y, color='k', linestyle='--', label=f'Observation : {Y:0.2}')
        plt.axvline(x=mu1, color='red', linestyle='--', label=f'Observation + mean bias : {mu1:0.2}')
        plt.axvline(x=mu, color='blue', linestyle='--', label=f'Artificial max : {mu:0.2}')
        if scgd0 > 0.0001:
            ax.plot(0, scgd0, marker='.', markersize=20, markeredgecolor=color, markeredgewidth=2,
                     color='none', markerfacecolor='lightgrey', label=f'P(0)={scgd0:0.3}')
            ax.fill_between(x=y, y1=gamma, where=(delta < y) & (y < 0), color='lightgrey')
        #ax.bar(0, scgd0, width=0.2, align='edge', color=color)
        #ax.bar(0, scgd0, width=0.2, color=color)
        plt.legend(loc ="upper right")
        fig.savefig(f"PDF/station_{station}_{Y}mm.svg", format='svg')

    if plot_parameters:
        #plt.plot(Y, k, linestyle='', marker='.', color='blue', label='k')
        #plt.plot(Y, theta, linestyle='', marker='+', color='blue', label='theta')
        plt.plot(Y, delta, linestyle='', marker='+', color='k', label='delta')
        plt.plot(Y, scgd0, linestyle='', marker='+', color='red', label='P(0)')
        plt.plot(Y,gamma_shape_PDF(0.1-delta, k=k, theta=theta), marker='+', color='blue', label='P0+')

        if Y == 0:
            plt.legend(loc ="upper right")

    return np.array(list(map(lambda t: gamma_shape_PDF(t-delta, k=k, theta=theta) if t > 0
                    else scgd0 if t == 0 else 0, x)))

def read_ensemble(datebegin, dateend, domain='GrandesRousses'):
    pearome = dict()
    for member in range(1,17):
        #filename = f'pearome_{member:03d}_{datebegin}_{dateend}.nc'
        # TODO : Verrue
        filename = f'pearome_{member:03d}_2021073106_2022071506.nc'
        filename = os.path.join(datadir, filename)
        pearome[member] = xr.open_dataset(filename)

    return pearome


if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend)
    filename = 'ANTILOPEQ_{0:s}_{1:s}_{2:s}.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), args.domain)
    if not os.path.exists(filename):
        print(f'WARNING : file {filename} does not exist, looking for it under {datadir}')
        filename = os.path.join(datadir, filename)
    if os.path.exists(filename):
        antilope = xr.open_dataset(filename)
    else:
        print(f'ERROR : file {filename} does not exist')
        sys.exit(1)
    pearome = read_ensemble(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))

    import pdb
    pdb.set_trace()

    date = args.datebegin
    while date <= args.dateend:
        antilope.sel(time=date)
        print()
        date = date + timedelta(days=1)


    x = list()
    y = list()
    alti = list()
    for station in np.unique(df['num_poste']):
        workdf = df.loc[df['num_poste'] == station]
        elevation = int(workdf['elevation'].mean())
        nb_obs = len(workdf)
        if nb_obs > 10:
            snow = workdf.loc[workdf['elevation'] > workdf['LPNX']]
            rain = workdf.loc[workdf['elevation'] <= workdf['LPNX']]

#            plot(workdf, rain, snow, elevation)

            mu0 = workdf['bias'].mean()
            sigma = statistics.stdev(workdf['bias'].values)
            x.append(mu0)
            y.append(sigma)
            alti.append(elevation)

            fig1 = plt.figure()
            for Y in np.arange(0, 5, 0.1):
                mu1 = Y - mu0
                #delta = delta0*np.exp(-mu)
                #mu = mu - delta
                delta = -2*np.exp(-Y)
                mu = mu1 - delta
                k = (2*sigma**2+mu**2+np.sqrt((mu**2*(4*sigma**2+mu**2))))/(2*sigma**2)  # condition : k > 1
                theta = mu / (k-1)
                SCGD_shape_PDF(Y, k=float(k), theta=float(theta), delta=float(delta), plot_parameters=True)
            fig1.savefig(f"parameters/station_{station}.svg", format='svg')


#    fig0,ax0 = plt.subplots()
#    sc = ax0.scatter(x, y, c=alti, marker="^")
#    plt.colorbar(sc, label='Elevation (m)')
#    ax0.set_xlabel('Mean ANTILOPE / rain gauge deviation (mm)')
#    ax0.set_ylabel('ANTILOPE / rain gauge standard deviation (mm)')
#    fig0.savefig('Mean_std_scatterplot.svg', format='svg')
#    plt.close(fig0)
