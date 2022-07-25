#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os
from datetime import datetime,timedelta
import pandas as pd  # Version 0.25.3
import numpy as np
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

map_massifs = dict(
    alpes = [*range(1, 24)],
    pyrenees = [*range(64, 75), *range(80, 92)],
    corse = [40, 41],
)

nb_obs_min = 60


def parse_command_line():
    description = "Evaluation of RADAR products (ANTILOPE or PANTHERE) using nivo-météo network observations"
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-d', '--rundate', help='Rundate for operational executions, format YYMMDDHH')
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
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


def read_data(RADAR_data, datebegin, dateend):
    # I- Lecture et mise en forme des données
    #########################################
    print(f'Reading the following radar data {RADAR_data} from product {args.product}')
    # I.1 Observations nivometeo
    # ---------------------------
    # nivometeo_data = 'obs_nivometeo_daily_RR_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d'), args.dateend.strftime('%Y%m%d'))
    nivometeo_data = 'obs_nivometeo_daily_RR.csv'
    nivometeo = pd.read_csv(nivometeo_data, sep=';', parse_dates=['dat'],
            dtype={'Q.num_poste': int, 'poste_nivo.nom_usuel': str, 'poste_nivo.alti': int, 'poste_nivo.lat_dg': float, 'poste_nivo.lon_dg': float, 'rr': float,
                'poste_nivo.massif_nivo': int, 'hist_reseau_poste.reseau_poste': int})
#    nivometeo_data = 'obs_nivometeo_hourly_RR.csv'
#    nivometeo = pd.read_csv(nivometeo_data, sep=';', parse_dates=['H.dat'],
#            dtype={'H.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'H.rr1':float,
#                'poste_nivo.massif_nivo':int, 'hist_reseau_poste.reseau_poste':int})
    # Renomage de certaines colonnes (pour le merge des DF et pour faciliter la manipulation)
    nivometeo['date'] = nivometeo['dat'].dt.date + pd.Timedelta("1d")  # Changement de type + matching dates with radar data (BDClim extraction
    # for date ymd is the observation from ymd6h to ym(d+1)6h )
    nivometeo = nivometeo.rename(columns={'Q.num_poste': 'num_poste', 'poste_nivo.lat_dg': 'lat', 'poste_nivo.lon_dg': 'lon',
        'poste_nivo.nom_usuel':'name', 'poste_nivo.massif_nivo':'massif_number', 'poste_nivo.alti':'elevation', 'rr':'rr_nivometeo'})  # facultatif

    # I.2 Produit radar
    # ------------------
    antilope = pd.read_csv(RADAR_data, sep=';', parse_dates=['date'], dtype={f'rr_{args.product}': float, 'num_poste': int}, na_values=['--'])
    antilope['date'] = antilope['date'].dt.date

    # II- Merge des DF et mise en forme des données
    ###############################################
    df = pd.merge(antilope, nivometeo, on=["date", "num_poste"])
    # lpn = pd.read_csv('LPN_nivometeo.csv', sep=';', parse_dates=['H_NIVO.DAT'], dtype={'H.num_poste':int, 'H_NIVO.ALTI_LPNX':int}, index_col=['H_NIVO.DAT'])
    lpn = pd.read_csv('LPN_nivometeo.csv', sep=';', parse_dates=['H_NIVO.DAT'], dtype={'H.num_poste':int, 'H_NIVO.ALTI_LPNX':int})
    lpn.rename(columns={'H_NIVO.ALTI_LPNX':'LPNX', 'H.num_poste':'num_poste'}, inplace=True)
    lpn = lpn[lpn['LPNX']>0]  # consider 0 values as missing observation
    lpn['date'] = lpn['H_NIVO.DAT'] + pd.Timedelta("12h")  # La LPN observée à 12:00 D concerne les précipitations entre D (6:00) et D+1 (6:00) que l'on veut identifier
    # par la date ym(D+1), on décale donc de 12h pour que la date de l'obs passe à D+1
    lpn.index = lpn['date']
    lpn = lpn.groupby(['num_poste']).resample('1D').max()  # When 2 observation (at 6:00 and 12:00) are available, set the daily LPN as the maximum
    lpn = lpn[~np.isnan(lpn['LPNX'])]['LPNX'].reset_index()
    lpn.date = lpn.date.dt.date
    df = pd.merge(df, lpn, on=["date", "num_poste"])

    # Pour ne prendre en compte que les situations de neige :
    #df = df[df['elevation']>df['LPNX']]

    # Selection de la période
    df = df.loc[df["date"]>=datetime.date(datebegin)].loc[df["date"]<=datetime.date(dateend)]
    # Retrait des données non exploitables
    df = df.loc[~df['rr_nivometeo'].isna()].loc[~df[f'rr_{args.product}'].isna()]  # Remove lines with missing value
    df = df.loc[df['massif_number']<99]  # Remove Stations not associated to 1 massif
    df = df.loc[~df['name'].str.contains('EDFNIVO')]  # Remove EDFNIVO stations

    # To apply a filter on the minimum number of observations :
    #tmp = df.groupby(['num_poste']).date.count()
    #tmp = tmp.loc[tmp>nb_obs_min] # Consider only points with a minimum number of observations
    #valid_stations = tmp.index.to_numpy()
    #df = df.loc[df['num_poste'].isin(valid_stations)] # Consider only points with a minimum number of observations

    return df


#    suffix = None
#    if args.threshold is not None:
#        df = df.loc[df['rr_nivometeo']>args.threshold] # If a threshold is given, filter data above
#        suffix = f'{args.threshold}mm'
#        nb_obs_min = 20
#    # Calcul des valeurs agrégées par station
#    rr_nivometeo  = df.groupby(['num_poste']).rr_nivometeo.mean()
#    rr_antilope   = df.groupby(['num_poste'])[f'rr_{args.product}'].mean()
##    if args.product in == 'antilope':
##        rr_antilope   = df.groupby(['num_poste']).rr_antilope.mean()
##    elif args.product in == 'antilopejp1':
##        rr_antilope   = df.groupby(['num_poste']).rr_antilopejp1.mean()
##    else:
##        rr_antilope   = df.groupby(['num_poste']).rr_panthere.mean()
#    nb_values     = df.groupby(['num_poste']).date.count()
#    elevations    = df.groupby(['num_poste']).elevation.mean()
#    lats          = df.groupby(['num_poste']).lat.mean()
#    lons          = df.groupby(['num_poste']).lon.mean()
#    massif_number = df.groupby(['num_poste']).massif_number.mean()
#    names         = df.groupby(['num_poste']).name.first()
#    # Regroupement dans une nouvelle dataframe (il est surement possible d'extraire directement cette DF depuis 'df' pour simplifier le code)
#    num_poste = df.groupby(['num_poste']).num_poste.mean()
#    workdict = {'elevation':elevations, 'rr_nivometeo':rr_nivometeo, f'rr_{args.product}':rr_antilope, 'ndays':nb_values, 'massif_number':massif_number,
#            'lats':lats, 'lons':lons, 'num_poste':num_poste}
#    #workdict = {'elevation':elevations, 'rr_nivometeo':rr_nivometeo, f'rr_{args.product}':rr_antilope, 'ndays':nb_values, 'massif_number':massif_number, 'lats':lats, 'lons':lons}
#    df_stat   = pd.DataFrame(workdict)
#    #workdf = workdf.loc[workdf['ndays']>nb_obs_min] # Consider only points with at least 100 observations

def plot(workdf, rain, snow, elevation):
    fig = plt.figure()
    plt.plot(snow['rr_nivometeo'], snow['bias'], linestyle='', marker='+', color='blue', label='Solid precipitation only')
    plt.plot(rain['rr_nivometeo'], rain['bias'], linestyle='', marker='+', color='red', label='Other precipitation')
    plt.title(f'Station n°{station} ({elevation}m, {nb_obs} observations)')
    plt.xlabel("24h precipitation ground observation (mm/24h)", fontsize=12)
    plt.ylabel("ANTILOPE bias", fontsize=12)
    plt.legend()
    fig.savefig(f"radar_bias/station_{station}.svg", format='svg')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(snow['rr_nivometeo'], snow['bias']/snow['rr_nivometeo'], linestyle='', marker='+', color='blue', label='Solid precipitation only')
    plt.plot(rain['rr_nivometeo'], rain['bias']/rain['rr_nivometeo'], linestyle='', marker='+', color='red', label='Other precipitation')
    plt.title(f'Station n°{station} ({elevation}m, {nb_obs} observations)')
    plt.xlabel("24h precipitation ground observation (mm/24h)", fontsize=12)
    plt.ylabel("ANTILOPE relative bias", fontsize=12)
    plt.legend()
    fig.savefig(f"relative_bias/station_{station}.svg", format='svg')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(snow['rr_nivometeo'], snow[f'rr_{args.product}'], linestyle='', marker='+', color='blue', label='Solid precipitation only')
    plt.plot(rain['rr_nivometeo'], rain[f'rr_{args.product}'], linestyle='', marker='+', color='red', label='Other precipitation')
    plt.title(f'Station n°{station} ({elevation}m, {nb_obs} observations)')
    plt.xlabel("24h precipitation ground observation (mm/24h)", fontsize=12)
    plt.ylabel("24h ANTILOPE precipitation estimate (mm/24h)", fontsize=12)
    plt.legend()
    fig.savefig(f"scatterplot/station_{station}.svg", format='svg')
    plt.close(fig)

    rrmax = max([np.max(workdf[f'rr_{args.product}']), np.max(workdf['rr_nivometeo'])])
    dx = 0.1
    x = np.arange(0, rrmax, dx)
    y = [len(workdf.loc[workdf[f'rr_{args.product}'] <= rr]) for rr in x]
    z = [len(workdf.loc[workdf['rr_nivometeo'] <= rr]) for rr in x]
    fig = plt.figure()
    plt.plot(x, y, linestyle='-', marker='', color='blue', label='ANTILOPE')
    plt.plot(x, z, linestyle='-', marker='', color='red', label='Nivometeo')
    plt.title(f'Station n°{station} ({elevation}m, {nb_obs} observations)')
    plt.xlabel("24h-precipitation (mm/24h)", fontsize=12)
    plt.ylabel("Observation frequency", fontsize=12)
    plt.legend()
    fig.savefig(f"CDF/station_{station}.svg", format='svg')
    plt.close(fig)

def plot_pdf():
    nbobs = len(df)
    rrmax = max([np.max(df[f'rr_{args.product}']), np.max(df[f'rr_nivometeo'])])
    rrmax=10
    dx = 0.1
    x = np.arange(0, rrmax, dx)
    pdfradar = [len(df.loc[(df[f'rr_{args.product}'] >= rr) & (df[f'rr_{args.product}'] < rr+dx)]) / nbobs for rr in x]
    pdfnivometeo = [len(df.loc[(df[f'rr_nivometeo'] >= rr) & (df[f'rr_nivometeo'] < rr+dx)]) / nbobs for rr in x]
    fig = plt.figure()
    plt.plot(x, pdfradar, linestyle='-', marker='', color='blue', label='ANTILOPE')
    plt.plot(x, pdfnivometeo, linestyle='-', marker='', color='red', label='Nivometeo')
    plt.title(f'{nbobs} observations')
    plt.xlabel("24h-precipitation (mm/24h)", fontsize=12)
    plt.ylabel("Probability of observation ", fontsize=12)
    plt.legend()
    fig.savefig(f"PDF_observations.svg", format='svg')
    plt.close(fig)


if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend)
    if args.product == 'antilope':
        RADAR_data = 'ANTILOPEQ_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
    elif args.product == 'antilopejp1':
        RADAR_data = 'ANTILOPEJP1Q_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
    elif args.product == 'kriging':
        RADAR_data = 'Kriging_{0:s}_{1:s}_{2:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), 'exponential')
    else:
        RADAR_data = 'PANTHERE_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))

    df = read_data(RADAR_data, args.datebegin, args.dateend)
    df['bias'] = df[f'rr_{args.product}'] - df['rr_nivometeo']

#    plot_pdf()
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
