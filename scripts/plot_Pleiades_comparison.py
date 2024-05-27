#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 18 march 2024

@author: Vernay

WORK IN PROGRESS

example:
--------
'''

import os
import numpy as np
import pandas as pd
import xarray as xr
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from snowtools.scripts.extract.vortex import vortexIO as io
from snowtools.scores import clusters
import snowtools.tools.xarray_preprocess as xrp
from snowtools.plots.boxplots import violinplot


def parse_command_line():
    description = "Plot figures comparing snow depth simulation(s) to a Pleiade observation"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-b', '--datebegin', type=str,
                        help="First date covered by the simulation file, format YYYYMMDDHH.")

    parser.add_argument('-e', '--dateend', type=str,
                        help="Last date covered by the simulation file, format YYYYMMDDHH.")

    parser.add_argument('-d', '--date', type=str, required=True,
                        help="Date of the reference Pleiade observation."
                             "Format YYYYMMDDHH")

    parser.add_argument('-x', '--xpids', nargs='+', type=str,
                        help="XPID(s) of the simulation(s) format XP_NAME@username")

    parser.add_argument('-a', '--vapp', type=str, default='edelweiss', choices=['s2m', 'edelweiss'],
                        help="Application that produced the target file")

    parser.add_argument('-u', '--uenv', type=str, default="uenv:edelweiss.1@vernaym",
                        help="User environment for static resources (format 'uenv:name@user')")

    #parser.add_argument('-t', '--thresholds', nargs='+', type=list, default=np.arange(2, 21, 4),
    parser.add_argument('-t', '--thresholds', nargs='+', type=list, default=np.arange(2, 30, 6),
                        help='Define bands for clustering')

    parser.add_argument('-w', '--workdir', type=str, default=f'{os.environ["HOME"]}/workdir/EDELWEISS/plot/Pleiades',
                        help='Working directory')

    parser.add_argument('-g', '--geometry', type=str, default='GrandesRousses250m',
                        help='Geometry of the simulation(s)')

    parser.add_argument('-o', '--obs_geometry', type=str, choices=['Lautaret250m', 'Huez250m'],
                        help='Geometry of the observation')

    args = parser.parse_args()
    return args


def read_uncertainty(var='Uncertainty'):
    if var == 'Uncertainty':
        ds = xr.open_dataset('/home/vernaym/workdir/ASSIMILATION/mask/GrandesRousses/Observation_error_L93.nc')
    elif var == 'Ratio':
        ds = xr.open_dataset('/home/vernaym/workdir/ASSIMILATION/mask/GrandesRousses/Estimated_ratio_L93.nc')
    ds = xrp.update_varname(ds)
    ds = ds.interp({'xx': obs.xx, 'yy': obs.yy})
    return ds[var]


def execute(xpids, obs, date, mask=True, member=None):

    dsplot = obs.rename({'DSN_T_ISBA': 'Obs'})

    # var = 'Ratio'
    var = 'Uncertainty'
    uncertainty = read_uncertainty(var)
    dsplot[var] = xr.where(~dsplot.Obs.isnull(), uncertainty, np.nan)

    filtered_obs = clusters.per_uncertainty(obs.DSN_T_ISBA, uncertainty, thresholds)
    dataplot = filtered_obs.to_dataframe(name='obs').dropna().reset_index().drop(columns=['xx', 'yy', 'time'])

    for xpid in xpids:
        print(xpid)
        proname = os.path.join(f'PRO_{xpid}.nc')
        simu = xr.open_dataset(proname, decode_times=False)
        # TODO : gérer le problème de coordonnées pour éviter les "rename" très lents !
        simu = xrp.decode_time(simu)
        simu = simu.sel({'xx': obs.xx, 'yy': obs.yy, 'time': pd.to_datetime(date[:8], format='%Y%m%d')})
        dsplot['Simu'] = xr.where(~dsplot.Obs.isnull(), simu.DSN_T_ISBA, np.nan)
        scatterplot(dsplot, xpid, var, date)

        filtered_simu = clusters.per_uncertainty(dsplot.Simu, uncertainty, thresholds)
        df = filtered_simu.to_dataframe(name=xpid).dropna().reset_index().drop(columns=['xx', 'yy', 'time'])
        # Concatenate datasets into the *dataplot* DataFrame
        dataplot = pd.concat([dataplot, df])

    plt.close('all')

    dataplot.columns = dataplot.columns.str.replace('middle_slices', 'Uncertainty')
    dataplot = dataplot.melt('Uncertainty', var_name='experiment', value_name='DSN_T_ISBA')

    title = f'Pleiades, {geometry}, {date[:8]}\n'
    violinplot.plot_ange(dataplot, 'DSN_T_ISBA', figname=f'{var}_{date}_' + '_'.join(xpids),
            title=title, yaxis="Uncertainty", violinplot=False)


def scatterplot(dataplot, xpid, xvar, date):
    df = dataplot.to_dataframe()
    if xvar == 'Uncertainty':
        df['yvar'] = ((df.Simu - df.Obs) / df.Obs).abs()
        # df['yvar'] = (df.Simu - df.Obs).abs()
        xlim = (2, 20)
        ylim = (-1, 5)
    elif xvar == 'Ratio':
        df['yvar'] = df.Simu / df.Obs
        xlim = (0, 2)
        ylim = (0, 2)
    sns.lmplot(x=xvar, y='yvar', data=df, fit_reg=True, markers='+')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.savefig(f'{xpid}_{date}.pdf')


if __name__ == '__main__':

    args = parse_command_line()
    datebegin       = args.datebegin
    dateend         = args.dateend
    date            = args.date
    xpids           = args.xpids
    workdir         = args.workdir
    geometry        = args.geometry
    obs_geometry    = args.obs_geometry
    vapp            = args.vapp
    uenv            = args.uenv
    thresholds      = args.thresholds

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    os.chdir(workdir)

    # 1. Get all input data

    # a) Pleiades observations
    kw = dict(date=datebegin, vapp=vapp)
    obsname = f'PLEIADES_{date}.nc'
    io.get_snow_obs_date(xpid='CesarDB_AngeH', geometry=obs_geometry, date=date, vapp='Pleiades', filename=obsname)
    obs = xr.open_dataset(obsname)
    obs = xrp.update_varname(obs)

    # d) Simulations
    for xpid in xpids:
        # TODO : gérer ça plus proprement
        if '@' not in xpid:
            user = os.environ["USER"]
            xpid = f'{xpid}@{user}'
        shortid = xpid.split('@')[0]
        # VERRUE pour gérer le décallage d'un jour en attendant de combler les données
#        if shortid.startswith('safran'):
#            deb = '2021080106'
#        else:
#            deb = datebegin  # 2021080207

        # TODO : à gérer autrement pour être flexible
        if shortid in ['SAFRAN', 'ANTILOPE', 'SAFRAN_pappus', 'ANTILOPE_pappus']:
            member = None
        else:
            member = 0

        # Get (filtered) PRO files with Vortex
        kw = dict(datebegin=datebegin, dateend=dateend, vapp=vapp, member=member, namebuild=None,
                filename=f'PRO_{shortid}.nc')
        io.get_pro(xpid=xpid, geometry=geometry, **kw)

    # TODO : à gérer autrement pour être flexible
    member = None

    execute(xpids, obs, date, member=member)

    # 3. Clean data
    for xpid in xpids:
        shortid = xpid.split('@')[0]
        if member is None:
            os.remove(f'PRO_{shortid}.nc')
        else:
            for member in range(member):
                os.remove(f'mb{member:03d}/PRO_{shortid}.nc')

    print(f'Execution OK,outputs under {workdir}')
