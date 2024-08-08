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

# Following installation required on sxcen :
# - Install xskillscore : pip install xskillscore
# - Upgarde numba       : pip install numba --upgrade
import xskillscore  # Requires an installation : pip install xskillscore
# https://xskillscore.readthedocs.io/en/stable/api/xskillscore.crps_ensemble.html

import snowtools.tools.xarray_preprocess as xrp
from snowtools.scripts.extract.vortex import vortexIO as io

from snowtools.scripts.post_processing import common_dict

members_map = common_dict.members_map
product_map = common_dict.product_map
xpid_map    = common_dict.xpid_map
colors_map  = common_dict.colors_map


def parse_command_line():
    description = "Plot figures comparing snow depth simulation(s) to a Pleiade observation"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-b', '--datebegin', type=str,
                        help="First date covered by the simulation file, format YYYYMMDDHH.")

    parser.add_argument('-e', '--dateend', type=str,
                        help="Last date covered by the simulation file, format YYYYMMDDHH.")

    parser.add_argument('-d', '--date', type=str, nargs=2, required=True,
                        help="Date of the assimilated and reference Pleiade observations"
                             "Format YYYYMMDDHH")

    parser.add_argument('-x', '--xpids', nargs='+', type=str,
                        help="XPID(s) of the simulation(s) format XP_NAME@username")

    parser.add_argument('-a', '--vapp', type=str, default='edelweiss', choices=['s2m', 'edelweiss'],
                        help="Application that produced the target file")

    parser.add_argument('-u', '--uenv', type=str, default="uenv:edelweiss.2@vernaym",
                        help="User environment for static resources (format 'uenv:name@user')")

    parser.add_argument('-w', '--workdir', type=str, default=f'{os.environ["HOME"]}/workdir/EDELWEISS/scores',
                        help='Working directory')

    parser.add_argument('-g', '--geometry', type=str, default='GrandesRousses250m',
                        help='Geometry of the simulation(s) / observation')

    parser.add_argument('-v', '--variable', type=str, default='DSN_T_ISBA',
                        help='Variable of interest (default : SnowDepth)')

    parser.add_argument('-o', '--obs_geometry', type=str, choices=['Lautaret250m', 'Huez250m', 'GrandesRousses250m'],
                        required=True, help='Geometry of the observation')

    parser.add_argument('-m', '--members', action='store_true',
                        help="To activate ensemble simulations")

    args = parser.parse_args()
    return args


def execute():

    # 1. Get all input data

    # a) Pleiades observations
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for idx, date in enumerate(dates_pleiades):
        obsname = f'PLEIADES_{date}.nc'
        io.get_snow_obs_date(xpid=xpid_map[date], geometry=obs_geometry, date=date, vapp='Pleiades', filename=obsname)
        # Open observation file as DataArray
        try:
            obs = xr.open_dataarray(obsname)
            obs = xrp.preprocess(obs, decode_time=False)
        except ValueError:
            obs = xr.open_dataset(obsname)
            obs = xrp.preprocess(obs, decode_time=False, mapping={'Band1': 'HTN', 'DEP': 'HTN', 'DSN_T_ISBA': 'HTN'})
            obs = obs['HTN']

        # c) Simulations
        for xpid in xpids:
            crps = dict()
            pearson = dict()
            bias = dict()
            spread = dict()
            # TODO : gérer ça plus proprement
            if '@' not in xpid:
                user = os.environ["USER"]
                xpid = f'{xpid}@{user}'
            shortid = xpid.split('@')[0]
            product = product_map[shortid]

            # Get (filtered) PRO files with Vortex
            if members:
                member = members_map[shortid]
            else:
                if shortid in ['safran', 'ANTILOPE', 'safran_pappus', 'ANTILOPE_pappus', 'SAFRAN', 'SAFRAN_pappus']:
                    member = None
                else:
                    member = [members_map[shortid][0]]

            # VERRUE pour gérer le décallage d'un jour en attendant de combler les données
            if (shortid.startswith('SAFRAN') or shortid.startswith('ANTILOPE')) and datebegin == '2021080207':
                deb = '2021080106'
            else:
                deb = datebegin  # 2021080207
            kw = dict(datebegin=deb, dateend=dateend, vapp=vapp, member=member, namebuild=None,
                    filename=f'PRO_{shortid}.nc', xpid=xpid, geometry=geometry)
            io.get_pro(**kw)

            # Get simulation without assimilation
            xpid_assim = f'{shortid}_assim'
            member_assim = members_map[xpid_assim]
            io.get_pro(
                datebegin   = deb,
                dateend     = dateend,
                vapp        = vapp,
                member      = member_assim,
                namebuild   = None,
                filename    = f'PRO_{xpid_assim}.nc',
                xpid        = xpid_assim,
                geometry    = geometry,
            )

            # Read data
            openloop = read_simu(xpid, member, date)
            assim = read_simu(xpid_assim, member_assim, date)

            pearson['opl'], crps['opl'], bias['opl'], spread['opl'] = compute_scores(openloop, obs, shortid, date)
            pearson['ass'], crps['ass'], bias['ass'], spread['ass']  = compute_scores(assim, obs, shortid, date)

            if idx == 0:
                linestyle = '-'
                # label     = product
            else:
                linestyle = ':'
                # label     = None

            x0 = bias['opl']
            x1 = bias['ass']
            y0 = spread['opl']
            y1 = spread['ass']
            ax[0].scatter(x0, y0, s=80, facecolors='none', edgecolors=colors_map[product])
            # ax[0].plot(x0, y0, marker='o', color=colors_map[product], markersize=6, fillstyle='none')
            ax[0].annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={'arrowstyle': '-|>', 'lw': 2, 'color': colors_map[product], 'linestyle': linestyle})
            # ax[0].arrow(x, y, dx, dy, color=colors_map[product], linestyle=linestyle, lw=3)

            x0 = pearson['opl']
            # dx = pearson['ass'] - pearson['opl']
            x1 = pearson['ass']
            y0 = crps['opl']
            # dy = crps['ass'] - crps['opl']
            y1 = crps['ass']
            ax[1].scatter(x0, y0, s=80, facecolors='none', edgecolors=colors_map[product])
            # ax[1].plot(x0, y0, marker='o', color=colors_map[product], markersize=6, fillstyle='none')
            ax[1].annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={'arrowstyle': '-|>', 'lw': 2, 'color': colors_map[product], 'linestyle': linestyle})
            # ax[1].arrow(x, y, dx, dy, color=colors_map[product], linestyle=linestyle, label=label, lw=3)

            clean(shortid, members_map[shortid])

    ax[0].set_xlabel('Mean absolute bias')
    ax[0].set_ylabel('Mean spread')
    ax[0].set_xlim(0, 0.7)
    ax[0].set_ylim(0, 0.7)
    ax[0].grid()
    ax[1].set_xlabel('Pearson correlation')
    ax[1].set_ylabel('Mean CRPS')
    ax[1].set_xlim(0.5, 1)
    ax[1].set_ylim(0, 0.6)
    ax[1].grid()
    # plt.legend()
    suffix = '_'.join([product_map[xpid.split('@')[0]] for xpid in xpids])
    fig.savefig(f'synthese_eval_assim_{suffix}.pdf')


def read_simu(xpid, members, date):
    listfiles = list()  # List of simulation PRO files
    shortid = xpid.split('@')[0]
    proname = f'PRO_{shortid}.nc'
    if members is not None and len(members) > 1:
        for mb in members:
            listfiles.append(f'mb{mb:03d}/{proname}')
    else:
        listfiles.append(f'{proname}')

    # Open all simulation PRO files at once
    simu = xr.open_mfdataset(listfiles, concat_dim='member', combine='nested').compute()
    simu = xrp.preprocess(simu, decode_time=False)
    # <xarray.Dataset>
    # Dimensions:     (time: 3, xx: 143, yy: 101, member: 16)
    # Coordinates:
    #   * time        (time) datetime64[ns] 2018-01-23 2018-03-16
    #   * xx          (xx) float64 9.379e+05 9.381e+05 ... 9.731e+05 9.734e+05
    #   * yy          (yy) float64 6.439e+06 6.439e+06 ... 6.464e+06 6.464e+06
    # Dimensions without coordinates: xpid
    # Data variables:
    #     DSN_T_ISBA  (member, time, yy, xx) float64 1.997 1.918 ... 0.0001194 0.2854
    # Get variable's DataArray
    simu = simu.sel({'time': pd.to_datetime(date[:8], format='%Y%m%d')}, method='nearest')
    # <xarray.Dataset>
    # Dimensions:     (xx: 143, yy: 101, member: 16)
    # Coordinates:
    #   time        datetime64[ns] 2018-01-23
    #   * xx        (xx) float64 9.379e+05 9.381e+05 ... 9.731e+05 9.734e+05
    #   * yy        (yy) float64 6.439e+06 6.439e+06 ... 6.464e+06 6.464e+06
    # Dimensions without coordinates: xpid
    # Data variables:
    #     DSN_T_ISBA  (member, time, yy, xx) float64 1.997 1.918 ... 0.0001194 0.2854

    # Set the 'xpid' dimension for simulation identification
    if members is not None:
        simu['member'] = members
    else:
        simu['member'] = [0]
    # <xarray.Dataset>
    # Dimensions:     (xx: 143, yy: 101, member: 16)
    # Coordinates:
    #   time        datetime64[ns] 2018-01-23
    #   * xx        (xx) float64 9.379e+05 9.381e+05 ... 9.731e+05 9.734e+05
    #   * yy        (yy) float64 6.439e+06 6.439e+06 ... 6.464e+06 6.464e+06
    #   * member    (member) int64 1 2 3 ... 16
    # Dimensions without coordinates: xpid
    # Data variables:
    #     DSN_T_ISBA  (member, time, yy, xx) float64 1.997 1.918 ... 0.0001194 0.2854

    return simu


def compute_scores(simu, obs, xpid, date):

    # Select common domains
    obs  = obs.sel({'xx': np.intersect1d(obs.xx, simu.xx), 'yy': np.intersect1d(obs.yy, simu.yy)})
    obs    = obs.where(obs > 0, drop=True)

    simu = simu.sel({'xx': np.intersect1d(obs.xx, simu.xx), 'yy': np.intersect1d(obs.yy, simu.yy)})['DSN_T_ISBA']
    # Mask missing values from the observatin dataset in the simulation dataset
    simu   = simu.where(~np.isnan(obs))
    simu   = simu.where(obs > 0, drop=True)
    mean   = simu.mean(dim='member')
    bias   = xr.apply_ufunc(np.abs, mean - obs).mean()
    spread = simu.std(dim='member').mean()

    # control_member = simu.sel({'member': 0})
    pearson = xr.corr(mean, obs, dim=['xx', 'yy'])

    # simu = simu.expand_dims(dim="time")
    # obs  = obs.expand_dims(dim="time")
    # crps = xskillscore.crps_ensemble(obs, simu, dim='time').mean()
    crps = xskillscore.crps_ensemble(obs, simu).mean()

    return pearson, crps, bias, spread


def clean(xpid, members):
    if members is not None and len(members) > 1:
        for member in members:
            os.remove(f'mb{member:03d}/PRO_{xpid}.nc')
    else:
        os.remove(f'PRO_{xpid}.nc')


def execution_info(workdir):
    print()
    print("===========================================================================================")
    print("                                     Execution result                                      ")
    print("===========================================================================================")
    print()
    print(f"Produced figures are available here : {workdir}")
    print()


if __name__ == '__main__':

    args = parse_command_line()
    datebegin       = args.datebegin
    dateend         = args.dateend
    dates_pleiades  = args.date
    xpids           = args.xpids
    workdir         = args.workdir
    geometry        = args.geometry
    vapp            = args.vapp
    uenv            = args.uenv
    obs_geometry    = args.obs_geometry
    members         = args.members

#    if ':' in args.members:
#        first_mb, last_mb = args.members.split(':')
#        members         = [mb for mb in range(int(first_mb), int(last_mb) + 1)]
#    else:
#        members = None

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    os.chdir(workdir)

    execute()
    execution_info(workdir)
