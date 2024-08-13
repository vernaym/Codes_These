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
import matplotlib
import matplotlib.pyplot as plt

# Installation required on sxcen : pip install gstools
# https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/03_variogram/00_fit_variogram.html
# import gstools as gs  # Very slow and does not work

# Installation required on sxcen : pip install scikit-gstat
# https://gatorglaciology.github.io/gstatsimbook/2_Variogram_model.html
import skgstat as skg
from skgstat import models

import snowtools.tools.xarray_preprocess as xrp
from snowtools.scripts.extract.vortex import vortexIO as io
from snowtools.scores import clusters

from snowtools.scripts.post_processing import common_dict

matplotlib.rcParams.update({'font.size': 18})

members_map = common_dict.members_map
product_map = common_dict.product_map
xpid_map    = common_dict.xpid_map
colors_map  = common_dict.colors_map

# Retrieve dictionnary to map clustering type to a proper label
label_map = clusters.label_map


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

    parser.add_argument('-u', '--uenv', type=str, default="uenv:edelweiss.2@vernaym",
                        help="User environment for static resources (format 'uenv:name@user')")

    parser.add_argument('-w', '--workdir', type=str, default=f'{os.environ["HOME"]}/workdir/EDELWEISS/scores',
                        help='Working directory')

    parser.add_argument('-g', '--geometry', type=str, default='GrandesRousses250m',
                        help='Geometry of the simulation(s) / observation')

    parser.add_argument('-o', '--obs_geometry', type=str, choices=['Huez250m', 'GrandesRousses250m'],
                        required=True, help='Geometry of the observation')

    parser.add_argument('-m', '--members', action='store_true',
                        help="To activate ensemble simulations")

    args = parser.parse_args()
    return args


def variogram(array, color, ax, var='HTN', label=None, model='spherical', linestyle='-', samples=None):

    # WARNING : maxlag should not be larger than half of the domain's dimension
    maxlag = 10

#    t1 = time.time()

    # Convert coordinates into distances
    array['xx'] = np.array([0.25 * idx for idx in range(len(array.xx))])
    array['yy'] = np.array([0.25 * idy for idy in range(len(array.yy))])
    df = array.to_dataframe().reset_index().dropna()
    # compute variogram
    # https://scikit-gstat.readthedocs.io/en/latest/reference/variogram.html
    # Sampling only 20% of the data to compute / fit the variogram gives results very similar
    # to the one obtain with all data and is much more efficient :
    # Computing variogram with only 20.0 % of points took 2.2013840675354004 s
    # Computing variogram with all points took 9.015846490859985 s
    # Fitting variogram with 20.0 % of the data took 2.8023908138275146 s
    # Fitting variogram with all data took 76.80178022384644 s

    V = skg.Variogram(df[['xx', 'yy']].values, df[var].values, use_nugget=True, normalisze=False, maxlag=maxlag,
        samples=samples,)  # Randomly draw a fraction (if *samples* < 1) of the points to reduce computational cost

    # extract variogram values
    xdata = V.bins
    ydata = V.experimental
    # color = next(ax._get_lines.prop_cycler)['color']
    ax.scatter(xdata, ydata, marker='+', s=40, color=color)

#    t2 = time.time()
#    print(f'Computing variogram with only {sample*100}% of points took {(t2-t1)} s')
#    V2 = skg.Variogram(df[['xx', 'yy']].values, df[var].values, use_nugget=True, normalisze=False, maxlag=maxlag)
#    # extract variogram values
#    xdata = V2.bins
#    ydata = V2.experimental
#    # color = next(ax._get_lines.prop_cycler)['color']
#    ax.scatter(xdata, ydata, s=12, color='red')
#    t3 = time.time()
#    print(f'Computing variogram with all points took {(t3-t2)} s')

    # Fit shperical Variogram
    V.model = model
    # V.parameters[0] = Range
    # V.parameters[1] = Sill
    # V.parameters[2] = Nugget
    x = np.linspace(0, maxlag, 20)
    y = [models.spherical(h, V.parameters[0], V.parameters[1], V.parameters[2]) for h in x]
    ax.plot(x, y, linestyle=linestyle, label=label, color=color, lw=4)
    ax.vlines(V.parameters[0], 0, V.parameters[1] + V.parameters[2], linestyle=':', color=color, lw=3)

#    t4 = time.time()
#    print(f'Fitting variogram with {sample*100}% of the data took {(t4-t3)} s')
#    V2.model = model
#    x = np.linspace(0, maxlag, 100)
#    y = [models.spherical(h, V2.parameters[0], V2.parameters[1], V2.parameters[2]) for h in x]
#    ax.plot(x, y, '-', label='With all data', color='red')
#    t5 = time.time()
#    print(f'Fitting variogram with all data took {(t5-t4)} s')


def execute():

    # 1. Get all input data

    # a) Pleiades observations
    obsname = f'PLEIADES_{date}.nc'
    io.get_snow_obs_date(xpid=xpid_map[date], geometry=obs_geometry, date=date, vapp='Pleiades', filename=obsname)
    # Open observation file as DataArray
    try:
        obs = xr.open_dataarray(obsname)
        obs = xrp.preprocess(obs, decode_time=False)
        obs = obs.rename('HTN')
    except ValueError:
        obs = xr.open_dataset(obsname)
        obs = xrp.preprocess(obs, decode_time=False, mapping={'Band1': 'HTN', 'DEP': 'HTN', 'DSN_T_ISBA': 'HTN'})
        obs = obs['HTN']

    fig, ax = plt.subplots(figsize=(10, 8))
    color = 'k'
    tmp = obs.copy()
    tmp = tmp.where(obs > 0, drop=True)
    # Label dots
    ax.scatter([], [], marker='+', s=40, color='k', label='Experimental variogram')
    # variogram(tmp, color, ax, label='Pleiades', samples=0.2)  # Computing optimisation
    variogram(tmp, color, ax, label='Pleiades')

    # c) Simulations
    for xpid in xpids:
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
        if (shortid.split('_')[0] in ['SAFRAN', 'ANTILOPE', 'KRIGING']) and datebegin == '2021080207':
            deb = '2021080106'
        else:
            deb = datebegin  # 2021080207
        kw = dict(datebegin=deb, dateend=dateend, vapp=vapp, member=member, namebuild=None,
                filename=f'PRO_{shortid}.nc', xpid=xpid, geometry=geometry)
        io.get_pro(**kw)

        simu = read_simu(xpid, member, date)

        color = colors_map[product]
        if 'assim' in product:
            linestyle = '--'
        else:
            linestyle = '-'

        tmp = simu.mean('member')
        tmp = tmp.where(obs > 0, drop=True)

        # compute variogram
        # variogram(tmp, color, ax, var='DSN_T_ISBA', label=product_map[shortid], linestyle=linestyle, samples=0.2)
        variogram(tmp, color, ax, var='DSN_T_ISBA', label=product_map[shortid], linestyle=linestyle)

        clean(shortid, member)

    # plt.title('Isotropoic Experimental Variogram')
    plt.xlabel('Distance (km)')
    plt.ylabel('Semi-variance (m²)')
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.grid(linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.legend()

    suffix = '_'.join([product_map[xpid.split('@')[0]] for xpid in xpids])
    plt.savefig(f'variogram_{suffix}_{date}.pdf')


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
    # Get variable's DataArray
    simu = simu.sel({'time': pd.to_datetime(date[:8], format='%Y%m%d')}, method='nearest')

    # Set the 'xpid' dimension for simulation identification
    if members is not None:
        simu['member'] = members
    else:
        simu['member'] = [0]

    return simu


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
    date            = args.date
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
