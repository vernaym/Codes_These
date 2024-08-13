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
import pandas as pd
import xarray as xr
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import snowtools.tools.xarray_preprocess as xrp
from snowtools.scripts.extract.vortex import vortexIO as io
from snowtools.plots.maps import plot2D
from snowtools.scores import clusters

from snowtools.scripts.post_processing import common_dict

members_map = common_dict.members_map
product_map = common_dict.product_map
xpid_map    = common_dict.xpid_map
colors_map  = common_dict.colors_map

# Retrieve dictionnary to map clustering type to a proper label
label_map = clusters.label_map

thresholds = [1500, 2000, 2500, 3000, 3500]

subdomain_map = dict(
    huez = dict(
        lonmin = 942000,
        lonmax = 955000,
        latmin = 6444000,
        latmax = 6460000,
    ),
    lautaret = dict(
        lonmin = 962500,
        lonmax = 980000,
        latmin = 6438000,
        latmax = 6455000,
    )
)

datesassim = ['2022022612', '2018012312']


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

    parser.add_argument('-x', '--xpids', type=str, nargs='+',
                        help="XPID of the simulation without assimilation format XP_NAME@username")

    parser.add_argument('-a', '--xpids_assim', type=str, default=None,
                        help="XPID of the simulation with assimilation format XP_NAME@username")

    parser.add_argument('-v', '--vapp', type=str, default='edelweiss', choices=['s2m', 'edelweiss'],
                        help="Application that produced the target file")

#    parser.add_argument('-u', '--uenv', type=str, default="uenv:edelweiss.2@vernaym",
#                        help="User environment for static resources (format 'uenv:name@user')")

    parser.add_argument('-w', '--workdir', type=str, default=f'{os.environ["HOME"]}/workdir/EDELWEISS/plot/HTN',
                        help='Working directory')

    parser.add_argument('-g', '--geometry', type=str, default='GrandesRousses250m',
                        help='Geometry of the simulation(s) / observation')

    parser.add_argument('-o', '--obs_geometry', type=str, choices=['Lautaret250m', 'Huez250m', 'GrandesRousses250m'],
                        help='Geometry of the observation (default=geometry)', default=None)

    parser.add_argument('-s', '--subdomain', type=str, choices=subdomain_map.keys(),
                        help='Subdomain over which the plot will be made', default='huez')

    parser.add_argument('-m', '--members', action='store_true',
                        help="To activate ensemble simulations")

    args = parser.parse_args()
    return args


def execute():

    args = parse_command_line()
    datebegin       = args.datebegin
    dateend         = args.dateend
    date            = args.date
    xpids           = args.xpids
    xpids_assim     = args.xpids_assim
    if xpids_assim is None:
        xpids_assim = [f'{xp}_assim' for xp in xpids]
    else:
        # TODO : ensure that len(xpids) == 1 in this case
        xpids_assim = [xpids_assim]
    geometry        = args.geometry
    subdomain       = args.subdomain
    vapp            = args.vapp
    if args.obs_geometry is None:
        obs_geometry = geometry
    else:
        obs_geometry    = args.obs_geometry
    members         = args.members

    # 1. Get all input data

    # a) Pleiades observations
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

    # b) DEM
    # io.get_const(uenv, 'relief', geometry, filename='TARGET_RELIEF.nc', gvar='RELIEF_GRANDESROUSSES250M_L93')
    # High-resolution (25m) DEM for fancy figures (set shade=True in plot_field calls)
    io.get_const('uenv:dem.1@vernaym', 'relief', geometry, filename='TARGET_RELIEF.nc',
            gvar='DEM_GRANDESROUSSES25M_L93')

    # Get Domain's DEM in case ZS not in simulation file
    mnt = xr.open_dataset('TARGET_RELIEF.nc')  # Target domain's Digital Elevation Model
    mnt = xrp.preprocess(mnt, decode_time=False)
    mnt = mnt['ZS']

    for idx, xpid in enumerate(xpids):

        # TODO : gérer ça plus proprement
        if '@' not in xpid:
            user = os.environ["USER"]
            xpid = f'{xpid}@{user}'
        shortid = xpid.split('@')[0]

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

        # Get simulation without assimilation
        kw = dict(datebegin=deb, dateend=dateend, vapp=vapp, member=member, namebuild=None,
                filename=f'PRO_{shortid}.nc', xpid=xpid, geometry=geometry)
        io.get_pro(**kw)

        # Get simulation without assimilation
        xpid_assim = xpids_assim[idx]
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

        plot_htn(openloop, obs, assim, shortid, xpid_assim, date, subdomain, dem=mnt)

        clean(shortid, member)
        clean(xpid_assim, member_assim)


def read_simu(xpid, members, date):
    listfiles = list()  # List of simulation PRO files
    shortid = xpid.split('@')[0]
    proname = f'PRO_{shortid}.nc'
    print(f'Read simu {xpid}: {proname}')
    print(members)
    if members is not None and len(members) > 1:
        for mb in members:
            listfiles.append(f'mb{mb:03d}/{proname}')
    else:
        listfiles.append(f'{proname}')
    print(listfiles)

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


def plot_htn(openloop, obs, assim, xpid, xpid_assim, date, subdomain, dem=None):
    savename = f'HTN_{xpid}_{xpid_assim}_{date}.pdf'

    lonmin = subdomain_map[subdomain]['lonmin']
    lonmax = subdomain_map[subdomain]['lonmax']
    latmin = subdomain_map[subdomain]['latmin']
    latmax = subdomain_map[subdomain]['latmax']
    openloop = openloop.where((openloop.xx > lonmin) & (openloop.xx < lonmax) & (openloop.yy < latmax) &
            (openloop.yy > latmin), drop=True)
    obs      = obs.where((obs.xx > lonmin) & (obs.xx < lonmax) & (obs.yy < latmax) & (obs.yy > latmin), drop=True)
    assim    = assim.where((assim.xx > lonmin) & (assim.xx < lonmax) & (assim.yy < latmax) & (assim.yy > latmin),
            drop=True)

    # obshtn = var_obshtn[date]
    vmin = 0
    # Round max to nearest 0.5m
    # vmax = round(float(max(openloop.DSN_T_ISBA.max(), obs.max(), assim.DSN_T_ISBA.max())) * 2) / 2
    vmax = round(float(max(openloop.DSN_T_ISBA.max(), obs.max(), assim.DSN_T_ISBA.max())))
    slices = int(vmax)

    # fig, ax = plt.subplots(1, 3, figsize=(36 * len(obs.xx) / len(obs.yy), 10), sharey=True)
    fig = plt.figure(figsize=(30 * len(obs.xx) / len(obs.yy), 10))
    ax = ImageGrid(
        fig, 111,          # as in plt.subplot(111)
        nrows_ncols   = (1, 3),
        axes_pad      = 0.15,
        share_all     = True,
        cbar_location = "right",
        cbar_mode     = "single",
        cbar_size     = "7%",
        cbar_pad      = 0.15,
    )

    opl = openloop.DSN_T_ISBA.mean(dim='member')
    print('plot openloop')
    plot2D.plot_field(opl, ax=ax[0], vmin=vmin, vmax=vmax, cmap=plt.cm.Blues, dem=dem, shade=False,
            isolevels=thresholds, slices=slices, add_colorbar=False)
    ax[0].set_title('Openloop mean snow depth (m)')

    print('plot observation')
    plot2D.plot_field(obs, ax=ax[1], vmin=vmin, vmax=vmax, cmap=plt.cm.Blues, dem=dem, shade=False,
            isolevels=thresholds, slices=slices, add_colorbar=False)
    #        shade=True,)
    if date in datesassim:
        ax[1].set_title('Assimilated snow depth (m)')
    else:
        ax[1].set_title('Observed snow depth (m)')

    print('plot assimilation')
    ass = assim.DSN_T_ISBA.mean(dim='member')
    im = plot2D.plot_field(ass, ax=ax[2], cmap=plt.cm.Blues, dem=dem, shade=False, vmin=vmin, vmax=vmax,
            isolevels=thresholds, slices=slices, add_colorbar=False)
    ax[2].set_title('Assimilation mean snow depth (m)')

    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel('')
        axis.set_ylabel('')

    ax[2].cax.colorbar(im, label='Snow depth(m)')

    plot2D.save_fig(savename, fig)


def clean(suffix, members):
    if members is not None and len(members) > 1:
        for member in members:
            os.remove(f'mb{member:03d}/PRO_{suffix}.nc')
    else:
        os.remove(f'PRO_{suffix}.nc')


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
    workdir         = args.workdir

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    os.chdir(workdir)

    execute()
    execution_info(workdir)
