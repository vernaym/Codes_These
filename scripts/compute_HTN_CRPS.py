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
from snowtools.plots.maps import plot2D
from snowtools.scores import clusters
from snowtools.plots.boxplots import violinplot


members_map = dict(
    RS27_pappus     = [mb for mb in range(17)],
    EnKF_pappus     = [mb for mb in range(17)],
    PF_pappus       = [mb for mb in range(17)],
    ANTILOPE_pappus = None,
    SAFRAN_pappus   = None,
)

label_map = dict(
    elevation   = 'Elevation (m)',
    uncertainty = 'Uncertainty',
)


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

#    parser.add_argument('-m', '--members', type=str, default='0:16',
#            help="Members associated to the experiment. Syntax first:last")

    parser.add_argument('-a', '--vapp', type=str, default='edelweiss', choices=['s2m', 'edelweiss'],
                        help="Application that produced the target file")

    parser.add_argument('-u', '--uenv', type=str, default="uenv:edelweiss.1@vernaym",
                        help="User environment for static resources (format 'uenv:name@user')")

    parser.add_argument('-c', '--clustering', type=str, default='uncertainty', choices=label_map.keys(),
                        help='Define clustering type')

    parser.add_argument('-t', '--thresholds', nargs='+', default=np.arange(2, 30, 6),
                        help='Define bands for clustering (default for uncertainty clustering)')

    parser.add_argument('-w', '--workdir', type=str, default=f'{os.environ["HOME"]}/workdir/EDELWEISS/scores',
                        help='Working directory')

    parser.add_argument('-g', '--geometry', type=str, default='GrandesRousses250m',
                        help='Geometry of the simulation(s) / observation')

    parser.add_argument('-v', '--variable', type=str, default='DSN_T_ISBA',
                        help='Variable of interest (default : SnowDepth)')

    parser.add_argument('-o', '--obs_geometry', type=str, choices=['Lautaret250m', 'Huez250m'], required=True,
                        help='Geometry of the observation')

    args = parser.parse_args()
    return args


def execute(xpid, obs, date, members, mask=True):

    listfiles = list()  # List of simulation PRO files
    shortid = xpid.split('@')[0]
    proname = f'PRO_{shortid}.nc'
    if members is not None:
        for mb in members:
            listfiles.append(f'mb{mb:03d}/{proname}')
    else:
        listfiles.append(f'{proname}')

    # Open all simulation PRO files at once
    simu = xr.open_mfdataset(listfiles, concat_dim='member', combine='nested').compute()
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
    simu = simu.sel({'time': pd.to_datetime(date[:8], format='%Y%m%d')})
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

    # Select common domains
    obs  = obs.sel({'xx': np.intersect1d(obs.xx, simu.xx), 'yy': np.intersect1d(obs.yy, simu.yy)})
    simu = simu.sel({'xx': np.intersect1d(obs.xx, simu.xx), 'yy': np.intersect1d(obs.yy, simu.yy)})
    # Mask missing values from the observatin dataset in the simulation dataset
    simu = simu.where(~np.isnan(obs))
    # <xarray.Dataset>
    # Dimensions:     (member: 16, yy: 79, xx: 63)
    # Coordinates:
    #  * xx          (xx) float64 9.579e+05 9.581e+05 ... 9.731e+05 9.734e+05
    #  * yy          (yy) float64 6.439e+06 6.439e+06 ... 6.458e+06 6.459e+06
    # Dimensions without coordinates: xpid
    # Data variables:
    #    DSN_T_ISBA  (member, yy, xx) float64 nan nan 1.132 1.234 ... 1.918 1.997  nan nan

    # xskillscore.crps_ensemble only allows to compute the mean CRPS along 1 or several dimensions.
    # We want a CRPS for each pixel of the domain, so we add a "fake" dimension to cmpute
    # the CRPS along this dimension and get 1 value per pixel
    simu = simu.expand_dims(dim="time")
    obs  = obs.expand_dims(dim="time")
    crps = xskillscore.crps_ensemble(obs.DSN_T_ISBA, simu.DSN_T_ISBA, dim='time')

    return crps


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
    clustering      = args.clustering
    thresholds      = args.thresholds
#    if ':' in args.members:
#        first_mb, last_mb = args.members.split(':')
#        members         = [mb for mb in range(int(first_mb), int(last_mb) + 1)]
#    else:
#        members = None

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    os.chdir(workdir)

    # 1. Get all input data

    # a) Pleiades observations
    kw = dict(date=datebegin, vapp=vapp)
    obsname = f'PLEIADES_{date}.nc'
    io.get_snow_obs_date(xpid='CesarDB_AngeH', geometry=obs_geometry, date=date, vapp='Pleiades', filename=obsname)
    # Open observation file as DataArray
    obs = xr.open_dataset(obsname)
    obs = xrp.update_varname(obs)

    if clustering == 'elevation':
        # Get Domain's DEM in case ZS not in simulation file
        io.get_const(uenv, 'relief', geometry, filename='TARGET_RELIEF.nc', gvar='RELIEF_GRANDESROUSSES250M_L93')
        mnt = xr.open_dataset('TARGET_RELIEF.nc')  # Target domain's Digital Elevation Model
        mnt = xrp.update_varname(mnt)
        mask = mnt.ZS
    elif clustering == 'uncertainty':
        ds = xr.open_dataset('/home/vernaym/workdir/ASSIMILATION/mask/GrandesRousses/Observation_error_L93.nc')
        ds = xrp.update_varname(ds)
        ds = ds.interp({'xx': obs.xx, 'yy': obs.yy})
        mask = ds.Uncertainty

    # c) Mask
    io.get_const(uenv, 'mask', geometry, filename='MASK.nc')

    dataplot = pd.DataFrame()
    # d) Simulations
    for xpid in xpids:
        # TODO : gérer ça plus proprement
        if '@' not in xpid:
            user = os.environ["USER"]
            xpid = f'{xpid}@{user}'
        shortid = xpid.split('@')[0]

        # Get (filtered) PRO files with Vortex
        members = members_map[shortid]
        kw = dict(datebegin=datebegin, dateend=dateend, vapp=vapp, member=members, namebuild=None,
                filename=f'PRO_{shortid}.nc', xpid=xpid, geometry=geometry)
        io.get_pro(**kw)

        crps = execute(xpid, obs, date, members)
        savename = f'CRPS_{xpid}_{date}.pdf'
        vmin = 0
        vmax = 4
        plot2D.plot_field(crps, savename, vmin=vmin, vmax=vmax, cmap=plt.cm.Reds)

        tmp = clusters.by_slices(crps, mask, thresholds)
        df = tmp.to_dataframe(name=shortid).dropna().reset_index().drop(columns=['xx', 'yy', 'time'], errors='ignore')
        dataplot = pd.concat([dataplot, df])

        # 3. Clean data
        if members is not None:
            for member in members:
                os.remove(f'mb{member:03d}/PRO_{shortid}.nc')
        else:
            os.remove(f'PRO_{shortid}.nc')

    dataplot.columns = dataplot.columns.str.replace('slices', label_map[clustering])
    dataplot = dataplot.melt(label_map[clustering], var_name='experiment', value_name='CRPS (m)')

    title = f'Pleiades, {geometry}, {date[:8]}\n'
    violinplot.plot_ange(dataplot, 'CRPS (m)', figname=f'CRPS_by_{clustering}_{date}_' + '_'.join(xpids),
            title=title, yaxis=label_map[clustering], violinplot=False, xmax=4)
