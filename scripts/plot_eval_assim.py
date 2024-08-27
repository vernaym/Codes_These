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
from matplotlib.patches import Ellipse

# Following installation required on sxcen :
# - Install xskillscore : pip install xskillscore
# - Upgarde numba       : pip install numba --upgrade
import xskillscore  # Requires an installation : pip install xskillscore
# https://xskillscore.readthedocs.io/en/stable/api/xskillscore.crps_ensemble.html

import snowtools.tools.xarray_preprocess as xrp
from snowtools.scripts.extract.vortex import vortexIO as io

from snowtools.scripts.post_processing import common_dict

from These.scripts import tools

matplotlib.rcParams.update({'font.size': 18})

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

    parser.add_argument('-u', '--uenv', type=str, default="uenv:edelweiss.3@vernaym",
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})

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

        # Sort yy coordinate to avoid problems in the histogram computation
        obs = obs.reindex(yy=list(np.sort(obs.yy)))

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
            product = product_map(shortid)

            # Get (filtered) PRO files with Vortex
            if members:
                member = members_map(shortid)
            else:
                if shortid in ['safran', 'ANTILOPE', 'safran_pappus', 'ANTILOPE_pappus', 'SAFRAN', 'SAFRAN_pappus']:
                    member = None
                else:
                    member = [members_map(shortid)[0]]

            # VERRUE pour gérer le décallage d'un jour en attendant de combler les données
            if (shortid.split('_')[0] in ['SAFRAN', 'ANTILOPE', 'KRIGING']) and datebegin == '2021080207':
                deb = '2021080106'
            else:
                deb = datebegin  # 2021080207
            kw = dict(datebegin=deb, dateend=dateend, vapp=vapp, member=member, namebuild=None,
                    filename=f'PRO_{shortid}.nc', xpid=xpid, geometry=geometry)
            io.get_pro(**kw)

            # Get simulation without assimilation
            xpid_assim = f'{shortid}_assim'
            member_assim = members_map(xpid_assim)
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
            pearson['ass'], crps['ass'], bias['ass'], spread['ass']  = compute_scores(assim, obs, xpid_assim, date)

            if idx == 0:
                linestyle = '-'
                label     = product
            else:
                linestyle = ':'
                label     = None

            # Plot Cesar's synthetic scores
            x0 = bias['opl'][0]
            x1 = bias['ass'][0]
            y0 = spread['opl'][0]
            y1 = spread['ass'][0]
            # ax[0].scatter(x0, y0, s=80, facecolors='none', edgecolors=colors_map[product])
            fact = 10
            if product in colors_map.keys():
                color = colors_map[product]
            else:
                color = None
            ellipse1 = Ellipse((x0, y0), width=bias['opl'][1] / fact, height=spread['opl'][1] / fact,
                              facecolor='none', edgecolor=color, linestyle=linestyle,
                              linewidth=2)
            ellipse2 = Ellipse((x1, y1), width=bias['ass'][1] / fact, height=spread['ass'][1] / fact,
                              facecolor='none', edgecolor=color, linestyle=linestyle,
                              linewidth=2)
            ax1.add_patch(ellipse1)
            ax1.add_patch(ellipse2)
            ax1.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={'arrowstyle': '-|>', 'lw': 3, 'color': color,
                        'linestyle': linestyle, 'mutation_scale': 30})

            x0 = pearson['opl'][0]
            x1 = pearson['ass'][0]
            y0 = crps['opl'][0]
            y1 = crps['ass'][0]
            ellipse1 = Ellipse((x0, y0), width=pearson['opl'][1], height=crps['opl'][1] / fact,
                              facecolor='none', edgecolor=color, linestyle=linestyle,
                              linewidth=2)
            ellipse2 = Ellipse((x1, y1), width=pearson['ass'][1], height=crps['ass'][1] / fact,
                              facecolor='none', edgecolor=color, linestyle=linestyle,
                              linewidth=2)
            ax2.add_patch(ellipse1)
            ax2.add_patch(ellipse2)
            ax2.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={'arrowstyle': '-|>', 'lw': 3, 'color': color,
                        'linestyle': linestyle, 'mutation_scale': 30})

            # Add empty plot for legend
            ax4.plot([], [], linestyle=linestyle, color=color, label=label, lw=3)

            if False:
                # Compute/plot rank histograms
                obs_p = obs.where(obs > 0, drop=True)
                sim_p = openloop.DSN_T_ISBA.where(obs > 0, drop=True)
                rh  = xskillscore.rank_histogram(obs_p, sim_p)
                fig2, ax = plt.subplots()
                ax.bar(rh['rank'], rh.data)
                ax.set_ylim(0, 1000)
                fig2.savefig(f'RankHistogram_{shortid}_{date}.pdf')

                sim_p = assim.DSN_T_ISBA.where(obs > 0, drop=True)
                rh = xskillscore.rank_histogram(obs_p, sim_p)
                fig3, ax = plt.subplots()
                ax.bar(rh['rank'], rh.data)
                ax.set_ylim(0, 1000)
                fig3.savefig(f'RankHistogram_{xpid_assim}_{date}.pdf')

            clean(shortid, members_map(shortid))

    ax1.set_xlabel('Mean absolute error of the ensemble mean (m)')
    ax1.set_ylabel('Mean ensemble spread (m)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid()
    ax2.set_xlabel('Pearson correlation coefficient')
    ax2.set_ylabel('Mean CRPS (m)')
    ax2.set_xlim(0.3, 1)
    ax2.set_ylim(0, 0.7)
    ax2.grid()
    ax4.legend(loc='center', frameon=False)
    ax4.axis('off')
    # plt.legend()
    dateassim = f'{dates_pleiades[0][0:4]}-{dates_pleiades[0][4:6]}-{dates_pleiades[0][6:8]}'
    dateeval = f'{dates_pleiades[1][0:4]}-{dates_pleiades[1][4:6]}-{dates_pleiades[1][6:8]}'
    custom_legend(ax3, dateassim, dateeval)
    plt.tight_layout()
    suffix = '_'.join([product_map(xpid.split('@')[0]) for xpid in xpids])
    fig.savefig(f'synthese_eval_assim_{suffix}.pdf')


def custom_legend(axis, dateassim, dateeval):

    x0 = 0.1
    x1 = 0.5
    x2 = 0.9

    axis.text(x0, 0.8, 'No assimilation', horizontalalignment='center', transform=axis.transAxes, weight='bold')
    axis.text(x2, 0.8, 'Assimilation', horizontalalignment='center', transform=axis.transAxes, weight='bold')

    ellipse = list()

    ellipse.append(Ellipse((x0, x1), width=0.16, height=0.15, transform=axis.transAxes,
                      facecolor='none', edgecolor='k', linestyle='-', linewidth=2))
    ellipse.append(Ellipse((x2, x1), width=0.105, height=0.1, transform=axis.transAxes,
                      facecolor='none', edgecolor='k', linestyle='-', linewidth=2))
    ellipse.append(Ellipse((x0, x0), width=0.16, height=0.15, transform=axis.transAxes,
                      facecolor='none', edgecolor='k', linestyle='--', linewidth=2))
    ellipse.append(Ellipse((x2, x0), width=0.105, height=0.1, transform=axis.transAxes,
                      facecolor='none', edgecolor='k', linestyle='--', linewidth=2))

    for item in ellipse:
        axis.add_patch(item)

    axis.text(x1, x1 + 0.1, f"Assimilation date ({dateassim})", horizontalalignment='center')
    axis.annotate("", xy=(x2, x1), xytext=(x0, x1), xycoords='axes fraction',
            arrowprops={'arrowstyle': '-|>', 'lw': 3, 'color': 'k',
                'linestyle': '-', 'mutation_scale': 30})

    axis.text(x1, x0 + 0.1, f"Evaluation date ({dateeval})", horizontalalignment='center')
    axis.annotate("", xy=(x2, x0), xytext=(x0, x0), xycoords='axes fraction',
            arrowprops={'arrowstyle': '-|>', 'lw': 3, 'color': 'k',
                'linestyle': '--', 'mutation_scale': 30})

    axis.axis('off')

#    plt.xlim(0, 4)
#    plt.ylim(0, 2)


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
    obs    = obs.where(~np.isnan(obs), drop=True)
    obs    = obs.where(obs > 0, drop=True)

    simu = simu.sel({'xx': np.intersect1d(obs.xx, simu.xx), 'yy': np.intersect1d(obs.yy, simu.yy)})['DSN_T_ISBA']
    # Mask missing values from the observatin dataset in the simulation dataset
    simu   = simu.where(~np.isnan(obs), drop=True)
    simu   = simu.where(obs > 0, drop=True)
    mean   = simu.mean(dim='member')
    bias   = xr.apply_ufunc(np.abs, mean - obs)
    spread = simu.std(dim='member')

    if True:
        figSK, axSK = plt.subplots()
        x = bias.data.flatten()
        y = spread.data.flatten()
        z = obs.data.flatten()
        x = x[~np.isnan(z)]
        y = y[~np.isnan(z)]
        z = z[~np.isnan(z)]
        sc = tools.plot_scatter(axSK, x, y, lims=[0, 3], color=z)
        axSK.set_xlabel('Absolute error of the ensemble mean (m)')
        axSK.set_ylabel('Ensemble spread (m)')
        cb = figSK.colorbar(sc)
        cb.set_label(label='Snow depth (m)', size=18)
        figSK.savefig(f'SpreadSkill_{xpid}_{date}.pdf')

    bs = [bias.mean().data, bias.std().data]
    sd = [spread.mean().data, spread.std().data]

    # control_member = simu.sel({'member': 0})
    pearson = xr.corr(simu, obs, dim=['xx', 'yy'])
    ps = [pearson.mean().data, pearson.std().data]

    crps = xskillscore.crps_ensemble(obs, simu, dim=[])
    cs = [crps.mean().data, crps.std().data]

    return ps, cs, bs, sd


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
