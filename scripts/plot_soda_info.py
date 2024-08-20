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
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from snowtools.scripts.extract.vortex import vortexIO as io
from snowtools.scripts.post_processing import common_dict
import snowtools.tools.xarray_preprocess as xrp
from snowtools.plots.maps import plot2D

from vortex import toolbox

toolbox.active_now = True

matplotlib.rcParams.update({'font.size': 18})

product_map = common_dict.product_map
xpid_map    = common_dict.xpid_map
colors_map  = common_dict.colors_map

thresholds = [1500, 2000, 2500, 3000, 3500]


def parse_command_line():
    description = "Plot figures comparing snow depth simulation(s) to a Pleiade observation"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-d', '--date', type=str, required=True,
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

    args = parser.parse_args()
    return args


def execute():

    # High-resolution (25m) DEM for fancy figures (set shade=True in plot_field calls)
    io.get_const('uenv:dem.1@vernaym', 'relief', geometry, filename='TARGET_RELIEF.nc',
            gvar='RELIEF_GRANDESROUSSES250M_L93')
    dem = xr.open_dataset('TARGET_RELIEF.nc')
    dem = xrp.preprocess(dem)

    for xpid in xpids:
        # TODO : gérer ça plus proprement
        if '@' not in xpid:
            user = os.environ["USER"]
            xpid = f'{xpid}@{user}'
        shortid = xpid.split('@')[0]
        product = product_map(shortid)
        filename = f'PART_{shortid}_{date}.txt'

        toolbox.input(
            kind           = 'PART',
            model          = 'soda',
            block          = 'soda',
            namebuild      = 'flat@cen',
            namespace      = 'vortex.multi.fr',
            vapp           = 's2m',
            vconf          = '[geometry:tag]',
            date           = date,
            dateassim      = date,
            geometry       = geometry,
            experiment     = xpid,
            local          = filename,
            fatal          = True,
        )

        df  = pd.read_csv(filename, sep=',', header=None)
        df['sum'] = df.nunique(axis=1)
        median = df[range(17)].apply(lambda x: x.median(), axis=1)
        mean = df[range(17)].apply(lambda x: x.mean(), axis=1)
        # Set pixels with no assimilation to Nan
        median[df['sum'] == 17] = np.nan
        mean[df['sum'] == 17] = np.nan

        out = xr.Dataset(
            data_vars = dict(
                median = (["yy", "xx"], median.values.reshape(len(dem.yy), len(dem.xx))),
                mean   = (["yy", "xx"], mean.values.reshape(len(dem.yy), len(dem.xx))),
            ),
            coords = dict(
                xx = (('xx'), dem.xx.data),
                yy = (('yy'), dem.yy.data),
            ),
            attrs  = dict(description="SODA output"),
        )

        for var in ['mean', 'median']:
            fig, ax  = plt.subplots()
            im = plot2D.plot_field(out[var], ax=ax, cmap=plt.cm.RdBu, dem=dem.ZS, shade=False, vmin=1, vmax=17,
                    isolevels=thresholds, add_colorbar=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            savename = f'SODA_{var}_{date}_{product}.pdf'
            plot2D.save_fig(savename, fig)

        os.remove(filename)

    os.remove('TARGET_RELIEF.nc')


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
    date            = args.date
    xpids           = args.xpids
    workdir         = args.workdir
    geometry        = args.geometry
    vapp            = args.vapp
    uenv            = args.uenv

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
