#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 05/11/2022

import xarray as xr
import argparse

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from snowtools.plots.maps import plot2D
from snowtools.tools.xarray_backend import CENBackendEntrypoint

from vortex import toolbox

toolbox.active_now = True


def parse_command_line():
    description = "Plot 2D field"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-f', '--filename', type=str, nargs='+', required=True,
                        help="Name(s) of the files to plot")

    parser.add_argument('-c', '--cumul', action='store_true',
                        help='Plot accumulation over time')

    parser.add_argument('-m', '--mean', action='store_true',
                        help='Plot ensemble mean')

    parser.add_argument('--vmin', type=float,
                        help='Min colorscale value')

    parser.add_argument('--vmax', type=float,
                        help='Max colorscale value')

    parser.add_argument('--cmap', default='YlGnBu',
                        help='Colormap')

    parser.add_argument('--label', nargs='*', default=None, help='Label of the field to plot')

    parser.add_argument('--coordinates', nargs=4, type=float, default=None,
                        help='list of min/max coordinates to extract, format:'
                        '[latmax, latmin, lonmin, lonmax] (unit: °)')

    args = parser.parse_args()
    return args


args = parse_command_line()
for filename in args.filename:
    field = xr.open_dataarray(filename, engine='cen')
    field = field.rio.write_crs('EPSG:4326')
    figname = filename.replace('.nc', '.pdf')

    if args.cumul and 'time' in field.dims:
        field = field.sum('time')
    if args.mean:
        field = field.mean('member')

    if args.coordinates is not None:
        latmax = args.coordinates[0]
        latmin = args.coordinates[1]
        lonmin = args.coordinates[2]
        lonmax = args.coordinates[3]
        field = field.where((field.xx >= lonmin) & (field.xx <= lonmax) & (field.yy >= latmin) &
                (field.yy <= latmax), drop=True)
        fig, ax = plt.subplots(figsize=(8 * len(field.xx) / len(field.yy), 6),
                subplot_kw=dict(projection=ccrs.PlateCarree()), layout='tight')
        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()), layout='tight')

    field = field.where(~field.isnull(), drop=True)
    if args.label:
        label = ' '.join(args.label)
        field = field.rename(label)

    plot2D.plot_field(field, ax=ax, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, boundaries=True, cities=True,
            massifs=True)
    ax.set_title("")
    plot2D.save_fig(figname, fig=fig)
