#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022
 
import os
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import numpy as np
import argparse

import matplotlib.pyplot as plt
from snowtools.plots.maps import cartopy

from pykrige.uk import UniversalKriging

import snowtools.scripts.extract.vortex.vortexIO as io

variogram  = 'exponential'  # The same as for ANTILOPE without RADAR data

# Liste des coordonnées attendues par la commande dap3: lat_max, lat_min, lon_max, lon_min
coords = dict(
    alp  = ['46875', '43125', '4500', '8500'],
    pyr  = ['43500', '42000', '-2000', '3500'],
    cor  = ['43000', '41000', '8000', '10500'],
    ange = ['45240', '44990', '6010', '6490'],
    #GrandesRousses = ['45240', '44990', '6010', '6490'],
    GrandesRousses = ['45440', '44790', '5810', '6690'],  # With 0.2 margin
)
domain = 'GrandesRousses'
ymax, ymin, xmin, xmax = coords[domain]

gridx = [x / 1000. for x in np.arange(int(xmin), int(xmax) + 10, 10)]
gridy = [y / 1000. for y in np.arange(int(ymin), int(ymax) + 10, 10)]

def parse_command_line():
    description = "Evaluation of RADAR products (ANTILOPE or PANTHERE) using nivo-météo network observations"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-p', '--plot', help = 'Plot krieged field for each date', default=False, action='store_true')
    parser.add_argument('-d', '--data', help = 'Data to use for kriging', default='pluvios_antilope', choices=['nivometeo', 'pluvios_antilope', 'all'])

    args = parser.parse_args()

    args.datebegin = get_date(args.datebegin)
    if args.dateend:
        args.dateend = get_date(args.dateend)
    else:
        args.dateend = args.datebegin + timedelta(hours=24)

    return args

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

def read_nivometeo_coords(domain):
    if os.path.exists('postes_nivometeo.csv'):
        metadata = pd.read_csv('postes_nivometeo.csv', sep=';')
        latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
        subdata = metadata[(metadata['poste_nivo.lat_dg']>=latmin) & (metadata['poste_nivo.lat_dg']<=latmax) & (metadata['poste_nivo.lon_dg']>=lonmin) & (metadata['poste_nivo.lon_dg']<=lonmax)]
        return dict(zip(np.array(subdata['poste_nivo.num_poste']), zip(np.array(subdata['poste_nivo.lat_dg']), np.array(subdata['poste_nivo.lon_dg']))))
    else:
        return None


def read_ref_coords(domain):
    metadata = pd.read_csv('obs_quotidiennes_RR.data', sep=';')
    latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
    subdata = metadata[(metadata['lat']>=latmin) & (metadata['lat']<=latmax) & (metadata['lon']>=lonmin) & (metadata['lon']<=lonmax)]
    return dict(zip(np.array(subdata['num_poste']), zip(np.array(subdata['lat']), np.array(subdata['lon']))))


if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend, dt=1)

    if args.data == 'nivometeo':
        reference = read_ref_coords(domain)
        fic = 'obs_nivometeo.csv'
        pluvios = pd.read_csv(fic, sep=';', parse_dates=['H.dat'], dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.massif_nivo':int,
            'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'poste_nivo.alti':int, 'rr':float, 'hist_reseau_poste.reseau_poste':int}, na_values=['--'])
        pluvios.rename(columns={'H.dat':'dat', 'H.num_poste':'num_poste', 'poste_nivo.lat_dg':'lat', 'poste_nivo.lon_dg':'lon', 'H.rr1':'rr',
            'poste_nivo.alti':'alti', 'hist_reseau_poste.reseau_poste':'reseau_poste'}, inplace=True)
    elif args.data == 'all':
        reference = dict()
        fic1 = 'obs_nivometeo.csv'
        pluvios1 = pd.read_csv(fic1, sep=';', parse_dates=['Q.dat'], dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.massif_nivo':int,
            'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'poste_nivo.alti':int, 'rr':float, 'hist_reseau_poste.reseau_poste':int}, na_values=['--'])
        pluvios1.rename(columns={'H.dat':'dat', 'H.num_poste':'num_poste', 'poste_nivo.lat_dg':'lat', 'poste_nivo.lon_dg':'lon', 'H.rr1':'rr',
            'poste_nivo.alti':'alti', 'hist_reseau_poste.reseau_poste':'reseau_poste'}, inplace=True)
        fic2 = "autres_obs.csv"
        pluvios2 = pd.read_csv(fic2, sep=';', parse_dates=['dat'], dtype={'num_poste':int, 'poste':str, 'lat':float, 'lon':float, 'alti':int, 'rr':float, 'reseau_poste':int}, na_values=['--'])
        pluvios = pd.concat([pluvios1, pluvios2], ignore_index=True, sort=True)
    else:
        reference = read_nivometeo_coords(domain)
        fic = "obs_quotidiennes_RR.data"
        pluvios = pd.read_csv(fic, sep=';', parse_dates=['dat'], dtype={'num_poste':int, 'poste':str, 'lat':float, 'lon':float, 'alti':int, 'rr':float, 'reseau_poste':int}, na_values=['--'])

    if reference is not None:
        pluvios = pluvios[~pluvios['num_poste'].isin(reference.keys())]  # sécurité pour assurer que le krigeage n'utilise pas d'obs d'évaluation
    pluvios = pluvios.loc[~pluvios['rr'].isna()]
    #outkrig = pd.DataFrame(columns=['date', 'num_poste', 'rr_kriging'])
    outkrig = pd.DataFrame()
    lon, lat = np.meshgrid(gridx, gridy)
    cumul = xr.DataArray(
        data   = np.zeros(shape=(len(gridy), len(gridx))),
        name   = 'rr_cumul',
        dims   =["lat", "lon"],
        coords =dict(lon=gridx, lat=gridy),
        attrs  =dict(description="Total precipitation", units="mm",),
    )
    out = xr.DataArray(
        data   = np.zeros(shape=(len(gridy), len(gridx), len(extract_period))),
        name   = 'rr',
        dims   = ["lat", "lon", "time"],
        coords = dict(lat=gridy, lon=gridx, time=extract_period),
        attrs  = dict(description="Precipitation", units="mm",),
    )
    for idx, rundate in enumerate(extract_period):
        print(rundate)
        startdate = rundate - timedelta(hours=24)
        # Extract hourly precipitation of the last 23-hours
        tmp  = pluvios.loc[pluvios['dat']<=rundate].loc[pluvios['dat']>startdate]
        nval = tmp.groupby(['num_poste']).dat.count()
        y    = tmp.groupby(['num_poste']).lat.mean()[nval==24]
        x    = tmp.groupby(['num_poste']).lon.mean()[nval==24]
        rr   = tmp.groupby(['num_poste']).rr.sum()[nval==24]
        alti = tmp.groupby(['num_poste']).alti.mean()[nval==24]

        if np.max(rr) > 0:
            #kriging = OrdinaryKriging(x.values, y.values, rr.values, variogram_model='linear')
            drift = [x, y, alti]
            #kriging = UniversalKriging(x.values, y.values, rr.values, variogram_model=variogram, drift_terms=drift_terms, point_drift=drift)
            kriging = UniversalKriging(x.values, y.values, rr.values, variogram_model=variogram)
            rr24, ss = kriging.execute('grid', gridx, gridy)
        else:
            print('No precipitation over the domain')
            rr24 = np.zeros(shape=(len(gridy), len(gridx)))


#                          I-   Plot field after kriging
#======================================================================================
        if args.plot:
            # plt.imshow(rr24, interpolation='none', cmap='jet')
            # plt.show()
            fig = cartopy.Map_alpes()
            #cf = plt.contourf(lon, lat, rr24.data, levels=100, cmap='YlGnBu')
            cf = plt.contourf(lon, lat, rr24.data, cmap='YlGnBu')
            plt.colorbar(cf, label='24h precipitation (mm)')
            fig.init_massifs()
            fig.addpoints(x.values, y.values, labels=np.around(rr, 1))
            #fig.set_figtitle('Universal Kriging for date {0:s} \n Variogram model = {1:s}, drift-terms = {2:s}'.format(rundate.strftime('%Y%m%d%H'), variogram, ','.join(drift_terms)))
            fig.set_figtitle('Universal Kriging for date {0:s}. \n Variogram model : {1:s}'.format(rundate.strftime('%Y%m%d%H'), variogram))
            plt.tight_layout()
            fig.save('map_{0:s}.svg'.format(rundate.strftime('%Y%m%d%H')), formatout='svg')
            fig.close()

#                   II-  Get kriged values on the reference stations
#======================================================================================
        if args.data == 'all':
            cumul.data = cumul.data + rr24.data
        else:
            if reference is not None:
                for num_poste, (lat, lon) in reference.items():
                    idx = np.argmin(np.abs(gridx-lon))
                    idy = np.argmin(np.abs(gridy-lat))
                    outkrig = outkrig.append({
                        'date': rundate,
                        'num_poste': int(num_poste),
                        'rr_kriging': rr24[idy][idx]
                    }, ignore_index=True)
            out.data[:, :, idx] = rr24.data

    datebegin = args.datebegin.strftime('%Y%m%d%H')
    dateend   = args.dateend.strftime('%Y%m%d%H')

    if args.data == 'all':
        outname = f"CUMUL_krigeage_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}.nc"
        cumul.to_netcdf(outname, mode='w')
    else:
        outname = f"KRIGING_{datebegin}_{dateend}.nc"
        out.to_netcdf(outname)
        if reference is not None:
            outkrig.set_index('date')
            outname = 'Kriging_{0:s}_{1:s}_{2:s}.csv'.format(args.data, args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
            outkrig.to_csv(outname, index=False, sep=';')

    io.put_meteo(
      kind         = 'Precipitation',
      geometry     = 'GrandesRousses1km',
      xpid         = 'Kriging@vernaym',
      vapp         = 'edelweiss',
      datebegin    = datebegin,
      dateend      = dateend,
      block        = 'hourly',
      filename     = outname,
    )
