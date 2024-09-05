#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 25/01/2022

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr

import argparse

import epygram
from bronx.stdtypes.date import Date
#from snowtools.scripts.extract.vortex import vortexIO as io
from snowtools.scripts.extract.vortex import vortex_get as io

##############################################################################################
# Ce script sert à extraire les données ANTILOPE de la BDAP correspondant aux postes nivometeo
# dans le but d'évaluer le biais d'ANTILOPE avec l'altitude.
# A priori on ne peut extraire les données de la BDAP que sous forme de grille, on fait donc
# une extraction pour chaque sous domaine d'intéret (alp, pyr, cor) avant de selectionner les
# pixels correspondant aux coordonnées des points des postes nivometeo.
# TODO : vérifier si ce n'est pas plus rapide d'extraire un grib complet (peu probable).
##############################################################################################

datadir = os.path.join(os.environ['HOME'], 'These', 'DATA')

# Identifiant du modèle dans la BDAP
#model_id = 'ANTILOPEQJP1'
#model_id = 'ANTILOPEQ'
# Identifiant de la grille
#grid = 'FRANXL1S100'
# Indentifiant du paramètre
#parameter = 'PRECIP'
# Identifiant du niveau
#level = 'SOL'
# Identifiant du reseau
#leadtime = 6
# Echeance
ech = 24
# Format
#fmt = 'GRIB2_C_MAX'

# Liste des coordonnées attendues par la commande dap3: lat_max, lat_min, lon_max, lon_min
coords = dict(
    #alp = ['46875', '43125', '4500', '8500'],
    #alp = ['46900', '43500', '4900', '7800'],
    alp = ['46900', '43000', '4500', '8000'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '41000', '8000', '10500'],
    #GrandesRousses = ['45250', '44750', '6000', '6500'],
    #GrandesRousses = ['45640', '44590', '5610', '7100'],  # With 0.2 margin (final domain : ['45440', '44790', '5810', '6690'],)
    GrandesRousses = ['45440', '44790', '5810', '6690'],  # With 0.2 margin (final domain : ['45240', '44990', '6010', '6490'])
    #ange = ['45240', '44990', '6010', '6490']
    ange = ['45440', '44790', '5810', '6690'],  # With 0.2 margin (final domain : ['45240', '44990', '6010', '6490'])
    HauteSavoie = ['46490', '45300', '5620', '7250'],  # WARNING : includes margin
)

geometry_map = dict(
    alp            = 'Alp1km',
    GrandesRousses = 'GrandesRousses1km',
    pyr            = 'Pyr1km',
)

# Pas en lat/lon de la grille
dl = ['10', '10']

def parse_command_line():
    description = "BDAP extraction of ANTILOPE data."
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-d', '--rundate', help='Rundate for operational executions, format YYMMDDHH')
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-d', '--domain', nargs='+', help='Domain of the file', choices=coords.keys(), default=['alp', 'pyr', 'cor', 'GrandesRousses'])
    parser.add_argument('-w', '--workdir', help='Runing directory (default for sotrtm35-sidev)', default='/home/mrns/vernaym/workdir')
#    parser.add_argument('-o', '--output', help='Output name of generated files')
    parser.add_argument('-m', '--model', help='Model from which the data must be extracted',
            choices=['ANTILOPEQ', 'ANTILOPEJP1Q', 'ANTILOPEH', 'ANTILOPEJP1H'], default='ANTILOPEJP1Q')
    # ANTILOPEQ disponible depuis le 2006070206 (grille FRAN0012)
    parser.add_argument('-g', '--grid', help='BDAP grid name from which to extract data', default='FRANXL1S100', choices=['FRAN0012', 'FRANXL1S100'])
    # La grille FRAN0012 est disponible depuis le 02/07/2006
    # La grille ANTILOPE FRANXL1S100 est disponible depuis le 23/11/2017
    parser.add_argument('-p', '--parameter', help='Parameter to extract', default='PRECIP')
    parser.add_argument('-l', '--level', help='Level to extract', default='SOL')
    parser.add_argument('-v', '--vortex', action='store_true', help='Store generated files on a vortex archive store')
#    parser.add_argument('-r', '--replace', action = 'store_true', help='Replace existing files')
#    parser.add_argument('-t', '--tar', action='store_true', help='Tar generated files (for reanalaysis applications')

    args = parser.parse_args()

    args.datebegin = get_date(args.datebegin)
    if args.dateend:
        args.dateend = get_date(args.dateend)
    else:
        args.dateend = args.datebegin

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
        date = datetime.strptime(a_string, '%Y%m%d%H').replace(minute=0, second=0, microsecond=0)
    except ValueError:
        try:
            date = datetime.strptime(a_string, '%y%m%d%H').replace(minute=0, second=0, microsecond=0)
        except ValueError:
            try:
                date = datetime.strptime(a_string, '%Y%m%d%H%M').replace(second=0, microsecond=0)
            except ValueError:
                try:
                    date = datetime.strptime(a_string, '%Y%m%d').replace(hour=6, minute=0, second=0, microsecond=0)
                except ValueError:
                    try:
                        date = datetime.strptime(a_string, '%y%m%d').replace(hour=6, minute=0, second=0, microsecond=0)
                    except ValueError:
                        print('The start date provided is not in a good format (YYYYMMDDHH or YYMMDDHH or YYYYMMDDHHMM or YYMMDD or YYYYMMDD)')
                        raise
    finally:
        return date

def date_range(start, end, dt):
    start = start
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
    metadata = pd.read_csv(os.path.join(datadir, 'postes_nivometeo.csv'), sep=';')
    latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
    subdata = metadata[(metadata['poste_nivo.lat_dg']>=latmin) & (metadata['poste_nivo.lat_dg']<=latmax) & (metadata['poste_nivo.lon_dg']>=lonmin) & (metadata['poste_nivo.lon_dg']<=lonmax)]
    return dict(zip(np.array(subdata['poste_nivo.num_poste']), zip(np.array(subdata['poste_nivo.lat_dg']), np.array(subdata['poste_nivo.lon_dg']))))

def read_obs_clim():
    #clim_data = 'obs_clim_daily.csv'
    clim = pd.read_csv(os.path.join(datadir, "obs_quotidienne_clim_RR.data"), sep=';', parse_dates=['date'], header=0,
            names = ['num_poste', 'lat', 'lon', 'elevation', 'nom', 'reseau_poste', 'date', 'rr_ref'],
            usecols=['num_poste', 'lat', 'lon', 'elevation', 'nom', 'date', 'rr_ref'],
            dtype={'num_poste':int, 'nom':str, 'rr_ref':int, 'lat':float, 'lon':float, 'rr_ref':float},
            )
    lats          = clim.groupby(['num_poste']).lat.mean()
    lons          = clim.groupby(['num_poste']).lon.mean()
    num_poste     = clim.groupby(['num_poste']).num_poste.mean()

    return dict(zip(np.array(num_poste), zip(np.array(lats), np.array(lons))))

def nearest(array, value):
    """ Find the closest element of 'array' to 'value'. """
    return float(array[np.abs(array - value).argmin()].data)

class ExtractGrib(object):

    def __init__(self, model, grid, domain, rundate):

        self.model          = model
        self.domain         = domain
        self.grid           = grid.upper()
        self.coords         = coords[domain]
        self.date           = rundate
        self.gribname       = '{0:s}_{1:s}.grib'.format(self.model, self.date.strftime('%Y%m%d%H'))

    def requete(self, parameter, level):

        self.rqst = 'requete'.format(level)
        f = open(self.rqst, "w")
        f.write('#RQST\n')
        f.write('#NFIC {0:s}\n'.format(self.gribname))
        f.write('#MOD {0:s}\n'.format(self.model))
        f.write('#PARAM {0:s}\n'.format(parameter))
        f.write('#Z_REF {0:s}\n'.format(self.grid))

        if coords is not None:
            f.write("#Z_EXTR INTERPOLATION\n")
            f.write('#Z_GEO ' + ' '.join(coords[self.domain]) + '\n')
            f.write('#Z_STP ' + ' '.join(dl) + '\n')
        f.write('#L_TYP {0:s}\n'.format(level))
        return True

    def extract(self, parameter, level, ech, cmd='dap3_dev'):
        self.requete(parameter, level)
        startdate = self.date - timedelta(hours=ech)  # File named "*ymdh" must contains the cumul since ym(h-ech)
        os.environ["DMT_DATE_PIVOT"] = startdate.strftime('%Y%m%d%H%M%S')
        print(os.environ["DMT_DATE_PIVOT"])
        os.system("{0:s} {1:d} {2:s}".format(cmd, ech, self.rqst))

    def run(self, parameter, level, ech):
        if os.path.exists(self.gribname):
            print('File {0:s} already exists'.format(self.gribname))
            return True
        else:
            self.extract(parameter, level, ech)
            if os.path.isfile(self.gribname):
                if os.stat(self.gribname).st_size > 0:
                    return True
                else:
                    os.remove(self.gribname)
                    self.grid = 'FRAN0012'
                    self.extract(parameter, level, ech)
                    if os.path.isfile(self.gribname):
                        if os.stat(self.gribname).st_size > 0:
                            return True
                        else:
                            os.remove(self.gribname)
                            return False
            else:
                return False


if __name__ == "__main__":
    args = parse_command_line()
    if args.model in ['ANTILOPEQ', 'ANTILOPEJP1Q']:
        dt = 24
        block = 'daily'
    elif args.model in ['ANTILOPEH', 'ANTILOPEJP1H']:
        dt = 1
        block = 'hourly'
    extract_period = date_range(args.datebegin, args.dateend, dt)
    workdir = os.getcwd()
    for domain in args.domain:
        print(domain)
        goto(workdir)
        nivometeo = read_nivometeo_coords(domain)
        #nivometeo = read_obs_clim()
        missing_grib = list()
        #workdir = os.path.join(args.workdir, domain)
        goto(domain)
        cumul = None
        reference_time = pd.Timestamp(args.datebegin)
        rr24 = None
        nan  = None
        #filename = '{0:s}_{1:s}_{2:s}_{3:s}.nc'.format(args.model, args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), domain)
        filename = f'{args.model}_{args.datebegin.strftime("%Y%m%d%H")}_{args.dateend.strftime("%Y%m%d%H")}_{domain}.nc'

        # Extract data from BDAP
        if not os.path.exists(filename):
            listgrib = list()
            for date in extract_period:
                print(date.strftime('%Y%m%d%H'))
                #if date.month in [1,2,3,4,11,12]:  # Consider only month with nivometeo observations
                print('{0:s}_{1:s}.grib'.format(args.model, date.strftime('%Y%m%d%H')))
                if not os.path.exists('{0:s}_{1:s}.grib'.format(args.model, date.strftime('%Y%m%d%H'))):
                    print(os.getcwd())
                    print(f'File {args.model}_{date.strftime("%Y%m%d%H")}.grib does not exist')
                    grib = ExtractGrib(args.model, args.grid, domain, date)
                    result = grib.run(args.parameter, args.level, dt)
                    gribname = grib.gribname
                else:
                    result = True
                    gribname = '{0:s}_{1:s}.grib'.format(args.model, date.strftime('%Y%m%d%H'))

                if result :
                    listgrib.append(gribname)
                else:
                    print('Missing date {0:s}'.format(date.strftime("%Y%m%d%H")))

            dates = [np.datetime64(date) for date in extract_period]
            # Read data and create NetCDF file
            ds  = xr.open_mfdataset(listgrib, concat_dim='valid_time', combine='nested', engine='cfgrib')
            ds = ds.drop('time').rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon', 'PRECIP': 'rr'})
            # Fill missing dates with Nan to cover the entire period
            ds = ds.reindex(time=dates, fill_value=np.nan).sortby("time")
            ds.to_netcdf(filename)

        datebegin = Date(args.datebegin).ymd6h
        tbout = io.put(
            kind           = 'Precipitation',
            geometry       = geometry_map[domain],
            xpid           = 'raw@vernaym',
            vapp           = 'antilope',
            block          = block,
            datebegin      = f'{datebegin}/+PT24H',
            dateend        = Date(args.dateend).ymd6h,
            filename       = filename,
        )

#        antilope = xr.open_dataset(filename)
#        # Extract specific values where evaluation data (obs nivometeo) is available
#        #for num_poste, (lat, lon) in nivometeo.iteritems():  # python2 (guppy)
#        selection = pd.DataFrame(columns=['date', 'num_poste', 'rr_antilope'], dtype=object)
#        for num_poste, (lat, lon) in nivometeo.items():  # python3
#            #nearest = geometry.nearest_points(lon, lat, {'n':'1'})  # returns indices of the point in "data"
#            antilope_lat = nearest(antilope.lat, lat)
#            antilope_lon = nearest(antilope.lon, lon)
#            tmp = antilope.loc[{'lat':antilope_lat, 'lon':antilope_lon}]
#            new = {'date':list(tmp.time.data), 'num_poste':len(tmp.time)*[num_poste], 'rr_antilope':list(tmp.rr.data)}
#            new = pd.DataFrame(new)
#            selection = pd.concat([selection, new], ignore_index=True)
#
#        selection.set_index('date')
#        goto(args.workdir)
#        outname = f'{args.model}_{args.datebegin.strftime("%Y%m%d%H")}_{args.dateend.strftime("%Y%m%d%H")}_{domain}.csv'
#        selection.to_csv(outname, index=False, sep=';')
