#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 25/01/2022

import os
import datetime
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import argparse

import epygram

##############################################################################################
# Ce script sert à extraire les données ANTILOPE de la BDAP correspondant aux postes nivometeo
# dans le but d'évaluer le biais d'ANTILOPE avec l'altitude.
# A priori on ne peut extraire les données de la BDAP que sous forme de grille, on fait donc 
# une extraction pour chaque sous domaine d'intéret (alp, pyr, cor) avant de selectionner les
# pixels correspondant aux coordonnées des points des postes nivometeo. 
# TODO : vérifier si ce n'est pas plus rapide d'extraire un grib complet (peu probable).
##############################################################################################

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
    alp = ['46875', '43125', '4500', '8500'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '41000', '8000', '10500'],
    ange = ['45240', '44990', '6010', '6490']
)

# Pas en lat/lon de la grille cible : 0.375 / 0.5
dl = ['10', '10']

def parse_command_line():
    description = "BDAP extraction of ANTILOPE data."
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-d', '--rundate', help='Rundate for operational executions, format YYMMDDHH')
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-d', '--domain', nargs='+', help='Domain of the file', choices=coords.keys(), default=['alp', 'pyr', 'cor'])
    parser.add_argument('-w', '--workdir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/workdir/extraction_antilope')
#    parser.add_argument('-o', '--output', help='Output name of generated files')
    parser.add_argument('-m', '--model', help='Model from which the data must be extracted',
            choices=['ANTILOPEQ', 'ANTILOPEJP1Q', 'ANTILOPEH', 'ANTILOPEJP1H'], default='ANTILOPEJP1Q')
    parser.add_argument('-g', '--grid', help='BDAP grid name from which to extract data', default='FRANXL1S100')
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
    metadata = pd.read_csv('postes_nivometeo.csv', sep=';')
    latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
    subdata = metadata[(metadata['lat_dg']>=latmin) & (metadata['lat_dg']<=latmax) & (metadata['lon_dg']>=lonmin) & (metadata['lon_dg']<=lonmax)]
    return dict(zip(np.array(subdata['num_poste']), zip(np.array(subdata['lat_dg']), np.array(subdata['lon_dg']))))

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
                
    def extract(self, parameter, level, cmd='dap3_dev'):
        self.requete(parameter, level)
        startdate = self.date - timedelta(days=1) # File named "*ymdh" must contains 24h cumul since ym(d-1)h
        os.environ["DMT_DATE_PIVOT"] = startdate.strftime('%Y%m%d%H%M%S')
        print(os.environ["DMT_DATE_PIVOT"])
        os.system("{0:s} {1:d} {2:s}".format(cmd, ech, self.rqst))
        
    def run(self, parameter, level):
        if os.path.exists(self.gribname):
            print('File {0:s} already exists'.format(self.gribname))
            return True
        else:
            self.extract(parameter, level)
            if os.path.isfile(self.gribname):
                return True
            else:
                return False
        
if __name__ == "__main__":
    args = parse_command_line()
    if args.model  in ['ANTILOPEQ', 'ANTILOPEJP1Q']:
        dt = 24
    elif args.model in ['ANTILOPEH', 'ANTILOPEJP1H']:
        dt = 1
    extract_period = date_range(args.datebegin, args.dateend, dt=dt)

    antilope = pd.DataFrame(columns=['date', 'num_poste', 'rr_antilope'])
    for domain in args.domain:
        print(domain)
        goto(args.workdir)
        nivometeo = read_nivometeo_coords(domain)
        missing_grib = list()
        workdir = os.path.join(args.workdir, domain)
        goto(workdir)
        for date in extract_period:
            print(date.strftime('%Y%m%d%H'))
            if date.month in [1,2,3,4,11,12]: # Consider only month with nivometeo observations
                if not os.path.exists('{0:s}_{1:s}.grib'.format(args.model, date.strftime('%Y%m%d%H'))):
                    grib = ExtractGrib(args.model, args.grid, domain, date)
                    result = grib.run(args.parameter, args.level)
                    gribname = grib.gribname
                else:
                    result = True
                    gribname = '{0:s}_{1:s}.grib'.format(args.model, date.strftime('%Y%m%d%H'))
                if result:
                    data = epygram.formats.resource(gribname, openmode='r', fmt='GRIB')
                    rr_field = data.readfield({'indicatorOfTypeOfLevel':1, 'paramId': 0, 'indicatorOfParameter': 61}, getdata= True)
                    metadata = data.get_message_at_position(0).asfield(getdata=False)
                    geometry = metadata.geometry
                    for num_poste, (lat, lon) in nivometeo.iteritems():
                        nearest = geometry.nearest_points(lon, lat, {'n':'1'}) # returns indices of the point in "data"
                        antilope = antilope.append({
                            'date': date,
                            'num_poste':  int(num_poste),
                            'rr_antilope': rr_field.data[nearest[1]][nearest[0]]
                        }, ignore_index=True)
                else:
                    print('Missing date {0:s}'.format(date.strftime("%Y%m%d%H")))

    antilope.set_index('date')
    goto(args.workdir)
    outname = '{0:s}_{1:s}_{2:s}.csv'.format(args.model, args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
    antilope.to_csv(outname, index=False, sep=';')



