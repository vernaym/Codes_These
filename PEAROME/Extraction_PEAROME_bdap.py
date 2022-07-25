#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 20/07/2022

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr

import argparse

import epygram

##############################################################################################
# Ce script sert à extraire les données post-traitées de la PEAROME depuis la BDAP.
# On fait une extraction par domaine d'intéret (alp, pyr, cor, GrandesRousses...)
# En plus des fichiers grib journaliers, un fichier netcdf avec des dimensions
# (lat, lon) corrrespondant au domaine et une dimension temporelle contenant la période
# couverte par l'extraction
##############################################################################################

# Identifiant du modèle dans la BDAP
#model_id = PG1PEAROM001 à PG1PEAROM016 [PG1PEAROME]
# Identifiant de la grille
#grid = 'FRANXL0025'
# Indentifiant du paramètre
#parameter = 'PRECIP'
# Identifiant du niveau
#level = 'SOL'
# Identifiant du reseau
#time = 9h ou 21h
# Echeance
#ech = dict(9=range(21, 46, 1), 21=range(9, 34, 1)) #  Extraction de 24 precipitation horaires couvrant J+1 (6h) --> J+2 (6h)
# Format
#fmt = 'GRIB2_C_MAX'

parameter  = 'PRECIP'
level_type = 'SOL'

# Identifiant des modèles dans la BDAP
model_desc = 'PG1PEAROME'
#model_desc = ['PG1PEAROME{member:03d}' for member in range(1,17)]  # PG1PEAROME001 à PG1PEAROME016
model_desc = {member:'PG1PEAROM{0:03d}'.format(member) for member in range(1,17)} # PG1PEAROME001 à PG1PEAROME016

# Liste des coordonnées attendues par la commande dap3: lat_max, lat_min, lon_max, lon_min
# TODO : assurer que les points de grilles du sous domaines (coordonnées + pas lat/lon) sont
# bien confondus aves les points de la grille native pour éviter une interpolation
coords = dict(
    alp = ['47000', '43000', '4500', '8500'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '41000', '8000', '11500'],
    GrandesRousses = ['45250', '44750', '6000', '6500']
)

# Pas en lat/lon de la grille cible en 1/1000 de °
dl = dict(
    FRANXL0025 = ['25', '25'],
)

def parse_command_line():
    description = "BDAP extraction of NWP (AS-PEAROME) model data."
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-d', '--rundate', help='Rundate for operational executions, format YYMMDDHH')
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-d', '--domain', nargs='+', help='Domain of the file', choices=coords.keys(), default=['alp', 'pyr', 'cor', 'GrandesRousses'])
    parser.add_argument('-w', '--workdir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/workdir/extraction_PEAROME')
#    parser.add_argument('-o', '--output', help='Output name of generated files')
    parser.add_argument('-m', '--model', help='NWP model from which the data must be extracted', 
                choices=['pearome', 'stats'], default='pearome')
    parser.add_argument('-c', '--cutoff', help='NWP model cutoff from which the data must be extracted', choices=['assimilation', 'prevision'], default='prevision')
    parser.add_argument('-g', '--grid', help='BDAP grid name from which to extract data', default='FRANXL0025')
    #parser.add_argument('-v', '--vortex', action='store_true', help='Store generated files on a vortex archive store (operational use only)')
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

class ExtractGrib(object):

    def __init__(self, model, grid, domain, rundate, ech=6, member=None):

        self.model          = model
        self.domain         = domain
        self.member         = member
        self.grid           = grid.upper()
        self.coords         = coords[domain]
        self.date           = rundate
        self.ech            = ech
        self.gribname       = '{0:s}_{1:s}_{2:d}_{3:s}.grib'.format(self.model, self.date.strftime('%Y%m%d%H'), self.ech, self.domain)
        self.extractedfiles = list()

    def requete(self):
        self.rqst = 'requete.tmp'
        #extractfile = '{0:s}_{1:02d}.grib'.format(self.model, self.ech)
        f = open(self.rqst, "w")
        f.write('#RQST\n')
        f.write('#NFIC {0:s}\n'.format(self.gribname))
        f.write('#MOD {0:s}\n'.format(model_desc[self.member]))
        f.write('#PARAM {0:s}\n'.format(parameter))
        f.write('#Z_REF {0:s}\n'.format(self.grid))
        if coords is not None:
            f.write("#Z_EXTR INTERPOLATION\n")
            f.write('#Z_GEO ' + ' '.join(coords[self.domain]) + '\n')
            f.write('#Z_STP ' + ' '.join(dl[self.grid]) + '\n')
        f.write('#L_TYP {0:s}\n'.format(level_type))

    def extract(self, cmd='dap3_dev'):
        self.requete()
        os.system("{0:s} {1:d} {2:s}".format(cmd, self.ech, self.rqst))
        self.extractedfiles.append(self.gribname)

        if os.path.exists(self.gribname) and os.path.getsize(self.gribname) > 0:
            return True
        else:
            return False

    def concatenate(self):
        os.system('cat ' + ' '.join(self.extractedfiles) + ' > {0:s}'.format(self.gribname))

    def run(self):
        if os.path.exists(self.gribname):
            print('File {0:s} already exists'.format(self.gribname))
            return None
        else:
            fileok = True
            if not self.extract():
                fileok = False

            if fileok:
                return None
            else:

                return self.gribname + ' for member {0:d}'.format(self.member)


if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend)
    for domain in args.domain:
        workdir = os.path.join(args.workdir, domain)
        goto(workdir)
        for member in range(1, 17):
            missing_grib = list()
            time = pd.date_range(args.datebegin, args.dateend)
            reference_time = pd.Timestamp(args.datebegin)
            rr24 = None
            nan  = None
            #cumul = None
            i=0
            for date in extract_period:
                i = i+1
                goto(os.path.join(workdir, nivologyseason(date), 'mb{0:03d}'.format(member)))
                # date = J (6h), on veut le cumul prévu entre J 6h et J+1 6h par le réseau de J-1 21h
                # Il faut donc extraire les echeances 9h à 24h de J-1 21h (=DMT_DATE_PIVOT)
                datepivot = date - timedelta(hours=9)
                os.environ["DMT_DATE_PIVOT"] = datepivot.strftime('%Y%m%d%H%M%S')
                #for ech in range(9, 34, 1):  # Pour le réseau de 21h
                ech = 33  # Correpond à un cumul 24h entre 6h J+1 et 6h J+2 (cf page 259 doc BDAP)
                print(os.environ["DMT_DATE_PIVOT"])
                grib = ExtractGrib(args.model, args.grid, domain, date, ech=ech, member=member)
                result = grib.run()
                if result is not None:
                    print('Missing date {0:s}'.format(date.strftime("%Y%m%d%H")))
                    missing_grib.append(result)
                    # Fill missing day with nan values
                    # WARNING : this only works if the first date of the period have valid data
                    if nan is None:
                        nan = np.empty(shape)
                        nan[:] = np.NaN
                    rr24 = np.append(rr24, np.array([nan]), axis=0)
                else:
                    gribname = grib.gribname
                    data = epygram.formats.resource(gribname, openmode='r', fmt='GRIB')
                    rr_field = data.readfield({'indicatorOfTypeOfLevel':1, 'paramId': 0, 'indicatorOfParameter': 61}, getdata= True)
                    metadata = data.get_message_at_position(0).asfield(getdata=False)
                    geometry = metadata.geometry
                    lon = geometry.get_lonlat_grid()[0]
                    lat = geometry.get_lonlat_grid()[1]
                    if rr24 is None:
                        rr24 = np.array([rr_field.data.data,])
                        shape = np.shape(rr_field.data.data)  # Save the data shape to fill missing dates with nan values
                    else:
                        rr24 = np.append(rr24, np.array([rr_field.data.data]), axis=0)
#                    if cumul is None:
#                        cumul = rr_field
#                    else:
#                        cumul += rr_field
            #  np.shape(rr24) = (time, 21, 21) et il faut un data de la forme (21, 21, time)
            rr = xr.DataArray(
                data = np.transpose(rr24, (1,2,0)),  # Pour passer la dimension temporelle en dernier : (lon, lat, time)
                name = 'rr',
                dims=["x", "y", "time"],
                coords=dict(lon=(["x", "y"], lon),lat=(["x", "y"], lat), time=time, reference_time=reference_time,),
                attrs=dict(description="24 hour precipitation",units="mm/24h"),
            )
            outname = '{0:s}_{1:03d}_{2:s}_{3:s}_{4:s}.nc'.format(args.model, member, args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), domain)
            rr.to_netcdf(outname)

            #cumul.dump_to_nc('CUMUL_{0:s}_{1:s}_{2:s}_{3:s}.nc'.format(args.model, args.datebegin.strftime("%Y%m%d%H"), args.dateend.strftime("%Y%m%d%H"), domain), variablename="rr_cumul")


            if len(missing_grib) > 0:
                    with open('missing_grib', 'w') as f:
                        for m in missing_grib:
                            f.write('{0:s}\n'.format(m))


