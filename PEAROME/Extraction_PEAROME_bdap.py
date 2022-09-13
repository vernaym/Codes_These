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

from vortex import toolbox
import common
import footprints

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

pearome_desc = {member:'PG1PEAROM{0:03d}'.format(member) for member in range(1,17)} # PG1PEAROME001 à PG1PEAROME016

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
    EURW1S40   = ['25', '25'],
    EURW1S100  = ['10', '10'],
)

paramID = dict(
    arome   = 228228,
    pearome = 0,
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
                choices=['pearome', 'stats', 'arome'], default='pearome')
    parser.add_argument('-c', '--cutoff', help='NWP model cutoff from which the data must be extracted', choices=['assimilation', 'prevision'], default='prevision')
    parser.add_argument('-g', '--grid', help='BDAP grid name from which to extract data', default='FRANXL0025', choices=['FRANXL0025', 'EURW1S40', 'EURW1S100'])
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

    def __init__(self, model, grid, domain, ech, origin, rundate=None, member=None):

        self.model          = model
        self.domain         = domain
        self.member         = member
        self.grid           = grid.upper()
        self.coords         = coords[domain]
        self.date           = rundate
        self.ech            = ech
        if rundate is None:
            self.gribname       = 'geometry.grib'
        else:
            self.gribname       = '{0:s}_{1:s}_{2:d}_{3:s}.grib'.format(self.model, self.date.strftime('%Y%m%d%H'), self.ech, self.domain)
        self.extractedfiles = list()
        self.vapp           = 'arome'
        # Identifiant des modèles dans la BDAP
        if model == 'stat':
            self.model_desc = 'PG1PEAROME'
        elif model == 'pearome':
            self.model_desc = pearome_desc[member]
            #model_desc = ['PG1PEAROME{member:03d}' for member in range(1,17)]  # PG1PEAROME001 à PG1PEAROME016
            self.vconf = 'pefrance'
        elif model == 'arome':
            self.model_desc = 'PAROME'
            self.vconf = '3dvarfr'
        self.origin = origin

    def requete(self):
        self.rqst = 'requete.tmp'
        #extractfile = '{0:s}_{1:02d}.grib'.format(self.model, self.ech)
        f = open(self.rqst, "w")
        f.write('#RQST\n')
        f.write('#NFIC {0:s}\n'.format(self.gribname))
        f.write('#MOD {0:s}\n'.format(self.model_desc))
        f.write('#PARAM {0:s}\n'.format(parameter))
        f.write('#Z_REF {0:s}\n'.format(self.grid))
        if coords is not None:
            f.write("#Z_EXTR INTERPOLATION\n")
            f.write('#Z_GEO ' + ' '.join(coords[self.domain]) + '\n')
            f.write('#Z_STP ' + ' '.join(dl[self.grid]) + '\n')
        f.write('#L_TYP {0:s}\n'.format(level_type))

    def extract_from_bdap(self, cmd='dap3_dev'):
        self.requete()
        os.system("{0:s} {1:d} {2:s}".format(cmd, self.ech, self.rqst))
        self.extractedfiles.append(self.gribname)
        if os.path.exists(self.gribname):
            if os.path.getsize(self.gribname) > 0:
                return True
            else:
                os.remove(self.gribname)
        return False

    def extract_from_hendrix(self):

        tbaro = toolbox.input(
            role           = 'Gridpoint',
            format         = 'grib',
            geometry       = self.grid,
            kind           = 'gridpoint',
            suite          = 'oper',
            local          = self.gribname,
            date           = self.date.strftime('%Y%m%d%H'),
            term           = self.ech,
            cutoff         = 'production',
            namespace      = 'vortex.multi.fr',
            block          = 'forecast',
            nativefmt      = '[format]',
            origin         = 'historic',
            model          = '[vapp]',
            vapp           = self.vapp,
            vconf          = self.vconf,
            fatal          = False,
            now            = True,
        )
        print('tbaro =', tbaro)
        if os.path.exists(self.gribname):
            return True
        else:
            return False

    def concatenate(self):
        os.system('cat ' + ' '.join(self.extractedfiles) + ' > {0:s}'.format(self.gribname))

    def run(self):
        if os.path.exists(self.gribname):
            if os.path.getsize(self.gribname) > 0:
                #print('File {0:s} already exists'.format(self.gribname))
                return None
            else:
                print('Removing empty file {0:s}'.format(self.gribname))
                os.remove(self.gribname)

        if self.origin == 'bdap':
            result = self.extract_from_bdap()
        elif self.origin == 'hendrix':
            result = self.extract_from_hendrix()

        if result:
            return None
        else:
            # No file has been extracted
            if self.member is not None:
                return self.gribname + ' for member {0:d}'.format(self.member)
            else:
                return self.gribname


class PrecipitationExtractor(object):

    def __init__(self, args, echeance, domain, timecoord, member=None):
        self.args      = args
        self.echeance  = echeance
        self.member    = member
        self.rr24      = None
        self.domain    = domain
        self.geometry  = None
        self.timecoord = timecoord

    def read_geometry(self):
        result = None
        if self.origin == 'hendrix':  # Need to extract a grib file from BDAP to get the correct geometry
            grib = ExtractGrib(self.args.model, self.args.grid, self.domain, self.echeance, 'bdap', member=self.member)
            result = grib.run()
        if result is not None:
            print('Missing grib, try another date (use default DMT_DATE_PIVOT)')
        else:
            data = epygram.formats.resource(grib.gribname, openmode='r', fmt='GRIB')
            metadata = data.get_message_at_position(0).asfield(getdata=False)
            self.geometry = metadata.geometry
            self.lon = self.geometry.get_lonlat_grid()[0]
            self.lat = self.geometry.get_lonlat_grid()[1]

    def read_grib(self, gribname):
        data = epygram.formats.resource(gribname, openmode='r', fmt='GRIB')
        if self.geometry is None:  # Reading first grib file
            self.read_geometry()
        if self.origin == 'bdap':
            # If the grib file has been extracted from the BDAP, the precipitation field has been computed at the extraction
            rr_field = data.readfield({'indicatorOfTypeOfLevel': 1, 'paramId': paramID[args.model], 'indicatorOfParameter': 61}, getdata= True)
        elif self.origin == 'hendrix':
            # If the grib file comes from hendrix, the precipitation field must be cumputed now
            rain = data.extract_subdomain({'parameterNumber': 65, 'level': 0}, self.geometry)
            snow = data.extract_subdomain({'parameterNumber': 66, 'level': 0}, self.geometry)
            rr_field = rain + snow
        if self.rr24 is None:  # Reading first grib file
            self.rr24 = np.array([rr_field.data,])
            self.shape = np.shape(rr_field.data)  # Save the data shape to fill missing dates with nan values
        else:
            self.rr24 = np.append(self.rr24, np.array([rr_field.data]), axis=0)
#       if cumul is None:
#           cumul = rr_field
#       else:
#           cumul += rr_field

    def extract(self, dt=0):
        missing_grib = list()
        reference_time = pd.Timestamp(self.args.datebegin)
        nan  = None
        #cumul = None
        i=0
        for date in extract_period[:-1]:  # Verrue pour avoir la bonne dimension temporelle
            i = i+1
            if self.member is not None:  # Extraction de la pearome depuis la BDAP
                self.origin = 'bdap'
                goto(os.path.join(workdir, 'mb{0:03d}'.format(self.member)))
            else:  # Extraction d'AROME depuis hendrix
                self.origin = 'hendrix'
                goto(workdir)
            datepivot = date - timedelta(hours=dt)
            os.environ["DMT_DATE_PIVOT"] = datepivot.strftime('%Y%m%d%H%M%S')
            #for ech in range(9, 34, 1):  # Pour le réseau de 21h
            print(os.environ["DMT_DATE_PIVOT"])
            grib = ExtractGrib(self.args.model, self.args.grid, self.domain, self.echeance, self.origin, date, member=self.member)
            result = grib.run()
            if result is not None:
                print('Missing date {0:s}'.format(date.strftime("%Y%m%d%H")))
                missing_grib.append(result)
                # Fill missing day with nan values
                # WARNING : this only works if the first date of the period have valid data
                if nan is None:
                    nan = np.empty(self.shape)
                    nan[:] = np.NaN
                self.rr24 = np.append(self.rr24, np.array([nan]), axis=0)
            else:
                self.read_grib(grib.gribname)

        rr = xr.DataArray(
            data = np.transpose(self.rr24, (1,2,0)),  # Pour passer la dimension temporelle en dernier : (lon, lat, time)
            name = 'rr',
            dims=["lat", "lon", "time"],
            coords=dict(lon=self.lon[0], lat=self.lat[:,0], time=self.timecoord, reference_time=reference_time,),
            attrs=dict(description="24 hour precipitation",units="mm/24h"),
        )

        if self.member is not None:
            outname = '{0:s}_{1:03d}_{2:s}_{3:s}_{4:s}.nc'.format(self.args.model, self.member, self.args.datebegin.strftime('%Y%m%d%H'), self.args.dateend.strftime('%Y%m%d%H'), self.domain)
        else:
            outname = '{0:s}_{1:s}_{2:s}_{3:s}.nc'.format(self.args.model, self.args.datebegin.strftime('%Y%m%d%H'), self.args.dateend.strftime('%Y%m%d%H'), self.domain)
        # WARNING : DO NOT WORK ON GUPPY !
        # Transfert the extracted files locally and rerun this script
        rr.to_netcdf(outname)

        #cumul.dump_to_nc('CUMUL_{0:s}_{1:s}_{2:s}_{3:s}.nc'.format(args.model, args.datebegin.strftime("%Y%m%d%H"), args.dateend.strftime("%Y%m%d%H"), domain), variablename="rr_cumul")


        if len(missing_grib) > 0:
                with open('missing_grib', 'w') as f:
                    for m in missing_grib:
                        f.write('{0:s}\n'.format(m))

if __name__ == "__main__":
    args = parse_command_line()

    for domain in args.domain:
        workdir = os.path.join(args.workdir, domain, args.model)
        goto(workdir)
        extract_period = date_range(args.datebegin, args.dateend)

        if args.model == 'pearome':
            timecoord = extract_period
            # Dans le cas de le PEAROME post_traitée : date = J (6h) et on veut le cumul prévu entre J 6h et J+1 6h par le réseau de J-1 21h
            # Il faut donc extraire les echeances 9h à 24h de J-1 21h (=DMT_DATE_PIVOT)
            dt = 9
            echeance = 33  # Correpond à un cumul 24h entre 6h J+1 et 6h J+2 (cf page 259 doc BDAP)
            for member in range(1, 17):
                precip = PrecipitationExtractor(args, echeance, domain, timecoord[:-1], member=member)
                precip.extract(dt=9)
        else:
            echeance = 24
            # AROME data are extracted from hendrix : 24h forecasts lead times provide the previous 24h precipitation accumulation
            extract_preiod = date_range(args.datebegin, args.dateend, dt=24)
            timecoord = extract_period[:-1]
            precip = PrecipitationExtractor(args, echeance, domain, timecoord)
            precip.extract()


