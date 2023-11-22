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
parameter = dict(
        pearome   = ['PRECIP'],
        parome    = ['PRECIP'],
        aspearome = ['PRECIP'],
        stats     = ['RR24_Q50'],
        #stats     = ['RR24_MIN', 'RR24_MAX', 'RR24_Q50'],
    )
level_type = 'SOL'

pearome_desc = {member:'PG1PEAROM{0:03d}'.format(member) for member in range(1,17)} # PG1PEAROME001 à PG1PEAROME016

# Liste des coordonnées attendues par la commande dap3: lat_max, lat_min, lon_max, lon_min
# TODO : assurer que les points de grilles du sous domaines (coordonnées + pas lat/lon) sont
# bien confondus aves les points de la grille native pour éviter une interpolation
coords = dict(
    #alp = ['47000', '43000', '4500', '8500'],
    #alp = ['46450', '44100', '5400', '7200'],  # extaction ANTILOPE
    #alp = ['46800', '43000', '4500', '8000'],  # To take into account localisation
    alp = ['46800', '43700', '5000', '7600'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '41000', '8000', '11500'],
    GrandesRousses = ['45640', '44590', '5610', '7100'],  # With 0.2 margin (final domain : ['45440', '44790', '5810', '6690'],)
)

# Pas en lat/lon de la grille cible en 1/1000 de °
dl = dict(
    FRANXL0025 = ['25', '25'],
    EURW1S40   = ['25', '25'],
    EURW1S100  = ['10', '10'],
)

paramID = dict(
    aarome    = 228228,
    parome    = 85029,
    pearome   = 0,
    aspearome = 85029,
    stats     = 0,
)
indicatorOfParameter = dict(
        pearome   = 61,
        parome    = 61,
        aspearome = 61,
        stats     = 13,
)

datadir = '/home/vernaym/These/DATA'

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
                choices=['pearome', 'aspearome', 'stats', 'aarome', 'parome'], default='pearome')
    parser.add_argument('-c', '--cutoff', help='NWP model cutoff from which the data must be extracted', choices=['assimilation', 'prevision'], default='prevision')
    parser.add_argument('-g', '--grid', help='BDAP grid name from which to extract data', default='FRANXL0025', choices=['FRANXL0025', 'EURW1S40', 'EURW1S100'])
    parser.add_argument('-r', '--read', action = 'store_true', help="Read data (to disable on guppy since xarray.to_netcdf doesn't work)", default=False)
    parser.add_argument('-t', '--pdt', help='Pas de temps des precipitation à extraire (horaires ou journalières)', choices=[1, 24], type=int)

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

def datespivot(start, end, reseau=21, dt=1):
    # datepivot = réseau de 21h dont la prévi couvre J-1 6h --> J 6h
    datepivot = start-timedelta(hours=24+start.hour+3)  # réseau de 21h (J-2)
    datespivot = list()
    validitydates = list()
    while datepivot <= end:
        datespivot.append(datepivot)
        for h in range(1, 25, dt):
            validitydates.append(datepivot+timedelta(hours=9)+timedelta(hours=h))
        datepivot = datepivot + timedelta(hours=24)

    return datespivot, validitydates


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
        if model == 'stats':
            self.model_desc = 'PG1PEAROME'
        elif model == 'aspearome':
            self.model_desc = pearome_desc[member]
            #model_desc = ['PG1PEAROME{member:03d}' for member in range(1,17)]  # PG1PEAROME001 à PG1PEAROME016
            self.vconf = 'pefrance'
        elif model in ['aarome', 'parome']:
            self.model_desc = 'PAROME'
            self.vconf = '3dvarfr'
        self.origin = origin
        self.missing = None

    def requete(self):
        self.rqst = 'requete.tmp'
        #extractfile = '{0:s}_{1:02d}.grib'.format(self.model, self.ech)
        f = open(self.rqst, "w")
        f.write('#RQST\n')
        f.write('#NFIC {0:s}\n'.format(self.gribname))
        f.write('#MOD {0:s}\n'.format(self.model_desc))
        f.write('#PARAM ' + ' '.join(parameter[self.model]) + '\n')
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
                print('Removing empty file {0:s}'.format(self.gribname))
                os.remove(self.gribname)
        return False

    def extract_from_hendrix(self):

        if self.model == 'parome':
            cutoff = 'production'
        elif self.model == 'aarome':
            cutoff = 'assimilation'

        tbaro = toolbox.input(
            role           = 'Gridpoint',
            format         = 'grib',
            geometry       = self.grid,
            kind           = 'gridpoint',
            suite          = 'oper',
            local          = self.gribname,
            date           = self.date.strftime('%Y%m%d%H'),
            term           = self.ech,
            cutoff         = cutoff,
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
                return self.gribname
            else:
                print('Removing empty file {0:s}'.format(self.gribname))
                os.remove(self.gribname)

        if self.origin == 'bdap':
            result = self.extract_from_bdap()
        elif self.origin == 'hendrix':
            result = self.extract_from_hendrix()

        if result:
            return self.gribname
        else:
            # No file has been extracted
            if self.member is not None:
                self.missing = self.gribname + ' for member {0:d}'.format(self.member)
            else:
                self.missing = self.gribname
            return None


class PrecipitationExtractor(object):

    def __init__(self, args, echeance, domain, timecoord, member=None):
        self.args      = args
        self.echeance  = echeance
        self.member    = member
        self.rr        = None
        self.domain    = domain
        self.geometry  = None
        self.timecoord = timecoord
        self.shape     = None
        self.nan  = None

    def read_geometry(self, data):
        result = True
        if self.origin == 'hendrix':  # Need to extract a grib file from BDAP to get the correct geometry
            grib = ExtractGrib(self.args.model, self.args.grid, self.domain, self.echeance, 'bdap', member=self.member)
            result = grib.run()
            data = epygram.formats.resource(result, openmode='r', fmt='GRIB')
        if result is None:
            print('Missing grib, try another date (use default DMT_DATE_PIVOT)')
        else:
            metadata = data.get_message_at_position(0).asfield(getdata=False)
            self.geometry = metadata.geometry
            self.lon = self.geometry.get_lonlat_grid()[0]
            self.lat = self.geometry.get_lonlat_grid()[1]

    def read_grib(self, gribname):
        data = epygram.formats.resource(gribname, openmode='r', fmt='GRIB')
        if self.geometry is None:  # Reading first grib file
            self.read_geometry(data)
        if self.origin == 'bdap':
            # If the grib file has been extracted from the BDAP, the precipitation field has been computed at the extraction
            rr_field = data.readfield({'indicatorOfTypeOfLevel': 1, 'paramId': paramID[args.model], 'indicatorOfParameter': indicatorOfParameter[args.model]}, getdata= True)
        elif self.origin == 'hendrix':
            # If the grib file comes from hendrix, the precipitation field must be cumputed now
            rain = data.extract_subdomain({'parameterNumber': 65, 'level': 0}, self.geometry)
            snow = data.extract_subdomain({'parameterNumber': 66, 'level': 0}, self.geometry)
            rr_field = rain + snow
        if self.shape is None:
            self.shape = np.shape(rr_field.data)  # Save the data shape to fill missing dates with nan values

        return rr_field

    def get_data(self, date):
        if self.args.model == 'aspearome':
            # Les champs de précip reconstitués après AS sont des cumuls de précipitation depuis le réseau de production
            # Les AS PEAROME tournent sur les réseaux de 9h et 21h, on utilise donc le réseau de 21h (J-2). Pour obtenir des cumuls
            # de précip entre 6h (J-1) et 6h (J) il faut donc faire la différence entre les cumuls de l'échéance 33h (cumul de 21h (J-2)
            # à 6h (J) et le cumul de l'échéance 9h (cumul de 21h (J-2) à 6h (J-1))
            #grib1 = ExtractGrib(self.args.model, self.args.grid, self.domain, 9, self.origin, date, member=self.member)  # ech 9h du réseau de 21h (J-2) valable pour J-1 (6h)
            #grib2 = ExtractGrib(self.args.model, self.args.grid, self.domain, 33, self.origin, date, member=self.member)  # ech 33h du réseau de 21h (J-2) valable pour J (6h)
            extract = [ExtractGrib(self.args.model, self.args.grid, self.domain, ech, self.origin, date, member=self.member) for ech in self.echeance]
            gribs = [grib.run() for grib in extract]  # WARNING : the order is important (increasing lead times)
        else:
            grib = ExtractGrib(self.args.model, self.args.grid, self.domain, self.echeance, self.origin, date, member=self.member)
            gribs = [grib.run()]

        fail = None in gribs
#        if fail:
#            print('Missing data for date {0:s}'.format(date.strftime("%Y%m%d%H")))
#            self.missing_grib.append(','.join(gribs))
        if args.read:
            self.read_data(gribs, fail)

    def read_data(self, gribs, fail):
        if fail:
            # Fill missing data with nan values
            # WARNING : this only works if the first date of the period have valid data
            if self.nan is None:
                if self.shape is None:
                    print('ERROR : no known shape to fill missing data')
                else:
                    self.nan = np.empty(self.shape)
                    self.nan[:] = np.NaN
            if len(gribs) == 1:
                self.update_data(self.nan)
            else:
                for idx in range(1, len(gribs)):
                    self.update_data(self.nan)
        else:
            if len(gribs) == 1:
                self.update_data(self.read_grib(gribs[0]).data)
            elif self.args.model == 'aspearome':
                # Calcul des cumuls horaires ou journalier
                for idx in range(1, len(gribs)):
                    rrh1  = self.read_grib(gribs[idx-1])
                    rrh2 = self.read_grib(gribs[idx])
                    newdata = rrh2.data-rrh1.data
                    self.update_data(newdata)

    def update_data(self, mydata):
        if self.rr is None:  # Reading first grib file
            self.rr = np.array([mydata,])
        else:
            self.rr = np.append(self.rr, np.array([mydata]), axis=0)

    def extract(self, datespivot):
        self.missing_grib = list()
        reference_time = pd.Timestamp(self.args.datebegin)
        #cumul = None

        for date in datespivot:
            if self.member is not None:  # Extraction de la pearome depuis la BDAP
                self.origin = 'bdap'
                goto(os.path.join(workdir, 'mb{0:03d}'.format(self.member)))
            #elif date<datetime(2022, 10, 1):
            else:
                self.origin = 'bdap'
                goto(workdir)
            #else:  # Extraction d'AROME depuis hendrix
            #    self.origin = 'hendrix'
            #    goto(workdir)
            os.environ["DMT_DATE_PIVOT"] = date.strftime('%Y%m%d%H%M%S')
            print(os.environ["DMT_DATE_PIVOT"])
            self.get_data(date)

        if len(self.missing_grib) > 0:
            with open('missing_grib', 'w') as f:
                for m in self.missing_grib:
                    f.write('{0:s}\n'.format(m))

        if self.args.read:

            rr = xr.DataArray(
                data = np.transpose(self.rr, (1,2,0)),  # Pour passer la dimension temporelle en dernier : (lon, lat, time)
                name = 'rr',
                dims=["lat", "lon", "time"],
                #coords=dict(lon=self.lon[0], lat=self.lat[:,0], time=self.timecoord, reference_time=reference_time,),
                coords=dict(lon=self.lon[0], lat=self.lat[:,0], time=self.timecoord, reference_time=self.timecoord[0],),
                attrs=dict(description="Precipitation",units="mm"),
            )

            datedeb = self.args.datebegin-timedelta(hours=24)
            if self.member is None:
                outname = '{0:s}_{1:s}_{2:s}_{3:s}.nc'.format(self.args.model, datedeb.strftime('%Y%m%d%H'), self.args.dateend.strftime('%Y%m%d%H'), self.domain)
                rr.to_netcdf(outname)
                return None
            else:
                # On sauvegarde un fichier par member (beaucoup plus rapide)
                outname = '{0:s}_{1:03d}_{2:s}_{3:s}_{4:s}.nc'.format(self.args.model, self.member, datedeb.strftime('%Y%m%d%H'), self.args.dateend.strftime('%Y%m%d%H'), self.domain)
                rr.to_netcdf(outname)
                return rr
            # WARNING : DOES NOT WORK ON GUPPY !
            # Transfert the extracted files locally and rerun this script
            #cumul.dump_to_nc('CUMUL_{0:s}_{1:s}_{2:s}_{3:s}.nc'.format(args.model, args.datebegin.strftime("%Y%m%d%H"), args.dateend.strftime("%Y%m%d%H"), domain), variablename="rr_cumul")
        else:
            return None


    def read_nivometeo_coords():
        metadata = pd.read_csv(os.path.join(datadir, 'postes_nivometeo.csv'), sep=';')
        latmax, latmin, lonmin, lonmax = np.array(coords[self.domain]).astype(float)/1000.
        subdata = metadata[(metadata['poste_nivo.lat_dg']>=latmin) & (metadata['poste_nivo.lat_dg']<=latmax) & (metadata['poste_nivo.lon_dg']>=lonmin) & (metadata['poste_nivo.lon_dg']<=lonmax)]
        return dict(zip(np.array(subdata['poste_nivo.num_poste']), zip(np.array(subdata['poste_nivo.lat_dg']), np.array(subdata['poste_nivo.lon_dg']))))

    def extract_station_values():
        # To Extract specific values where evaluation data (obs nivometeo) is available
        nivometeo = self.read_nivometeo_coords(self.domain)
        data = pd.DataFrame(columns=['date', 'num_poste', 'rr_{0:s}'.format(self.product)], dtype=object)
        for num_poste, (lat, lon) in nivometeo.iteritems():  # python2 (guppy)
            nearest = geometry.nearest_points(lon, lat, {'n':'1'})  # returns indices of the point in "data"
            data = data.append({
                'date':date,
                'num_poste':int(num_poste),
                'rr.{0:s}'.format(self.product): self.model_field.data[nearest[1]][nearest[0]]
            }, ignore_index=True)

        return data

    def write_station_data(data):
        data.set_index('date')
        outname = '{0:s}_{1:s}_{2:s}.csv'.format(self.product, args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
        data.to_csv(outname, index=False, sep=';')



if __name__ == "__main__":
    args = parse_command_line()

    for domain in args.domain:
        workdir = os.path.join(args.workdir, domain, args.model)
        goto(workdir)
        #timecoord = date_range(args.datebegin, args.dateend, dt=args.pdt)
        datespivot, timecoord = datespivot(args.datebegin, args.dateend, dt=args.pdt)

        if args.model in ['aspearome', 'stats']:
            if args.pdt == 24:
                # Dans le cas de le PEAROME post_traitée : date = J (6h) et on veut le cumul prévu entre J 6h et J+1 6h par le réseau de J-1 21h
                # Il faut donc extraire les echeances 9h à 24h de J-1 21h (=DMT_DATE_PIVOT)
                #dt = 9  # réseau 21h (J-1)
                echeances = [9, 33]  # WARNING : pour les statistiques (quantiles), l'échéance33h correpond à un cumul 24h entre 6h J+1 et 6h J+2
                # (cf page 259 doc BDAP). En revanche pour les 16 "membres" (champs de précip) reconstitués il faut bien prendre l'échéance 24h
                # du réseau de 6h ou faire une différence entre 2 échéances distantyes de 24h.
            elif args.pdt == 1:
                #dt = 1
                echeances = range(9, 34)
            precipitation = dict()
            for member in range(1, 17):
                precip = PrecipitationExtractor(args, echeances, domain, timecoord, member=member)
                precipitation[member] = precip.extract(datespivot)
            goto(workdir)
            datedeb = args.datebegin-timedelta(hours=24)
            # Fichier trop gros sur les Alpes ==> "Processus arrêté"
            #outname = '{0:s}_{1:s}_{2:s}_{3:s}.nc'.format(args.model, datedeb.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), domain)
            #data = xr.concat([arr for arr in precipitation.values()], pd.Index(precipitation.keys(), name="member")).transpose('lat', 'lon', 'time', 'member')
            #data.to_netcdf(outname)
        else:
            echeance = 24
            # AROME data are extracted from hendrix : 24h forecasts lead times provide the previous 24h precipitation accumulation
            extract_period = date_range(args.datebegin, args.dateend, dt=24)
            #timecoord = extract_period[:-1]
            timecoord = extract_period
            precip = PrecipitationExtractor(args, echeance, domain, timecoord)
            precip.extract(extract_period)


