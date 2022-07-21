#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 20/07/2022

import os
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np

import argparse

import epygram

##############################################################################################
# Ce script sert à extraire les données post-traitées de la PEAROME depuis la BDAP.
# On fait une extraction pour chaque sous domaine d'intéret (alp, pyr, cor,...)
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
#ech = 
# Format
#fmt = 'GRIB2_C_MAX'

list_level = dict(
    SOL        = [],
    SURFACE2M  = [2,],
    SURFACE10M = [10,],
    HAUTEUR    = [20, 50, 100, 500],
    ISOBARE    = [1000, 950, 925, 900, 850, 800, 700, 600, 500],
    ISO_T      = [27315],
)

list_parameters = dict(
    SOL        = ['PRLCV', 'PRLGE', 'PRNCV', 'PRNGE', 'P'],
    SURFACE2M  = ['T', 'HU'],
    SURFACE10M = ['U', 'V'],
    HAUTEUR    = ['P', 'T', 'U', 'V', 'HU'],
    ISOBARE    = ['Z', 'T', 'U', 'V', 'HU'],
    ISO_T      = ['ALTITUDE'],
)

# Identifiant des modèles dans la BDAP
model_desc = dict(
    arpege = dict(
        assimilation = 'PAA',
        prevision    = 'PA',
        ),
    pearp  = dict(
        prevision = 'PEARP',
        )
)

# Liste des coordonnées attendues par la commande dap3: lat_max, lat_min, lon_max, lon_min
coords = dict(
    alp = ['46875', '43125', '4500', '8500'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '40750', '8000', '11500'],
    ange = ['45240', '44990', '6010', '6490']
)

# Pas en lat/lon de la grille cible
dl = ['10', '10']

def parse_command_line():
    description = "BDAP extraction of NWP (AS-PEAROME) model data."
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-d', '--rundate', help='Rundate for operational executions, format YYMMDDHH')
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-d', '--domain', nargs='+', help='Domain of the file', choices=coords.keys(), default=['alp', 'pyr', 'cor'])
    parser.add_argument('-w', '--workdir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/workdir/extraction_PEAROME')
#    parser.add_argument('-o', '--output', help='Output name of generated files')
    parser.add_argument('-m', '--model', help='NWP model from which the data must be extracted', 
                choices=['pearome', 'stats'], default='pearome')
    parser.add_argument('-c', '--cutoff', help='NWP model cutoff from which the data must be extracted', choices=['assimilation', 'prevision'], default='prevision')
    parser.add_argument('-g', '--grid', help='BDAP grid name from which to extract data', default='FRANXLS0025')
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

    def __init__(self, model, cutoff, grid, domain, rundate, ech=6, member=None):

        self.model          = model
        self.cutoff         = cutoff
        self.domain         = domain
        self.member         = member
        self.grid           = grid.upper()
        self.coords         = coords[domain]
        self.date           = rundate
        self.ech            = ech
        self.gribname       = '{0:s}_{1:s}_{2:d}.grib'.format(self.model, self.date.strftime('%Y%m%d%H'), self.ech)
        self.extractedfiles = list()

    def requete(self, parameters, level_type):


        actual_level_type = level_type

        self.rqst = 'requete_{0:s}'.format(level_type)
        extractfile = '{0:s}_{1:s}.grib'.format(self.model, level_type)
        f = open(self.rqst, "w")
        f.write('#RQST\n')
        f.write('#NFIC {0:s}\n'.format(extractfile))
        if self.member is not None:
            f.write('#MOD {0:s}{1:03d}\n'.format(model_desc[self.model][self.cutoff], self.member))
        else:
            f.write('#MOD {0:s}\n'.format(model_desc[self.model][self.cutoff]))
        f.write('#PARAM ' + ','.join(parameters) +'\n')
        f.write('#Z_REF {0:s}\n'.format(self.grid))
        if coords is not None:
            f.write("#Z_EXTR INTERPOLATION\n")
            f.write('#Z_GEO ' + ' '.join(coords[self.domain]) + '\n')
            f.write('#Z_STP ' + ' '.join(dl) + '\n')
        f.write('#L_TYP {0:s}\n'.format(actual_level_type))
        if level_type != 'SOL':
            f.write('#L_LST ' + ','.join(str(l) for l in list_level[level_type]) + '\n')

        return extractfile

    def extract(self, parameters, level_type, cmd='dap3_dev'):
        extractfile = self.requete(parameters, level_type)
        os.environ["DMT_DATE_PIVOT"] = self.date.strftime('%Y%m%d%H%M%S')
        print(os.environ["DMT_DATE_PIVOT"])
        os.system("{0:s} {1:d} {2:s}".format(cmd, self.ech, self.rqst))
        self.extractedfiles.append(extractfile)

        if os.path.exists(extractfile) and os.path.getsize(extractfile) > 0:
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
            for lvl_type in list_level.keys():
                parameters = list_parameters[lvl_type]
                if not self.extract(parameters, lvl_type) and lvl_type not in ['HAUTEUR', 'ISO_T']:
                    fileok = False

            if fileok:
                self.concatenate()
                return None
            else:
                if os.path.isfile(self.gribname):
                    os.remove(self.gribname)
                if self.member is not None:
                    return self.gribname + ' for member {0:d}'.format(self.member)
                else:
                    return self.gribname


if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend, args.cutoff, args.model)
    for domain in args.domain:
        missing_grib = list()
        workdir = os.path.join(args.workdir, domain)
        goto(workdir)
        for date in extract_period:
            if args.cutoff == 'prevision':
                for ech in range(0, 97, 6): 
                    if args.model == 'pearp':
                        ech = ech + 12
                        for member in range(35):
                            goto(os.path.join(workdir, nivologyseason(date), 'mb{0:03d}'.format(member)))
                            grib = ExtractGrib(args.model, args.cutoff, args.grid, domain, date, ech=ech, member=member)
                            result = grib.run()
                            if result is not None:
                                missing_grib.append(result)
                    else:
                        grib = ExtractGrib(args.model, args.cutoff, args.grid, domain, date, ech=ech)
                        result = grib.run()
                        if result is not None:
                            missing_grib.append(result)

            else:
                grib = ExtractGrib(args.model, args.cutoff, args.grid, domain, date, ech=6)
                result = grib.run()
                if result is not None:
                    missing_grib.append(result)
        goto(workdir)
        if len(missing_grib) > 0:
            with open('missing_grib', 'w') as f:
                for m in missing_grib:
                    f.write('{0:s}\n'.format(m))



