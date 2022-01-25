#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 07/03/2019

import os
import datetime
from datetime import datetime, timedelta

import argparse

#RQST/#NFIC ${FILEOUT}/#MOD ANTILOPEJP1/#PARAM PRECIP/#Z_REF FRANXL1S100/#Z_AXE HORIZONTAL
#D_LST ${dateheure}/#T_LST 60
#L_TYP SOL/#FORM GRIB2_C_MAX



# Identifiant du modèle dans la BDAP
model_id = 'ANTILOPEQJP1'
#model_id = 'ANTILOPEQ'
# Identifiant de la grille
grid = 'FRANXL1S100'
# Indentifiant du paramètre
parameter = 'PRECIP'
# Identifiant du niveau
level = 'SOL'
# Identifiant du reseau
leadtime = 6
# Echeance
ech = 24
# Format
fmt = 'GRIB2_C_MAX'



# Liste des coordonnées attendues par la commande dap3: lat_max, lat_min, lon_max, lon_min
coords = dict(
    alp = ['46875', '43125', '4500', '8500'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '40750', '8000', '11500'],
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
    parser.add_argument('-m', '--model', help='Model from which the data must be extracted', choices=['ANTILOPEQ', 'ANTILOPEQJP1'], default='ANTILOPEQ')
    parser.add_argument('-g', '--grid', help='BDAP grid name from which to extract data', default='FRANXL1S100')
    parser.add_argument('-v', '--vortex', action='store_true', help='Store generated files on a vortex archive store')
#    parser.add_argument('-r', '--replace', action = 'store_true', help='Replace existing files')
#    parser.add_argument('-t', '--tar', action='store_true', help='Tar generated files (for reanalaysis applications')

    args = parser.parse_args()

    args.datebegin = get_date(args.datebegin)
    if args.dateend:
        args.dateend = get_date(args.dateend)
    else:
        args.dateend = args.datebegin + timedelta(hours=24)

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
                    date = datetime.strptime(a_string, '%Y%m%d').replace(hour=0, minute=0, second=0, microsecond=0)
                except ValueError:
                    try:
                        date = datetime.strptime(a_string, '%y%m%d').replace(hour=0, minute=0, second=0, microsecond=0)
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

    def __init__(self, model, grid, domain, rundate):
        
        self.model          = model
        self.domain         = domain
        self.grid           = grid.upper()
        self.coords         = coords[domain]
        self.date           = rundate
        self.gribname       = '{0:s}_{1:s}.grib'.format(self.model, self.date.strftime('%Y%m%d%H'))
        self.extractedfiles = list()
        
    def requete(self, parameters, level_type):
        
        self.rqst = 'requete'.format(level_type)
        extractfile = '{0:s}_{1:s}.grib'.format(self.model, level_type)
        f = open(self.rqst, "w")
        f.write('#RQST\n')
        f.write('#NFIC {0:s}\n'.format(extractfile))
        f.write('#MOD {0:s}\n'.format(self.model))
        f.write('#PARAM {0:s}\n'.format(parameter))
        f.write('#Z_REF {0:s}\n'.format(self.grid))
        if coords is not None:
            f.write("#Z_EXTR INTERPOLATION\n")
            f.write('#Z_GEO ' + ' '.join(coords[self.domain]) + '\n')
            f.write('#Z_STP ' + ' '.join(dl) + '\n')
        f.write('#L_TYP {0:s}\n'.format(level))
        
        return extractfile
                
    def extract(self, parameters, level, cmd='dap3_dev'):
        extractfile = self.requete(parameters, level)
        os.environ["DMT_DATE_PIVOT"] = self.date.strftime('%Y%m%d%H%M%S')
        print(os.environ["DMT_DATE_PIVOT"])
        os.system("{0:s} {1:d} {2:s}".format(cmd, ech, self.rqst))
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
            if not self.extract(parameter, level):
                fileok = False
            
            if fileok:
                #self.concatenate()
                return None
            else:
                if os.path.isfile(self.gribname):
                    os.remove(self.gribname)
                    return self.gribname
            
        
if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend)
    for domain in args.domain:
        missing_grib = list()
        workdir = os.path.join(args.workdir, domain)
        goto(workdir)
        for date in extract_period:
            grib = ExtractGrib(args.model, args.grid, domain, date)
            result = grib.run()
            if result is not None:
                missing_grib.append(result)

        if len(missing_grib) > 0:
            with open('missing_grib', 'w') as f:
                for m in missing_grib:
                    f.write('{0:s}\n'.format(m))



