#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 25/01/2022

import os
import datetime
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

##########################################################################################
# Ce script extrait les précipitations journalières analysées par PATHERE sur le
# point des postes du réseau nivo-météo.
# Les données PANTHERES ont préalablement été extraites de Hendrix par l'outil
# 'lunairs' sur la serveur sotrtm33-sidev.
##########################################################################################

# Liste des coordonnées attendues par la commande dap3: lat_max, lat_min, lon_max, lon_min
coords = dict(
    alp = ['46875', '43125', '4500', '8500'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '41000', '8000', '10500'],
)

datebegin = datetime(2018, 12, 1, 6, 0)
dateend   = datetime(2019, 4, 30, 6, 0)

workdir = '/home/vernaym/workdir/evaluation_PANTHERE' 
                    
dist = lambda dx,dy: np.sqrt(dx**2+dy**2)

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

if __name__ == "__main__":

    extract_period = date_range(datebegin, dateend)

    panthere = pd.DataFrame(columns=['date', 'num_poste', 'rr_panthere'])
    goto(workdir)
    nivometeo = pd.read_csv('postes_nivometeo.csv', sep=';')
    for date in extract_period:
        print(date)
        if date.month in [1,2,3,4,11,12]: # Consider only month with nivometeo observations
            ficname = '{0:s}_010000_DATA.text'.format(date.strftime('%Y%m%d%H%M')) # 201904030600_010000_DATA.text
            fic = os.path.join(workdir, ficname)
            if os.path.exists(fic):
                radar = pd.read_csv(fic, sep=' ', comment='#', names=['lat', 'lon', 'rr'])
                def closest(row):
                    darr = dist(radar['lat']-row['poste_nivo.lat_dg'], radar['lon']-row['poste_nivo.lon_dg'])
                    idx = np.where(darr == np.amin(darr))[0][0]
                    return radar['rr'][idx]
                rr_panthere = nivometeo.apply(closest, axis=1)
                tmp = pd.DataFrame()
                tmp['rr_panthere'] = rr_panthere
                tmp['date']        = date
                tmp['num_poste']   = nivometeo['poste_nivo.num_poste']
                panthere = panthere.append(tmp, ignore_index=True, sort=False)
        else:
            print('Missing date {0:s}'.format(date.strftime("%Y%m%d%H")))

    panthere.set_index('date')
    goto(workdir)
    outname = 'PANTHERE_{0:s}_{1:s}.csv'.format(datebegin.strftime('%Y%m%d%H'), dateend.strftime('%Y%m%d%H'))
    panthere.to_csv(outname, index=False, sep=';')



