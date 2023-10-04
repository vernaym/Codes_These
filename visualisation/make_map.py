#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 21/04/2023

# Script pour générer une carte contenant l'ensemble des informations disponibles
# sur les précipitations en 24h d'une date donnée.

import os, sys
import datetime
import numpy as np
import xarray as xr
import pandas as pd

import footprints
from vortex import toolbox
import cen,iga
toolbox.active_now = True

from bronx.stdtypes.date import Date, Period

from These.radar.Preprocessing_ANTILOPE import AntilopePreprocessing
from These.visualisation.plot_precip_map import PrecipitationAnalysis


def usage():
    print("USAGE make_map.py date")
    print("format de la date : YYYYMMDD (précipitations de YYYYMMD-1 6h à YYYYMMDD6H")
    sys.exit(1)

try:
    date = Date(sys.argv[1])
except Exception as e:
    usage()
    raise e

datebegin = date.replace(hour=6)
dateend   = datebegin + Period(hours=24)

#datadir = '/home/vernaym/workdir/visualisation'
datadir = '/home/vernaym/extraction_obs'  # On sxcen

#domain = 'alp'

def get_antilope(domain, obs_auto=None):

    filename = f'ANTILOPE_{domain}_{date.ymd}.nc'
    if not os.path.exists(filename):
        toolbox.input(
            #role           = 'Observations',
            #geometry       = self.conf.geometry[self.conf.vconf],
            #suite          = self.conf.suite,
            kind           = 'Observation',
            date           = date.ymdh,
            begindate      = f'{datebegin.replace(hour=7).ymdh}',
            enddate        = dateend.ymdh,
            local          = filename,
            model          = 'antilope',
            hostname       = 'sotrtm35-sidev.meteo.fr',
            username       = 'vernaym',
            tube           = 'ftp',
            remote         = f'/home/mrns/vernaym/extraction_antilope/{domain}/ANTILOPEH_[begindate]_[enddate]_{domain}.nc',  # ANTILOPEH_2023042007_2023042106_alp.nc
            unknown        = True,
            #cutoff         = 'assimilation',
            #now            = True,
        )

    pp = AntilopePreprocessing(date, domain, filename)
    antilope = pp.run(obs_auto=obs_auto)

    return antilope

def get_nivometeo():
    fic_score = os.path.join(datadir, f'obs_nivometeo_daily_RR_{datebegin.ymd}_{datebegin.ymd}.csv')
    if os.path.exists(fic_score):
        nivometeo = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
                dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'massif':int, 'rr': float, 'reseau_poste': int}, na_values=['--'])
        #nivometeo = nivometeo.loc[nivometeo["date"]==np.datetime64(date)+np.timedelta64(1,'D')]  # Useless in real time
        nivometeo = nivometeo.loc[nivometeo["date"]==np.datetime64(date)]  # Useless in real time
        if len(nivometeo)>0:
            return nivometeo
        else:
            return None
    else:
        return None

def get_obs_auto(domain):
    #fic_score = os.path.join(datadir, f'auto.data')
    fic_score = os.path.join(datadir, f'obs_horaires_RR_{datebegin.ymd}_{dateend.ymd}_{domain}.csv')  # obs_horaires_RR.data
    auto = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
            dtype={'num_poste':int, 'nom':str, 'lat':float, 'lon':float, 'alti':int, 'rr': float, 'reseau_poste': int}, na_values=['--'])
    auto=auto.loc[(auto["date"]>np.datetime64(datebegin)) & (auto["date"]<=dateend)]
    auto = auto[~np.isnan(auto['rr'])]

    #df = auto.set_index(['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date']).sort_index()
    #df = df.assign(date=newdates).drop(columns=['date'])  # Replace date column
    auto = auto.set_index(['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date']).sort_index()
    auto=auto.groupby(['num_poste', 'nom', 'lat', 'lon', 'alti', 'reseau_poste']).sum()
    auto = auto.reset_index()

    return auto

def get_safran(domain):

    filename = f'SAFRAN_{domain}_{datebegin.ymd}.nc'  #TODO : donner un nom plus explicite

    toolbox.input(
        role           = 'Ana_massifs',
        kind           = 'MeteorologicalForcing',
        vapp           = 's2m',
        vconf          = domain,
        source_app     = 'arpege',
        source_conf    = '4dvarfr',
        cutoff         = 'assimilation',
        local          = filename,
        experiment     = 'oper',
        block          = 'massifs',
        geometry       = domain,
        nativefmt      = 'netcdf',
        model          = 'safran',
        date           = (dateend+Period(hours=3)).ymd6h,  # Trouver une solution plus élégante
        datebegin      = datebegin.ymd6h,
        dateend        = dateend.ymd6h,
        namespace      = 'vortex.multi.fr',
        fatal          = False,
    ),

    if os.path.exists(filename):
        safran = xr.open_dataset(filename)
        #safran = safran.where(safran.ZS==1500., drop=True)
        safran['rr'] = (safran['Rainf']+safran['Snowf'])*3600.
        dates = pd.date_range(datebegin+Period(hours=1), dateend, freq='1H')
        safran = safran.loc[{'time':dates}]
        #safran['rr'] = safran.where((safran.time>np.datetime64(datebegin)) & (safran.time<=np.datetime64(dateend)), drop=True)  # This adds time dimension to ZS variable
        safran['rr'] = safran['rr'].sum('time')
        return safran
    else:
        return None

# 1. Read nivometeo observations
nivometeo = get_nivometeo()

# TODO : plot all data on the same map !

antilope = dict()
safran = dict()
auto = dict()
for domain in ['alp', 'pyr']:

    # 2. Read automatic observations
    auto[domain] = get_obs_auto(domain)

    # 3. Récupération de ANTILOPE depuis sotrtm35-sidev
    antilope[domain] = get_antilope(domain, auto[domain])

    # 4. Récupération de l'analyse SAFRAN oper de 9h
    safran[domain] = get_safran(domain)
    #safran = None


myplot = PrecipitationAnalysis(date, antilope=antilope, nivometeo=nivometeo, auto=auto, safran=safran, var='analysis')  # Plot corrected field
myplot.plot()
myplot.save()
myplot = PrecipitationAnalysis(date, antilope=antilope, nivometeo=nivometeo, auto=auto, safran=safran, var='rr')  # Plot raw ANTILOPE field
myplot.plot()
myplot.save()

