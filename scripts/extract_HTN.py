#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Extraction des données d'observation depuis la BDCLIM
# Matthieu Lafaysse 10 sept 2014
# Modified Léo Viallon-Galinier 22 oct 2021
# Script simplifié pour obtenir un format obs.csv depuis la BDCLIM

import argparse

from snowtools.scripts.extract.obs.bdquery import question

parser = argparse.ArgumentParser(
    description="""
    Read HTN (snow height) observations from BDCLim
    for all available stations: both Nivoses and
    "nivo-meteo" network.
    """
)

parser.add_argument("date_min", help="Start date")
parser.add_argument("date_max", help="End date")
parser.add_argument("station", help="Station number or name (if present in mapping dict)")
args = parser.parse_args()

datedeb = args.date_min
datefin = args.date_max

num_poste_map = dict(
    GalibierNivose    = '05079402',
    LacBlancNivose    = '38191403',
    RochillesNivose   = '73306401',
    HuezNivometeo     = '38191002',  # ALPE-D'HUEZ - 1860m
    #HuezNivometeo     = '38191400',  # L Alpe d Huez (SATA) - 1860m --> poste nivometeo
)

if isinstance(args.station, str):
    num_poste = num_poste_map[args.station]
else:
    num_poste = args.station

outname = f'HTN_{args.station}_{datedeb}_{datefin}.obs'
outmode = 'w'

# II.1.1 construction de la question pour les postes nivo pour la HTN
# ---------------------------------------------------------------------------
# On prend toutes les heures

table = 'H_NIVO'
listvar = ["to_char(dat,'YYYY-MM-DD-HH24-MI')", f"to_char({table}.num_poste, 'fm00000000')", "lat_dg", "lon_dg", 'alti', "ALTI_LPNX",
          "neigetot"]
header = ['dat', 'num_poste', 'lat', 'lon', 'alt', 'lpn', 'neigetot']
if num_poste == '38191002':
    table = 'H'
    listvar = ["to_char(dat,'YYYY-MM-DD-HH24-MI')", f"to_char({table}.num_poste, 'fm00000000')", "lat_dg", "lon_dg", 'alti', "neigetot"]
    header = ['dat', 'num_poste', 'lat', 'lon', 'alt', 'neigetot']

# Postes NIVOSE
question = question(
    listvar=listvar,
    table=table,
    listjoin=[f'POSTE_NIVO ON {table}.NUM_POSTE = POSTE_NIVO.NUM_POSTE and type_nivo != 2'],
    period=[datedeb, datefin],
    listorder=['dat', f'{table}.num_poste'],
    listconditions=[f"to_char(dat,'HH24') = '06' and {table}.NUM_POSTE = '{num_poste}'"]
)
question.run(outputfile=outname, header=header, mode=outmode)
