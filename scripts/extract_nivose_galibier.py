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
parser.add_argument("-o", "--output", help="Output file. If none selected, produce NIVOMETEO.obs and NIVOSE.obs files",
                    default=None, dest='output')
args = parser.parse_args()

datedeb = args.date_min
datefin = args.date_max

if args.output:
    nivose_obs_fn = args.output
    nivomto_obs_fn = args.output
    nivomto_mode = 'a'
else:
    nivose_obs_fn = 'NIVOSE.obs'
    nivomto_obs_fn = 'NIVOMETEO.obs'
    nivomto_mode = 'w'

# II.1.1 construction de la question pour les postes nivo pour la HTN
# ---------------------------------------------------------------------------
# On prend toutes les heures

# Postes NIVOSE
question = question(
        listvar=["to_char(dat,'YYYY-MM-DD-HH24-MI')", "to_char(h.num_poste, 'fm00000000')", "neigetot"],
        table='H',
        listjoin=['POSTE_NIVO ON H.NUM_POSTE = POSTE_NIVO.NUM_POSTE and type_nivo = 3'],
        period=[datedeb, datefin],
        listorder=['dat', 'h.num_poste'],
        listconditions=["to_char(dat,'HH24') = '06' and H.NUM_POSTE = '05079402'"]
        )
question.run(outputfile=nivose_obs_fn, header=['dat', 'num_poste', 'neigetot'], mode=nivomto_mode)

