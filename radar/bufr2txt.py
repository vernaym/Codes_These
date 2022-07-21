import os, sys
import datetime
import glob

# Script de génération de cumuls 24h de la lame d'eau PANTHERE à partir
# de lames d'eau 5 minutes.
# A lancer sur sotrtm33-sidev
# Extraction des lames d'eau 5 minutes avec la commande sur sotrtm33-sidev :
# ./lunerad -id  CMF.PAN -d1 201904010600 -d2 201905010600

datebegin = sys.argv[1]
dateend   = sys.argv[2]

#date = datetime.datetime(2018, 12, 1, 6, 0)
date = datetime.datetime.strptime(datebegin, '%Y%m%d%H%M')
while date <= datetime.datetime.strptime(dateend, '%Y%m%d%H%M'):
    os.system('./bufr2txt -unit U_MM -id CMF.PAN -d1 {0:s} -f txt -cum 1440 -w'.format(date.strftime('%Y%m%d%H%M')))
    print('./bufr2txt -unit U_MM -id CMF.PAN -d1 {0:s} -f txt -cum 1440 -w'.format(date.strftime('%Y%m%d%H%M')))
    old_date = date - datetime.timedelta(days=1)
    for f in glob.glob('/home/mrns/vernaym/tmp/RADAR/COMPOSIT_ELLIPSO/LAME_EAU/{0:s}*.bfr'.format(old_date.strftime('%Y%m%d'))):
        #print('DBUG ',f)
        os.remove(f)
    date = date + datetime.timedelta(days=1)
