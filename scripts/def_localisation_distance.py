#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 10/11/2022

import os, sys
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import xarray as xr
import time

import argparse

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    domain = 'alp'  # default value
else:
    domain = sys.argv[1]

datadir = '/home/vernaym/These/DATA'
workdir = '/home/vernaym/workdir/ASSIMILATION/'
savedir = '/home/vernaym/These/figures/evaluation/as_pearome'

# Liste des coordonnées des domaines connus lat_max, lat_min, lon_max, lon_min
coords = dict(
    alp = ['46875', '43125', '4500', '8500'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '41000', '8000', '10500'],
    #GrandesRousses = ['45240', '44990', '6010', '6490'],
    GrandesRousses = ['45210', '45020', '6040', '6460'],  # TODO : modifier quand les bords du domaines seront inclus dans la localisation
    ange = ['45240', '44990', '6010', '6490']
)

def speedtest(function):
    def wrapper(*args, **kw):
        t1 = time.time()
        result = function(*args, **kw)
        t2 = time.time()
        print('The method {0:s} took {1:f}ms'.format(function.__name__, (t2-t1)*1000.0))
        return result
    return wrapper

@speedtest
def read_nivometeo_obs():

    nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['date'], header=0,
            names=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'massif', 'obs', 'unused'],
            usecols=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'obs'],
            dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'obs':float},
        )

    latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
    nivometeo = nivometeo.loc[(nivometeo['lat']>=latmin) & (nivometeo['lat']<=latmax) & (nivometeo['lon']>=lonmin) & (nivometeo['lon']<=lonmax)]  # Select area
    nivometeo.date = nivometeo.date + pd.Timedelta("1d6h")   #BDClim extraction for date ymd is the observation from ymd6h to ym(d+1)6h
    nivometeo.set_index(['num_poste','date'], inplace=True)

    return nivometeo.to_xarray()

@speedtest
def read_simu(filename):

    if os.path.exists(filename):
        simulation =  xr.open_dataset(filename)
    else:
        print(f'ERROR : file {filename} does not exist')
        sys.exit(1)

    if 'hourly' in filename:
        # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
        # Problem : the xarray tools to do that allows only accumulations between
        # 0h and 23h.
        # solution : shift time serie by 7h, compute 24h accumulations and
        # shift back !
        #print(simulation.time.data[127*24-1])
        #print(simulation.time.data[126*24])
        simulation['time'] = simulation.time-np.timedelta64(7, 'h')
        #toto=simulation.loc[{'lat':44.99, 'lon':6.01, 'member':1}].rr.data
        simulation = simulation.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
        #tata=simulation.loc[{'lat':44.99, 'lon':6.01, 'member':1}].rr.data
        #print(toto[126*24:127*24-1])
        #print(np.sum(toto[126*24:127*24-1]), tata[126])
        simulation['time'] = simulation.time+np.timedelta64(30, 'h')
        #print(simulation.time.data[126])
        #import pdb
        #pdb.set_trace()

    return simulation

@speedtest
def read_raw_ensemble():
    #filenames = [os.path.join(datadir, f'aspearome_{mb:03d}_2021073106_2022070106_GrandesRousses_daily.nc') for mb in range(1,17)]
    filenames = [os.path.join(datadir, f'aspearome_{mb:03d}_2021102806_2022060206_alp_hourly.nc') for mb in range(1,17)]
    #raw = xr.open_mfdataset(filenames, combine='nested', concat_dim='member').compute().clip(0)
    raw = xr.open_mfdataset(filenames, combine='nested', concat_dim='member', chunks={'time': 24})  # Setting chunks is critical (read the doc !)
    raw['member']=np.arange(1,17)
    # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
    # Problem : the xarray tools to do that allows only accumulations between
    # 0h and 23h.
    # solution : shift time serie by 7h, compute 24h accumulations and
    # shift back !
    raw['time'] = raw.time-np.timedelta64(7, 'h')
    raw = raw.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
    raw['time'] = raw.time+np.timedelta64(30, 'h')
    #raw = raw.compute().clip(0)  # TODO : try without computing (seems towork !)
    raw = raw.clip(0)  # TODO : try without computing (seems towork !)
    raw = raw.transpose('lat', 'lon', 'time', 'member')  # transpose data to put dimension in the same order as assimilated fields

    return raw

@speedtest
def read_antilope():
    filename = 'ANTILOPEQ_2021103000_2022060200_alp.nc'
    if not os.path.exists(os.path.join(datadir, filename)):
        filename = 'ANTILOPEH_2021103000_2022060200_alp.nc'
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    if filename.startswith('ANTILOPEH'):
        # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
        # Problem : the xarray tools to do that allows only accumulations between
        # 0h and 23h.
        # solution : shift time serie by 7h, compute 24h accumulations and
        # shift back !
        antilope['time'] = antilope.time-np.timedelta64(7, 'h')
        t1 = time.time()
        antilope = antilope.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
        t2 = time.time()
        print(f'Computing daily antilope took {(t2-t1)*1000}.ms')
        antilope['time'] = antilope.time+np.timedelta64(30, 'h')
        antilope.to_netcdf(os.path.join(datadir, 'ANTILOPEQ_2021103000_2022060200_alp.nc'))

    return antilope

def rank_histogram(simu, obs, product, ax, *args):
    maxsim = np.amax(simu, 1)
    # On retire les point/dates où tout est à 0 (le rang d el'observation n'est alors pas définit)
    #simu = simu[(~np.isnan(obs)) & ((obs>0) | (maxsim>0))]
    #obs = obs[(~np.isnan(obs)) & ((obs>0) | (maxsim>0))]
    simu = simu[(~np.isnan(obs)) & (obs>0)]
    obs = obs[(~np.isnan(obs)) & (obs>0)]
#    if np.shape(simu)[1]>17:
#    if product == 'raw1':
#        import pdb
#        pdb.set_trace()
    position = np.array([])
    for idx, value in enumerate(obs):
        #position = np.append(position, np.searchsorted(np.sort(simu[idx]), value, side='right'))
        position = np.append(position, np.searchsorted(np.sort(simu[idx]), value))
    count, bins = np.histogram(position, bins=17)
    ax.hist(bins[:-1], bins, weights=count)
    #ax.hist(position, bins=range(np.shape(simu)[1]))
    #ax.hist(position, bins=range(17))

def get_data():

    def nearest(array, value):
        """ Find element of "array" the closer to 'value' """
        # Security to ensure that the station is within the simulated domain.
        if np.abs(array - value).data.min() < 0.1:
            return float(array[np.abs(array - value).argmin()].data)
        else:
            print(f'ERROR : no corresponding pixel found for value {value}')
            import pdb
            pdb.set_trace()

    data = read_nivometeo_obs()
    # Remove time dimension from metadata :
    data['lon']=np.max(data.lon, axis=1)
    data['lat']=np.max(data.lat, axis=1)
    data['alti']=np.max(data.alti, axis=1)
    data['nom']=np.max(data.nom, axis=1)

    dates = data.date

    antilope = read_antilope()
    antilope = antilope.loc[{'time':dates}]

    raw = read_raw_ensemble()
    raw = raw.loc[{'time':dates}]
    t1 = time.time()
    raw = raw.compute()
    t2 = time.time()
    print(f'Computing raw ensemble took {(t2-t1)*1000}.ms')

    raw_interp = raw.interp(lon=antilope.lon, lat=antilope.lat).clip(0)  # Avoid <0 precipitation values
    data['member'] = np.arange(1,17)

#    extract_data = dict(obs=np.array([[]]), antilope=np.array([[]]), raw=np.array([]), raw_interp=np.array([]),
#            raw1=np.array([]), raw2=np.array([]), raw3=np.array([]), raw4=np.array([]), raw5=np.array([]),
#            raw7=np.array([]), raw10=np.array([]), raw20=np.array([]),
#        )
    extract_data = dict()
    def add_data(product, array, axis=None):
        if product in extract_data.keys():
            if product == 'antilope':
                extract_data[product] = np.append(extract_data[product], array)
            else:
                print(product)
#                if product == 'raw20':
#                    import pdb
#                    pdb.set_trace()
                # TODO : probleme avec la localisation : pour les postes trop proches des bords, le nombre de pixel est < et
                # la concaténation ne peut être faite.
                # TODO : vérifier ce quon veut obtenir au final pour tracer les histogrammes de rang... (150 dates*54 postes, N membres)
                # ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 26240 and the array at index 1 has size 26896
                try:
                    extract_data[product] = np.append(extract_data[product], array, axis=0)
                except:
                    import pdb
                    pdb.set_trace()
        else:
            extract_data[product] = array


    liste_postes = np.array([])
    for idx, num_poste in enumerate(data.num_poste.data):
        print(f'Station {idx+1}/{len(data.num_poste.data)}')
        print(num_poste)
        tmp = data.loc[{'num_poste':num_poste}]
        lat = tmp.lat
        lon = tmp.lon
        obs = tmp.obs.data
        alti = tmp.alti.data.max()
        #if len(obs[~np.isnan(obs)]) >= 100:  # Filter stations with too few observations
        if len(obs[~np.isnan(obs)]) >= 100 and num_poste != 6073405:  # WARNING : verrue pour virer le poste trop proche du bord du domaine
            #np.append(extract_data['obs'], obs)
            add_data('obs', obs)
            liste_postes = np.append(liste_postes, num_poste)
            #np.append(extract_data['antilope'], antilope.sel({'lat':nearest(antilope.lat, lat), 'lon':nearest(antilope.lon, lon)}).rr.data)
            add_data('antilope', antilope.sel({'lat':nearest(antilope.lat, lat), 'lon':nearest(antilope.lon, lon)}).rr.data)
            raw_lat = nearest(raw.lat, lat)
            raw_lon = nearest(raw.lon, lon)
            #np.append(extract_data['raw'], raw.sel({'lat':raw_lat, 'lon':raw_lon}).rr.data)
            add_data('raw', raw.sel({'lat':raw_lat, 'lon':raw_lon}).rr.data)
            model_lat = nearest(raw_interp.lat, lat)
            model_lon = nearest(raw_interp.lon, lon)
            #np.append(extract_data['raw_interp'], raw_interp.sel({'lat':model_lat, 'lon':model_lon}).rr.data, axis=1)
            add_data('raw_interp', raw_interp.sel({'lat':model_lat, 'lon':model_lon}).rr.data)
            sel = raw_interp
            #for dloc in [50, 25, 10, 7, 5, 4, 3, 2, 1]:  # decreasing order important for computing efficiency
            for dloc in [10, 7, 5, 4, 3, 2, 1]:  # decreasing order important for computing efficiency
                idy = np.searchsorted(sel.lat, model_lat)
                idx = np.searchsorted(sel.lon, model_lon)
                localisation_lat = np.array(
                            [sel.lat.data[idy+dlat] if idy+dlat>=0 and idy+dlat<len(sel.lat) else np.nan
                                for dlat in range(-dloc,dloc+1)]
                        )
                localisation_lat = localisation_lat[~np.isnan(localisation_lat)]
                localisation_lon = np.array(
                            [sel.lon.data[idx+dlon] if idx+dlon>=0 and idx+dlon<len(sel.lon) else np.nan
                                for dlon in range(-dloc,dloc+1)]
                        )
                localisation_lon = localisation_lon[~np.isnan(localisation_lon)]
                sel = sel.sel({'lat':localisation_lat, 'lon':localisation_lon})
                #np.append(extract_data[f'raw{dloc}'], sel.rr.data.transpose((2,0,1,3)).reshape(sel.rr.data.shape[2], -1), axis=1)
                # TODO : vérifier la transposition pour obtenir un vecteur (150 dates, 16 membres * Nlat * Nlon)
                add_data(f'raw{dloc}', sel.rr.data.transpose((2,0,1,3)).reshape(sel.rr.data.shape[2], -1))


    for product in extract_data.keys():
        if product not in ['obs', 'antilope']:
            print(product)
            # TODO : problème lorsque la distance de loc est trop grande et que certains points se retrouvent en "bordure"
            # de domaine : les arrays dans extract_data sont de longueur différente pour chaque station
#            if product == 'raw7':
#                import pdb
#                pdb.set_trace()

            #shape = np.shape(np.array(extract_data[product]))
            #newshape=(shape[0]*shape[1], shape[2])
            fig,ax = plt.subplots()
            # TODO : vérifier que l'ordre des données est bien conservé pour comprarer les simus aux bonnes obs !
            #rank_histogram(np.array(extract_data[product]).reshape(newshape), np.array(extract_data['obs']).flatten(), product, ax)
            rank_histogram(extract_data[product], extract_data['obs'], product, ax)
            fig.savefig(f'{savedir}/rank_histogram_{product}.pdf', format='pdf')

if __name__ == '__main__':

    data = get_data()

