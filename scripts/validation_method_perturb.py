
import os
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rio
#from pyproj import Proj, transform
# https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
from pyproj import Transformer

import matplotlib.pyplot as plt

from snowtools.scripts.post_processing import common_dict

import cen
from vortex import toolbox
toolbox.active_now = True

datadir = '/home/vernaym/DATA'

df = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20210801_20220801.csv'), sep=';')
df = df.loc[(df['poste_nivo.lat_dg'] >= 44.99) & (df['poste_nivo.lat_dg'] <= 45.24) &
        (df['poste_nivo.lon_dg'] >= 6.010) & (df['poste_nivo.lon_dg'] <= 6.49)]

# df['ndays'] = 1
# ref_cumul = df.groupby('Q.num_poste').agg({'Q.rr': sum, 'poste_nivo.lat_dg': min,
#     'poste_nivo.lon_dg': min, 'poste_nivo.alti': min, 'ndays': sum})
# x_coords = ref_cumul['poste_nivo.lon_dg'].values
# y_coords = ref_cumul['poste_nivo.lat_dg'].values

# inProj  = Proj(init="EPSG:4326")
# outProj = Proj(init='EPSG:2154')
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154")


def nearest(value, array):
    """ Find the closest element of 'array' to 'value'. """
    return array[np.abs(array - value).argmin()]


product_map = dict(
    SAFRAN       = 'SAFRAN_perturb',
    RS27_perturb = 'ASANTILOPE_perturb',
    RS27_sorted  = 'SRS',
)

colors_map  = common_dict.colors_map

fig, ax = plt.subplots()
axmin = 0
axmax = 150
ax.plot([axmin, axmax], [axmin, axmax], linestyle='-', color='k')

xpids = ['SAFRAN', 'RS27_perturb', 'RS27_sorted']
for xpid in xpids:

    if xpid == 'SAFRAN':
        datebegin  = '2021080106'
    else:
        datebegin = '2021080206'

    product = product_map[xpid]

    toolbox.input(
        datebegin  = datebegin,
        dateend    = '2022080106',
        date       = '[dateend]',
        experiment = f'{xpid}@vernaym',
        geometry   = 'GrandesRousses250m',
        kind       = 'MeteorologicalForcing',
        namespace  = 'vortex.multi.fr',
        filename   = f'mb[member]/FORCING_{xpid}.nc',
        block      = 'meteo',
        vapp       = 'edelweiss',
        vconf      = '[geometry:tag]',
        cutoff     = 'assimilation',
        member     = [mb for mb in range(17)],
        namebuild  = 'flat@cen',
    )

    filenames = [f'mb{member:03d}/FORCING_{xpid}.nc' for member in range(17)]
    ds = xr.open_mfdataset(filenames, concat_dim='member', combine='nested', chunks='auto')
    precip = (ds.Rainf + ds.Snowf) * 3600.
    # precip = precip.chunk({'x': 10, 'y': 10, 'time': len(precip.time), 'member': 17})
    precip = precip.chunk({'x': 10, 'y': 10})
    # precip.rio.write_crs("EPSG:2154", inplace=True)
    # precip_latlon = precip.rio.reproject("EPSG:4326")
    # cumul  = precip.sum('time')
    # cumul.rio.write_crs("EPSG:2154", inplace=True)
    # cumul_latlon = cumul.rio.reproject("EPSG:4326")

    label = True
    for num_poste in df['Q.num_poste'].unique():
        tmp = df[df['Q.num_poste'] == num_poste]
        nom = tmp['poste_nivo.nom_usuel'].values[0]

        # Number of days between first and last obs
        datebegin = pd.to_datetime(tmp['Q.dat'].sort_values().values[0])
        dateend   = pd.to_datetime(tmp['Q.dat'].sort_values().values[-1])
        # Number of days between first and last obs
        ndays = (dateend - datebegin).days
        # Count number of rows in 'tmp'
        nobs = tmp.shape[0]
        print(num_poste, ndays, nobs)

        # Consider stations with at least 90% of observation over a 3-months period
        if nobs > ndays * 0.9 and nobs > 90:
            simu = precip.sel(time=slice(datebegin, dateend))

            x = float(tmp['poste_nivo.lon_dg'].unique()[0])
            y = float(tmp['poste_nivo.lat_dg'].unique()[0])
            # X, Y = transform(inProj, outProj, x, y)
            X, Y = transformer.transform(x, y)

            # simu = simu.interp(x=X, y=Y, method='nearest')
            simu = simu.sel(x=nearest(X, simu.x.data), y=nearest(Y, simu.y.data))
            simu = simu.compute()
            cumul = simu.sum('time')

            spread_simu = float(cumul.std('member').data)
            mean_simu = float(cumul.mean('member').data)
            obs = float(tmp['Q.rr'].sum())

            if np.abs(mean_simu - obs) > axmax:
                print('DBUG ', num_poste, nom)
                print('mean_simu, obs = ', mean_simu, obs)

            if label:
                plt.plot(np.abs(mean_simu - obs), spread_simu, marker='D', color=colors_map[product], label=product,
                        linestyle='')
                label = False
            else:
                plt.plot(np.abs(mean_simu - obs), spread_simu, marker='D', color=colors_map[product],
                        linestyle='')

ax.set_xlim([axmin, axmax])
ax.set_ylim([axmin, axmax])
ax.set_xlabel('Ensemble mean error (kg/m²)')
ax.set_ylabel('Ensemble spread (kg/m²)')
ax.legend(loc='lower right')
suffix = '_'.join(xpids)
fig.savefig(f'SpreadSkill_{suffix}.pdf')
