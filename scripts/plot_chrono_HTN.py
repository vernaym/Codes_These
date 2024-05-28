import os
import xarray as xr
import argparse
import pandas as pd
from snowtools.scripts.extract.vortex import vortexIO as io
import matplotlib.pyplot as plt

coords = dict(
    Galibier = dict(
        xx        = 965767.64,
        yy        = 6445415.30,
    ),
    LacBlanc = dict(
        xx        = 944584.42,
        yy        = 6452410.74,
    ),
)

parser = argparse.ArgumentParser()

parser.add_argument("datebegin", help="Start date")
parser.add_argument("dateend", help="End date")
parser.add_argument("point", help="Name of the point to plot", choices=coords.keys())
args = parser.parse_args()

pleiades_map = {
    '2018': dict(dates=['2018012312', '2018031612'], geometry='Lautaret250m'),
    '2019': dict(dates=['2019051312'], geometry='Huez250m'),
}
datebegin = args.datebegin
dateend   = args.dateend
point     = args.point

xx = coords[point]['xx']
yy = coords[point]['yy']

year = dateend[:4]
dates_pleiades = pleiades_map[year]['dates']
geometry = pleiades_map[year]['geometry']

io.get_pro(datebegin=datebegin, dateend=dateend, xpid='SAFRAN_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_SAFRAN.nc', vapp='edelweiss')
io.get_pro(datebegin=datebegin, dateend=dateend, xpid='ANTILOPE_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_ANTILOPE.nc', vapp='edelweiss')
io.get_pro(datebegin=datebegin, dateend=dateend, xpid='RS27_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_RS27.nc', members='0-0-1', vapp='edelweiss')
io.get_snow_obs_date(xpid='CesarDB_AngeH', geometry=geometry, date=dates_pleiades,
        vapp='Pleiades', filename='Pleiades_[date:ymdh].nc')

safran = xr.open_dataarray('PRO_SAFRAN.nc')
antilope = xr.open_dataarray('PRO_ANTILOPE.nc')
rs27 = xr.open_dataarray('PRO_RS27.nc')
time = safran.time

fig, ax = plt.subplots(figsize=(14, 4))
plt.plot(time, safran, label='SAFRAN', color='red')
plt.plot(time, antilope, label='ANTILOPE', color='blue')
plt.plot(time, rs27, label='AS-ANTILOPE', color='green')
if os.path.exists('NIVOSE.obs'):
    #obs = pd.read_csv('NIVOSE.obs', sep=';', parse_dates=['dat'])
    obs = pd.read_csv('NIVOSE.obs', sep=',', parse_dates=['dat'])
    plt.plot(obs.dat.values, obs.neigetot.values / 100, color='k', label='Nivose')
if os.path.exists('NIVOSE2.obs'):
    #obs = pd.read_csv('NIVOSE.obs', sep=';', parse_dates=['dat'])
    obs = pd.read_csv('NIVOSE2.obs', sep=',', parse_dates=['dat'])
    plt.plot(obs.dat.values, obs.neigetot.values / 100, color='k')

legend = True
for date in dates_pleiades:
    pleiades = xr.open_dataarray(f'Pleiades_{date}.nc')
    htn = pleiades.interp({'x': xx, 'y': yy}, method='nearest')
    plt.vlines(pd.to_datetime(date, format='%Y%m%d%H'), 0, 3.5, color='k', linestyle=':')
    if legend:
        plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k',
                label='Pleiades')
    else:
        plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k')
    legend = False
plt.legend()
plt.tight_layout()
plt.savefig('Chrono.pdf')
