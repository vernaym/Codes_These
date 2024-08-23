import os
import numpy as np
import xarray as xr
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from snowtools.scripts.extract.vortex import vortexIO as io
import snowtools.tools.xarray_preprocess as xrp

coords = dict(
    Galibier = dict(
        xx        = 965767.64,
        yy        = 6445415.30,
    ),  # 2559m
    LacBlanc = dict(
        xx        = 944584.42,
        yy        = 6452410.74,
    ),  # 2720m
    NivometeoHuez = dict(
        xx        = 942705.64,
        yy        = 6447916.82,
    ),  # 1860m
    RochillesNivose = dict(
        xx        = 972852.5,
        yy        = 6448853.87
    ),  # 2444m
)

parser = argparse.ArgumentParser()

parser.add_argument("datebegin", help="Start date")
parser.add_argument("dateend", help="End date")
parser.add_argument("point", help="Name of the point to plot", choices=coords.keys())
args = parser.parse_args()

pleiades_map = {
    '2018': dict(dates=['2018012312', '2018031612'], geometry='GrandesRousses250m'),
    '2019': dict(dates=['2019051312'], geometry='Huez250m'),
    '2022': dict(dates=['2022022612', '2022050112'], geometry='GrandesRousses250m'),
}
datebegin = args.datebegin
dateend   = args.dateend
point     = args.point

xx = coords[point]['xx']
yy = coords[point]['yy']

year = dateend[:4]
dates_pleiades = pleiades_map[year]['dates']
geometry = pleiades_map[year]['geometry']

io.get_pro(datebegin=datebegin, dateend=dateend, xpid='SAFRAN_perturb', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_SAFRAN.nc', vapp='edelweiss')
io.get_pro(datebegin=datebegin, dateend=dateend, xpid='KRIGING_perturb', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_KRIGING.nc', vapp='edelweiss')
io.get_pro(datebegin=datebegin, dateend=dateend, xpid='ANTILOPE_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_ANTILOPE.nc', vapp='edelweiss')
if datebegin == '2021080106':
    datebegin_RS27 = '2021080207'
else:
    datebegin_RS27 = datebegin
io.get_pro(datebegin=datebegin_RS27, dateend=dateend, xpid='RS27_sorted_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_RS27.nc',vapp='edelweiss')

xpid_map = {
    '2017080106': 'CesarDB',
    '2018080106': 'CesarDB_AngeH',
    '2021080106': 'CesarDB',
}
io.get_snow_obs_date(xpid=xpid_map[datebegin], geometry=geometry, date=dates_pleiades,
        vapp='Pleiades', filename='Pleiades_[date:ymdh].nc')

safran = xr.open_dataset('PRO_SAFRAN.nc')
safran = safran.DSN_T_ISBA
kriging = xr.open_dataset('PRO_KRIGING.nc')
kriging = kriging.DSN_T_ISBA
antilope = xr.open_dataset('PRO_ANTILOPE.nc')
antilope = antilope.DSN_T_ISBA
rs27 = xr.open_dataset('PRO_RS27.nc')
rs27 = rs27.DSN_T_ISBA
time = safran.time

fig, ax = plt.subplots(figsize=(14, 4))
plt.plot(time, safran, label='SAFRAN', color='grey')
plt.plot(time, kriging, label='KRIGING', color='red')
plt.plot(time, antilope, label='ANTILOPE', color='blue')
plt.plot(rs27.time, rs27, label='AS-ANTILOPE', color='green')
deb = pd.Timestamp(year=int(datebegin[0:4]), month=int(datebegin[4:6]), day=int(datebegin[6:8]),  tz="UTC")
end = pd.Timestamp(year=int(dateend[0:4]), month=int(dateend[4:6]), day=int(dateend[6:8]),  tz="UTC")
if os.path.exists('HTN.obs'):
    #obs = pd.read_csv('NIVOSE.obs', sep=';', parse_dates=['dat'])
    obs = pd.read_csv('HTN.obs', sep=';', parse_dates=['dat'])
    obs = obs[(obs.dat>=deb) & (obs.dat<=end)]
    plt.plot(obs.dat.values, obs.neigetot.values / 100, color='k', label='In-situ observation')
if os.path.exists('HTN2.obs'):
    #obs = pd.read_csv('NIVOSE.obs', sep=';', parse_dates=['dat'])
    obs = pd.read_csv('HTN.obs', sep=';', parse_dates=['dat'])
    obs = obs[(obs.dat>=deb) & (obs.dat<=end)]
    plt.plot(obs.dat.values, obs.neigetot.values / 100, color='k')

legend = True
for date in dates_pleiades:
    pleiades = xr.open_dataset(f'Pleiades_{date}.nc')
    pleiades = xrp.preprocess(pleiades, mapping={'DSN_T_ISBA': 'HTN', 'DEP': 'HTN'})
    htn = pleiades.interp({'xx': xx, 'yy': yy}, method='nearest').HTN
    plt.vlines(pd.to_datetime(date, format='%Y%m%d%H'), 0, 1.5, color='k', linestyle=':')
    if legend:
        plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k',
                label='Pleiades')
    else:
        plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k')
    legend = False
plt.legend()
plt.tight_layout()
plt.savefig(f'Chrono.pdf')
