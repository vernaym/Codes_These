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
    NivometeoHuez = dict(
        xx        = 942705.64,
        yy        = 6447916.82,
    ),  # 1860m
)

parser = argparse.ArgumentParser()

parser.add_argument("datebegin", help="Start date")
parser.add_argument("dateend", help="End date")
parser.add_argument("point", help="Name of the point to plot", choices=coords.keys())
args = parser.parse_args()

pleiades_map = {
    '2018': dict(dates=['2018012312', '2018031612'], geometry='Lautaret250m'),
    '2019': dict(dates=['2019051312'], geometry='Huez250m'),
    '2022': dict(dates=['2022022612', '2022050112'], geometry='Huez250m'),
}
datebegin = args.datebegin
dateend   = args.dateend
point     = args.point

xx = coords[point]['xx']
yy = coords[point]['yy']

year = dateend[:4]
dates_pleiades = pleiades_map[year]['dates']
geometry = pleiades_map[year]['geometry']

if datebegin == '2021080106':
    deb = '2021080207'
else:
    deb = datebegin
io.get_pro(datebegin=deb, dateend=dateend, xpid='RS27_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_RS27.nc', vapp='edelweiss')
io.get_pro(datebegin=deb, dateend=dateend, xpid='RS27_sorted_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_RS27_sorted.nc', vapp='edelweiss')
io.get_pro(datebegin=deb, dateend=dateend, xpid='EnKF36_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_EnKF36.nc', vapp='edelweiss')
io.get_pro(datebegin=deb, dateend=dateend, xpid='PF32_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_PF32.nc', vapp='edelweiss')

xpid_map = {
    '2021080106': 'CesarDB',
    '2019080106': 'CesarDB_AngeH',
    '2018080106': 'CesarDB_AngeH',
}
io.get_snow_obs_date(xpid=xpid_map[datebegin], geometry=geometry, date=dates_pleiades,
        vapp='Pleiades', filename='Pleiades_[date:ymdh].nc')

rs_sorted = xr.open_dataarray('PRO_RS27_sorted.nc')
rs = xr.open_dataarray('PRO_RS27.nc')
enkf = xr.open_dataarray('PRO_EnKF36.nc')
pf = xr.open_dataarray('PRO_PF32.nc')

fig, ax = plt.subplots(figsize=(14, 4))
#ax.fill_between(rs_sorted.time, rs_sorted.min(dim='member'), rs_sorted.max(dim='member'), alpha=0.2, label='RS_sorted')
for member in rs_sorted.member:
    ax.plot(rs_sorted.time, rs_sorted.sel({'member': member}), color='red', label='RSS')
ax.fill_between(rs.time, rs.min(dim='member'), rs.max(dim='member'), alpha=0.5, label='RS')
ax.fill_between(enkf.time, enkf.min(dim='member'), enkf.max(dim='member'), alpha=0.5, label='EnKF')
ax.fill_between(pf.time, pf.min(dim='member'), pf.max(dim='member'), alpha=0.5, label='PF')

deb = pd.Timestamp(year=int(datebegin[0:4]), month=int(datebegin[4:6]), day=int(datebegin[6:8]), tz="UTC")
end = pd.Timestamp(year=int(dateend[0:4]), month=int(dateend[4:6]), day=int(dateend[6:8]), tz="UTC")
if os.path.exists('HTN.obs'):
    obs = pd.read_csv('HTN.obs', sep=';', parse_dates=['dat'])
    obs = obs[(obs.dat >= deb) & (obs.dat <= end)]
    plt.plot(obs.dat.values, obs.neigetot.values / 100, color='k', label='In-situ observation')
if os.path.exists('HTN2.obs'):
    obs = pd.read_csv('HTN.obs', sep=';', parse_dates=['dat'])
    obs = obs[(obs.dat >= deb) & (obs.dat <= end)]
    plt.plot(obs.dat.values, obs.neigetot.values / 100, color='k')

legend = True
for date in dates_pleiades:
    pleiades = xr.open_dataset(f'Pleiades_{date}.nc')
    htn = pleiades.interp({'x': xx, 'y': yy}, method='nearest').DSN_T_ISBA
    plt.vlines(pd.to_datetime(date, format='%Y%m%d%H'), 0, 1.5, color='k', linestyle=':')
    if legend:
        plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k',
                label='Pleiades')
    else:
        plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k')
    legend = False
plt.legend()
plt.tight_layout()
plt.savefig('Chrono_ensemble.pdf')
