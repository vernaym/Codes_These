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

if datebegin == '2021080106':
    datebegin_RS27 = '2021080207'
    ymax = 1.3  # Huez 2021/2022
else:
    datebegin_RS27 = datebegin
    ymax = 3.5  # Rochilles Nivose 2017/2018

plot_lpn = False

io.get_pro(datebegin=datebegin_RS27, dateend=dateend, xpid='RS27_sorted_pappus', vconf=point, geometry='SinglePoint',
        namespace='vortex.cache.fr', filename='PRO_RS27.nc',vapp='edelweiss')
rs27 = xr.open_dataset('PRO_RS27.nc')
rs27 = rs27.DSN_T_ISBA
if plot_lpn:
    io.get_pro(datebegin=datebegin_RS27, dateend=dateend, xpid='RS27_LPNp200', vconf=point, geometry='SinglePoint',
            namespace='vortex.cache.fr', filename='PRO_RS27_bis.nc',vapp='edelweiss')
    rs27_bis = xr.open_dataset('PRO_RS27_bis.nc')
    rs27_bis = rs27_bis.DSN_T_ISBA
    io.get_meteo(
        kind        = 'ISO_WETBT',
        datebegin   = datebegin,
        dateend     = dateend,
        geometry    = "EURW1S40",
        xpid        = "ExtractionBDAP@vernaym",
        filename    = 'ISO_TPW.nc',
        vapp        = 'edelweiss',
        source_app  = 'arome',
        source_conf = '3dvarfr',
    )

else:
    io.get_pro(datebegin=datebegin, dateend=dateend, xpid='SAFRAN_perturb', vconf=point, geometry='SinglePoint',
            namespace='vortex.cache.fr', filename='PRO_SAFRAN.nc', vapp='edelweiss')
    io.get_pro(datebegin=datebegin, dateend=dateend, xpid='KRIGING_perturb', vconf=point, geometry='SinglePoint',
            namespace='vortex.cache.fr', filename='PRO_KRIGING.nc', vapp='edelweiss')
    io.get_pro(datebegin=datebegin, dateend=dateend, xpid='ANTILOPE_pappus', vconf=point, geometry='SinglePoint',
            namespace='vortex.cache.fr', filename='PRO_ANTILOPE.nc', vapp='edelweiss')
    io.get_pro(datebegin=datebegin, dateend=dateend, xpid='AROME_perturb', vconf=point, geometry='SinglePoint',
            namespace='vortex.cache.fr', filename='PRO_AROME.nc', vapp='edelweiss')
    safran = xr.open_dataset('PRO_SAFRAN.nc')
    safran = safran.DSN_T_ISBA
    kriging = xr.open_dataset('PRO_KRIGING.nc')
    kriging = kriging.DSN_T_ISBA
    antilope = xr.open_dataset('PRO_ANTILOPE.nc')
    antilope = antilope.DSN_T_ISBA
    arome = xr.open_dataset('PRO_AROME.nc')
    arome = arome.DSN_T_ISBA

xpid_map = {
    '2017080106': 'CesarDB',
    '2018080106': 'CesarDB_AngeH',
    '2021080106': 'CesarDB',
}
io.get_snow_obs_date(xpid=xpid_map[datebegin], geometry=geometry, date=dates_pleiades,
        vapp='Pleiades', filename='Pleiades_[date:ymdh].nc')

fig, ax = plt.subplots(figsize=(14, 4))

if plot_lpn:
    plt.plot(rs27.time, rs27, label='AS-ANTILOPE', color="blue")
    plt.plot(rs27_bis.time, rs27_bis, label='AS-ANTILOPE LPN+200m', color="green")
else:
    plt.plot(safran.time, safran, label='SAFRAN', color="#1f78b4")
    plt.plot(kriging.time, kriging, label='KRIGING', color="#e31a1c")
    plt.plot(arome.time, arome, label='AROME', color="#6a3d9a")
    plt.plot(antilope.time, antilope, label='ANTILOPE', color="#b2df8a")
    plt.plot(rs27.time, rs27, label='AS-ANTILOPE', color="#33a02c")

deb = pd.Timestamp(year=int(datebegin[0:4]), month=int(datebegin[4:6]), day=int(datebegin[6:8]),  tz="UTC")
end = pd.Timestamp(year=int(dateend[0:4]), month=int(dateend[4:6]), day=int(dateend[6:8]),  tz="UTC")
if os.path.exists('HTN.obs'):
    #htn = pd.read_csv('NIVOSE.htn', sep=';', parse_dates=['dat'])
    htn = pd.read_csv('HTN.obs', sep=';', parse_dates=['dat'])
    htn = htn[(htn.dat >= deb) & (htn.dat <= end)]
    ax.plot(htn.dat.values, htn.neigetot.values / 100, color='k', label='In-situ observation')

    if plot_lpn:
        alt = htn.alt.unique()[0]
        ax.set_ylim([-ymax, ymax])
        ax2 = ax.twinx()
        lpn = pd.read_csv('LPN.obs', sep=';', parse_dates=['dat'])
        lpn = lpn[(lpn.dat >= deb) & (lpn.dat <= end)]
        # Add LPN
        #ax.fill_between(lpn.dat.values, 0, np.where(lpn.lpn < lpn.alti, ymax, 0), color='grey', alpha=0.5, step='pre', label='LPN > station')
        #ax.fill_between(lpn.dat.values, 0, np.where(lpn.lpn < lpn.alti + 200, ymax, 0), color='grey', alpha=0.1, step='pre', label='LPN > station + 200m' )
        ax2.plot(lpn.dat.values, lpn.lpn.values - alt, marker='*', color='k', linestyle='', label='Maximum rain elevation')

        isot = xr.open_dataarray('ISO_TPW.nc')
        isot = isot.sel({'latitude': lpn.lat.unique(), 'longitude': lpn.lon.unique(), 'ISO_TPW': 27415}, method='nearest')
        isot = isot.sel({'valid_time': lpn.dat.values})
        isot = isot.where(isot.valid_time == lpn.dat.values).squeeze()
        isot = isot.where(~np.isnan(lpn.lpn.values))
        #isot = isot.sel({'valid_time': lpn.dat.values[~np.isnan(lpn.lpn.values)]}).squeeze()
        #isot = isot.where(isot.time.dt.hour == 6).squeeze()
        ax2.plot(isot.time.data, isot.data - alt, marker='*', color='b', linestyle='', label='Simulation LPN')
        ax2.set_ylim([-1800, 1800])
        ax2.set_ylabel("Elevation difference with Huez elevation (m)")
        ax2.legend(loc='lower right')
        diff = lpn.lpn.values - isot.data
        pos = np.where(diff > 0)
        neg = np.where(diff < 0)
        ax2.vlines(lpn.dat.values[pos], isot.data[pos] - alt, lpn.lpn.values[pos] - alt, color='blue', linestyle='--', label='LPN under-estimation')
        ax2.vlines(lpn.dat.values[neg], lpn.lpn.values[neg] - alt, isot.data[neg] - alt, color='red', linestyle='--', label='LPN over-estimation')
    else:
        ax.set_ylim([0, ymax])


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
    ax.vlines(pd.to_datetime(date, format='%Y%m%d%H'), -ymax, ymax, color='k', linestyle=':')
    if legend:
        ax.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k',
                label='Pleiades')
    else:
        ax.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k')
    legend = False
ax.set_ylabel('Snow depth (m)')
ax.legend(loc='upper left')
#ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.3, 0.5, 0.5))
plt.tight_layout()
plt.savefig('Chrono.pdf')
