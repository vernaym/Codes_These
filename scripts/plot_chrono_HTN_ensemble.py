import os
import xarray as xr
import argparse
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from snowtools.scripts.extract.vortex import vortexIO as io
import snowtools.tools.xarray_preprocess as xrp
import snowtools.scripts.post_processing.extract_point as pp


coords = pp.reference_points

xpid_map = dict(
    RS27_pappus              = 'RS',
    EnKF36_pappus            = 'EnKF',
    PF32_pappus              = 'PF',
    RS27_sorted_pappus       = 'SRS',
    RS27_pappus_assim        = 'RS_assim',
    EnKF36_pappus_assim      = 'EnKF_assim',
    PF32_pappus_assim        = 'PF_assim',
    RS27_sorted_pappus_assim = 'SRS_assim',
    RS27_spa_erroOBS_025     = 'SRS_err025',
    ANTILOPE_pappus          = 'ANTILOPE',
    SAFRAN_pappus            = 'SAFRAN',
)

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--datebegin", help="Start date")
parser.add_argument("-e", "--dateend", help="End date")
parser.add_argument("-x", "--xpids", nargs='+', help="XPID(s) to plot")
parser.add_argument("-p", "--point", help="Name of the point to plot", choices=coords.keys())
args = parser.parse_args()

pleiades_map = {
    '2018': dict(dates=['2018012312', '2018031612'], geometry='Lautaret250m', xpid='CesarDB_AngeH'),
    '2019': dict(dates=['2019051312'], geometry='Huez250m', xpid='CesarDB_AngeH'),
    '2022': dict(dates=['2022022612', '2022050112'], geometry='GrandesRousses250m', xpid='CesarDB'),
}
datebegin = args.datebegin
dateend   = args.dateend
xpids     = args.xpids
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

fig, ax = plt.subplots(figsize=(14, 4))
for xpid in xpids:
    if 'assim' in xpid or xpid == 'RS27_spa_erroOBS_025':
        vapp = 's2m'
    else:
        vapp = 'edelweiss'
    io.get_pro(datebegin=deb, dateend=dateend, xpid=xpid, vconf=point, geometry='SinglePoint',
            namespace='vortex.cache.fr', filename=f'PRO_{xpid}.nc', vapp=vapp)
    ds = xr.open_dataarray(f'PRO_{xpid}.nc')

    if 'sorted' in xpid or len(xpids) == 1:
        for i, member in enumerate(ds.member):
            if i == 0:
                ax.plot(ds.time, ds.sel({'member': member}), color='blue', alpha=0.5, linestyle='-', linewidth=1,
                        label='AS-ANTILOPE')
            elif i == 1:
                ax.plot(ds.time, ds.sel({'member': member}), color='blue', alpha=0.5, linestyle=':', linewidth=1,
                        label=xpid_map[xpid])
            else:
                ax.plot(ds.time, ds.sel({'member': member}), color='blue', alpha=0.5, linestyle=':', linewidth=1)
    else:
        ax.fill_between(ds.time, ds.min(dim='member'), ds.max(dim='member'), alpha=0.5, label=xpid_map[xpid])


io.get_snow_obs_date(xpid=pleiades_map[year]['xpid'], geometry=geometry, date=dates_pleiades,
        vapp='Pleiades', filename='Pleiades_[date:ymdh].nc')

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
assim = False
for date in dates_pleiades:
    pleiades = xr.open_dataset(f'Pleiades_{date}.nc')
    pleiades = xrp.preprocess(pleiades, mapping={'DSN_T_ISBA': 'HTN', 'DEP': 'HTN'})
    htn = pleiades.sel({'xx': xx, 'yy': yy}, method='nearest').HTN
    # htn = pleiades.interp({'x': xx, 'y': yy}, method='nearest').DSN_T_ISBA
    if date in ['2018012312', '2022022612']:
        #plt.vlines(pd.to_datetime(date, format='%Y%m%d%H'), htn - 0.2, htn + 0.2, color='red', linestyle='-',
        #        linewidth=2)
        #plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, color='red', linestyle='', marker='_', markersize=20,
        #        label='Pleiades (assimilated)')
        gauss = np.random.normal(loc=htn, scale=0.2, size=10000)
        pos = matplotlib.dates.date2num(pd.to_datetime(date, format='%Y%m%d%H'))
        vl = plt.violinplot(gauss, showmeans=False, showmedians=True, showextrema=False, positions=[pos], widths=15,
                side='high')
        # Set the color of the violin patches
        for item in vl['bodies']:
            item.set_color('red')
        # Set the color of the median lines
        #for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        for partname in ['cmedians']:
            vl[partname].set_colors('red')
        assim = True
    else:
        # plt.vlines(pd.to_datetime(date, format='%Y%m%d%H'), 0, 1.5, color='k', linestyle=':')
        if legend:
            plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k',
                    label='Pleiades (evaluation)')
        else:
            plt.plot(pd.to_datetime(date, format='%Y%m%d%H'), htn, linestyle='', marker='.', markersize=20, color='k')
        legend = False

# Get automatically generated legend
handles, labels = plt.gca().get_legend_handles_labels()
if assim:
    # manually define the violinplot label
    patch = mpatches.Patch(color='red', label='Pleiades (assimilated)')
    # handles is a list, so append manual patch
    handles.append(patch)
# plot the legend
plt.legend(handles=handles)

plt.ylabel('Snow depth (m)')

plt.tight_layout()
products = '_'.join(xpids)
plt.savefig(f'Chrono_ensemble_{point}_{products}.pdf')
