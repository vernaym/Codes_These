
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

from snowtools.scripts.extract.vortex import vortexIO as vio

datadir = '/mnt/lfs/d10/mrns/users/NO_SAVE/vernaym/workdir/EDELWEISS/DATA'
# datadir = '/home/vernaym/These/NO_TRANSFER/DATA'
savedir = '/mnt/lfs/d10/mrns/users/NO_SAVE/vernaym/workdir/EDELWEISS/analyse_situation_LPN'

lonmin = 6.010
lonmax = 6.490
latmin = 44.990
latmax  = 45.240

station   = 'Huez1860'
datebegin = '2021-12-27-06'
dateend   = '2021-12-29-12'

# 38191400;L Alpe d Huez (SATA);1860;45.097667;6.072833;51
# 38191408;L Alpe d Huez 2350;2340;45.105000;6.097500;51
# TODO : read from dataframe
extract_point = dict(
    Huez1860 = dict(
        num = 38191400,
        lat = 45.097667,
        lon = 6.072833,
        alt = 1860,
    ),
    Huez2350 = dict(
        num = 38191408,
        lat = 45.105000,
        lon = 6.097500,
        alt = 2340,
    ),
)


def timelapse(da, savename):

    fig, ax = plt.subplots()

    da.isel(time=0).plot(cmap=plt.cm.terrain, vmin=0, vmax=4000)

    def animate(time):
        # da.isel(time=time).plot(cmap=plt.cm.YlGnBu, vmin=0, vmax=5, add_colorbar=False)
        da.isel(time=time).plot(cmap=plt.cm.terrain, vmin=0, vmax=4000, add_colorbar=False)
        plt.title(f"Time: {da.isel(time=time).time.data}")

    ani = animation.FuncAnimation(fig, animate, len(da.time.data), interval=400, blit=False, repeat=False)

    ani.save(os.path.join(savedir, savename), fps=30)


def interactive(da, savename, cmap=plt.cm.gist_earth, vmin=None, vmax=None):
    """
    https://stackoverflow.com/questions/29407665/how-to-add-a-time-control-panel-to-a-funcanimation-from-matplotlib
    """
    from matplotlib.widgets import Slider, Button

    da = da.where((da.longitude >= lonmin) & (da.longitude <= lonmax) & (da.latitude >= latmin) &
            (da.latitude <= latmax), drop=True)
    huez = da.sel({'latitude': 45.1, 'longitude': 6.1}, method='nearest').squeeze()

    if vmin is None:
        vmin = da.min().data
    if vmax is None:
        vmax = da.max().data

    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [16, 3, 1]})
    # fig, ax = plt.subplots()

    # axtime = plt.axes([0.15, 0.01, 0.5, 0.05])
    # resetax = plt.axes([0.7, 0.01, 0.1, 0.05])
    stime = Slider(ax[2], 'Time', 0, len(da.time.data) - 1, valinit=0, valstep=1)

    # da.isel(time=0).plot(ax=ax, cmap=plt.cm.terrain, vmin=0, vmax=4000)
    da.isel(time=0).plot(ax=ax[0], cmap=cmap, vmin=vmin, vmax=vmax)
    huez.plot(ax=ax[1], color='k')
    vline = ax[1].axvline(huez.isel(time=0).time.data)
    ax[1].set_title('')

    def update(val):
        time = stime.val
        da.isel(time=time).plot(ax=ax[0], cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
        ax[0].set_title(f"Time: {da.isel(time=time).time.data}")

        vline.set_data(vline.set_data([huez.isel(time=time).time.data, huez.isel(time=time).time.data], [0, 1]))
        # huez.plot(ax=ax[1], color='k')
        # ax[1].axvspan(huez.isel(time=0).time.data, huez.isel(time=time).time.data, alpha=0.2, color='grey')
        # ax[1].axvline(huez.isel(time=time).time.data)
        fig.canvas.draw()

    # Routines to reset and update sliding bar
    def reset(event):
        stime.reset()

    stime.on_changed(update)
#    button = Button(resetax, 'Reset')
#    button.on_clicked(reset)
    pickle.dump(fig, open(os.path.join(savedir, savename), 'wb'))
    plt.show()
    plt.close()


def get_transition_zone(ds, ZS=None):
    out = ZS.copy().to_dataset()
    transition = xr.where((ds['Rainf'] > 0) & (ds['Snowf'] > 0), ZS, np.nan)
    out['max_elevation'] = transition.rolling({'x': 15, 'y': 15}, center=True, min_periods=10).max()
    out['min_elevation'] = transition.rolling({'x': 15, 'y': 15}, center=True, min_periods=10).min()

    return out[['max_elevation', 'min_elevation']]


def read_nivometeo():
    # Read nivometeo stations metadata for filtering
    nivometeo = pd.read_csv('postes_nivometeo.csv', sep=';')
    nivometeo = nivometeo.rename(columns={'poste_nivo.num_poste': 'num_poste', 'poste_nivo.lat_dg': 'lat',
        'poste_nivo.lon_dg': 'lon', 'poste_nivo.alti': 'alti', 'poste_nivo.nom_usuel': 'nom'})
    nivometeo = nivometeo.loc[(nivometeo['lat'] >= latmin) & (nivometeo['lat'] <= latmax) &
        (nivometeo['lon'] >= lonmin) & (nivometeo['lon'] <= lonmax)]  # Select area
    nivometeo = nivometeo.set_index('num_poste')

#         num_poste    poste_nivo.nom_usuel  poste_nivo.alti        lat       lon  hist_reseau_poste.reseau_poste
#    7      5063407     La Grave - La Meije             2420  45.025500  6.288667                              51
#    8      5063410         LA GRAVE 3200 M             3196  45.006000  6.253000                              51
#    50    38020400         AURIS-EN-OISANS             1600  45.057167  6.076333                              51
#    52    38191400    L Alpe d Huez (SATA)             1860  45.097667  6.072833                              51
#    53    38191408      L Alpe d Huez 2350             2340  45.105000  6.097500                              51
#    55    38253400  Les 2 Alpes (Toura NE)             2550  44.996667  6.171833                              51
#    56    38253407      Les 2 ALpes Jandri             3200  44.997000  6.204667                              51
#    58    38289401                   Olmet             1350  45.128167  6.071667                              51
#    62    38527400            VAUJANY-NIVO             1720  45.158833  6.097667                              51
#    96    73173400            Les Karellis             1603  45.227667  6.404000                              51
#    104   73280402       ST SORLIN D'ARVES             2090  45.217167  6.202500                              51
#    107   73306403                Valloire             2295  45.160833  6.463500                              51

    df = pd.read_csv('LPN_nivometeo_20161101_20220419.csv', sep=";", parse_dates=["H_NIVO.DAT"])
    # Select stations in the Grandes Rousses domain
    df = df[df['H_NIVO.NUM_POSTE'].isin(nivometeo.index)]
    # Select period
    df = df[(df['H_NIVO.DAT'] >= datebegin) &
            (df['H_NIVO.DAT'] <= pd.to_datetime(dateend) + pd.to_timedelta('24h'))].set_index('H_NIVO.NUM_POSTE')
    df = df.rename(columns={'H_NIVO.DAT': 'date', 'H_NIVO.ALTI_LPNX': 'lpnx'})

    out = df.join(nivometeo[['lat', 'lon', 'alti', 'nom']])

    return out


def get_mnt():
    geometry = 'GrandesRousses250m'
    vio.get_const('uenv:dem.2@vernaym', 'relief', geometry, filename='TARGET_RELIEF.nc',
                gvar='RELIEF_GRANDESROUSSES250M_4326')
    # Get Domain's DEM in case ZS not in simulation file
    mnt = xr.open_dataset('TARGET_RELIEF.nc').rename({'longitude': 'x', 'latitude': 'y'})['elevation']
    return mnt


def get_isowbt():
    isowbt = xr.open_dataarray(os.path.join(datadir, 'ISO_WETBT.arome-3dvarfr_2021080106_2022080106.nc'))
    out = isowbt.sel({'ISO_TPW': 27415}).drop_vars('time').rename({'valid_time': 'time'})
    out = out.sel(time=slice(datebegin, dateend)).rename({'longitude': 'x', 'latitude': 'y'})
    return out


def get_lhrxp(reference):
    lhr = xr.open_dataset(os.path.join(datadir, 'Precipitation_LHR_GrandesRousses250m_2021080206_2022080106.nc'))
    lhr = lhr.assign_coords({'x': reference['x'].data, 'y': reference['y'].data})
    lhr = lhr.sel(time=slice(datebegin, dateend))
    return lhr


def get_isoxp(reference):
    iso = xr.open_dataset(os.path.join(datadir, 'FORCING_RS27_GrandesRousses250m_2021080206_2022080106.nc'))
    iso = iso[['Rainf', 'Snowf']] * 3600.
    iso = iso.assign_coords({'x': reference['x'].data, 'y': reference['y'].data})
    iso = iso.sel(time=slice(datebegin, dateend))
    return iso


if __name__ == '__main__':

    # antilope = xr.open_dataarray(os.path.join(datadir, 'ANTILOPEH_2021080106_2022080106_alp.nc'))
    # timelapse(antilope.sel(time=slice('2021-12-28 00', '2021-12-30 00')),
    # timelapse(iso1wbt.sel(time=slice('2021-12-28 00', '2021-12-30 00')), savename='LPN_2021-12-28_2021-12-30.gif')
    # savename = 'LPN_2021-12-28_2021-12-30.pdf'
    # interactive(iso1wbt, savename=savename)

    mnt = get_mnt()
    obs = read_nivometeo()
    iso1wbt = get_isowbt()
    iso1wbt_250m = iso1wbt.interp({'y': mnt.y, 'x': mnt.x})
    lhr = get_lhrxp(mnt)
    # iso = get_isoxp(mnt)
    iso = lhr.copy()
    precipitation = iso['Rainf'] + iso['Snowf']
    iso['Rainf'] = precipitation.where(iso1wbt_250m > mnt, 0)
    iso['Snowf'] = precipitation.where(iso1wbt_250m <= mnt, 0)

    # Compute rain/snow transition zone from moving window
    gp = lhr.groupby('time')
    lhr.update(gp.apply(get_transition_zone, ZS=mnt))

    for num_poste in obs.index.unique():

        tmp = obs.loc[num_poste]  # Ensure tmp is still a dataframe
        if isinstance(tmp, pd.Series):
            tmp = tmp.to_frame().T
        # Extract point
        lat = tmp.lat.unique().squeeze()
        lon = tmp.lon.unique().squeeze()
        alt = tmp.alti.unique().squeeze()
        nom = tmp.nom.unique().squeeze()

        lhr_huez = lhr.sel({'x': lon, 'y': lat}, method='nearest').squeeze()
        iso_huez = iso.sel({'x': lon, 'y': lat}, method='nearest').squeeze()
        iso1wbt_huez = iso1wbt.sel({'x': lon, 'y': lat}, method='nearest').squeeze()
        alt_huez = mnt.sel({'x': lon, 'y': lat}, method='nearest').squeeze()

        time = pd.DataFrame(iso_huez.time.data, columns=['date'])
        time['lpnx'] = np.nan
        data = time.set_index('date').combine_first(tmp.set_index('date'))['lpnx']
        data = data.bfill()  # Fill observation backward in time
        data = data.loc[data.index <= dateend]

        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].axhline(y=alt, color='k', label='Nivometeo station')
        ax[0].plot(data.index, data.values, color='grey', label='LPNX nivometeo observation')
        ax[0].axhline(y=alt_huez.data, color='k', linestyle='--', label='Corresponding simulation pixel')
        ax[0].fill_between(lhr_huez.time, lhr_huez.min_elevation, lhr_huez.max_elevation,
                alpha=0.5, color='red', label='LHR')
        ax[0].plot(iso1wbt_huez.time.data, iso1wbt_huez.data, color='blue', label='ISO-WBT-1°C')
        ax[0].set_ylim(1000, 3000)
        ax[0].set_ylabel('Elevation (m)')
        ax[0].legend()

        total_precipitation = (lhr_huez['Snowf'] + lhr_huez['Rainf']).cumsum()
        ax[1].fill_between(total_precipitation.time, total_precipitation, alpha=0.5, color='grey',
                label='Total precipitation')
        ax[1].plot(lhr_huez.time.data, lhr_huez['Snowf'].cumsum(), color='red', label='Snowf LHR')
        ax[1].plot(iso_huez.time.data, iso_huez['Snowf'].cumsum(), color='blue', label='Snowf ISO')
        ax[1].set_ylim(0, total_precipitation.data[-1] * 1.1)
        ax[1].set_ylabel('Precipitation (mm)')
        ax[1].legend()

        fig.savefig(f'analyse_LPN_{nom}_{datebegin}_{dateend}.pdf')
