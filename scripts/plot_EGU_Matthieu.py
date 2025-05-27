
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#from These.scripts import make_mask
from snowtools.plots.maps import plot2D


latmax = 45.240
latmin = 44.990
lonmin = 6.010
lonmax = 6.490

vmin = 0
vmax = 40

dem = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_ALPES_WGS84_250m_bilinear.nc')
if 'elevation' not in dem.keys():
    dem = dem.rename({'Band1': 'elevation'})['elevation']
    dem = dem.where((dem['lat']>=latmin) & (dem['lat']<=latmax) & (dem['lon']>=lonmin) & (dem['lon']<=lonmax), drop=True)
dem = dem.rename({'lon': 'xx', 'lat': 'yy'})

auto = pd.read_csv('/home/vernaym/These/NO_TRANSFER/DATA/obs_quotidienne_auto_RR_20211101_20220430.csv', sep=';')
nivometeo = pd.read_csv('/home/vernaym/These/NO_TRANSFER/DATA/obs_nivometeo_daily_RR_20211201_20220430.csv', sep=';')
nivometeo = nivometeo.rename(columns={'Q.dat': 'date', 'Q.rr': 'rr', 'poste_nivo.lat_dg': 'lat', 'poste_nivo.lon_dg': 'lon'})
auto = auto.loc[(auto['date']=='2021-12-08 06:00:00') & (auto['lon'] >= lonmin) & (auto['lon'] <= lonmax) & (auto['lat'] >= latmin) & (auto['lat'] <= latmax)]
auto = auto.loc[~auto['num_poste'].isin([5063003, 38163006])]  # Remove EDFNIO observations
nivometeo = nivometeo.loc[(nivometeo['date']=='2021-12-07 00:00:00') & (nivometeo['lon'] >= lonmin) & (nivometeo['lon'] <= lonmax) & (nivometeo['lat'] >= latmin) & (nivometeo['lat'] <= latmax)]

obs = pd.concat([auto, nivometeo])

arome    = False
antilope = False
pearome  = True

if arome:
    ds    = xr.open_dataset('AROME_alp_daily.nc')
    #arome = ds.sel({'time': '2021-12-08 06', 'lon': slice(lonmin, lonmax), 'lat': slice(latmin, latmax)})
    tmp = ds.sel({'time': '2021-12-08 06'})
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=ccrs.PlateCarree()), layout='compressed')
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    #cml = make_mask.plot_field(fig, ax, tmp.rr, cmap=plt.cm.YlGnBu, vmin=vmin, vmax=vmax, elevation=True)
    plot2D.plot_field(tmp.rr.rename('Precipitation'), ax=ax, dem=dem, cmap=plt.cm.YlGnBu, vmin=vmin, vmax=vmax)
    obs.plot.scatter('lon', 'lat', c='rr', s=16**2, edgecolor='black', cmap=plt.cm.YlGnBu, vmin=0, vmax=vmax, ax=ax,
              colorbar=False)
    ax.set_title('')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    fig.savefig('AROME_20211208.pdf')

if antilope:
    ds    = xr.open_dataset('ANTILOPE_GrandesRousses_hourly.nc')
    #arome = ds.sel({'time': '2021-12-08 06', 'lon': slice(lonmin, lonmax), 'lat': slice(latmin, latmax)})
    tmp = ds.sel({'time': slice('2021-12-07 07', '2021-12-08 06')}).sum('time')
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection=ccrs.PlateCarree()), layout='compressed')
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    #cml = make_mask.plot_field(fig, ax, tmp.rr, cmap=plt.cm.YlGnBu, vmin=vmin, vmax=vmax, elevation=True)
    tmp = tmp.rr.rename('Precipitation').assign_attrs(units='mm/24h', description='24 hour precipitation')
    plot2D.plot_field(tmp, ax=ax, dem=dem, cmap=plt.cm.YlGnBu, vmin=vmin, vmax=vmax)
    obs.plot.scatter('lon', 'lat', c='rr', s=16**2, edgecolor='black', cmap=plt.cm.YlGnBu, vmin=0, vmax=vmax, ax=ax,
              colorbar=False)
    ax.set_title('')
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    fig.savefig('ANTILOPE_20211208.pdf')

if pearome:
    ds    = xr.open_dataset('PEAROME_GrandesRousses_daily.nc')
    ds = ds.sel({'time': '2021-12-08 06'})
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16,9), subplot_kw=dict(projection=ccrs.PlateCarree()))
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.05, 0.03, 0.9])
    i = 0
    j = 0
    for member in ds.member.data:
        ax[i, j].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
        tmp = ds.sel({'member': member})
        im = plot2D.plot_field(tmp.rr.rename('Precipitation'), ax=ax[i, j], dem=dem, cmap=plt.cm.YlGnBu, vmin=vmin, vmax=vmax, add_colorbar=False)
        ax[i, j].set_title('')
        j = j + 1
        if j==4:
            j = 0
            i = i + 1

    cb = fig.colorbar(im, cax=cbar_ax)
    cb.ax.tick_params(labelsize=20)
    cb.set_label('Precipitation [mm/24h]', size=24)

    fig.savefig('PEAROME_20211208.pdf')



