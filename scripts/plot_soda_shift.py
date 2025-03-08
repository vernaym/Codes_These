
import os
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from These.scripts import make_mask

latmax = 45.240
latmin = 44.990
lonmin = 6.010
lonmax = 6.490


def plot_feedback():
    datadir = '/home/vernaym/workdir/EDELWEISS/precipitation_analysis/RandomSampling/RS27_sorted_feedback'
    ds = xr.open_dataset(os.path.join(datadir, 'SODA_feedback_20220226.nc'))
    tmp = (ds['mean'] - 9) / 8
    tmp = tmp.rename('shift')
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.LambertConformal()))
    ax.set_extent([ds.xx.min(), ds.xx.max(), ds.yy.min(), ds.yy.max()], crs=ccrs.LambertConformal())
    make_mask.plot_field(fig, ax, tmp, elevation=True, cmap=plt.cm.RdBu, vmin=-1, vmax=1, colorbar=True)
    plt.show()


def plot_full_feedback():

    datadir = '/home/vernaym/workdir/EDELWEISS/precipitation_analysis/RandomSampling/RS27_sorted_feedback_extended'
    ds = xr.open_dataset(os.path.join(datadir, 'SODA_shift_estimated_from_ANTILOPE_vs_AROME_ratio.nc'))

    ds = ds.where((ds.lon >= lonmin) & (ds.lon <= lonmax) & (ds.lat >= latmin) & (ds.lat <= latmax), drop=True)

    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    make_mask.plot_field(fig, ax, ds['shift'], elevation=True, cmap=plt.cm.RdBu, vmin=-1, vmax=1, colorbar=True)
    plt.show()
