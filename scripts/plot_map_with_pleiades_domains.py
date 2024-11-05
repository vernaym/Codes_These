from snowtools.plots.maps import plot2D
import matplotlib.pyplot as plt
import snowtools.tools.xarray_preprocess as xrp
import xarray as xr
import numpy as np
import os

from pyproj import Proj, transform

import plot_elevation

#from snowtools.scripts.extract.vortex import vortex_get as io

# Grandes Rousses domain
latmax = 45.240,
latmin = 44.990,
lonmin = 6.010,
lonmax = 6.490,

def proj_mnt(mnt):
    outProj = Proj(init='epsg:4326')
    inProj = Proj(init='epsg:2154')
    x, y = np.meshgrid(mnt['xx'], mnt['yy'])
    X, Y = transform(inProj, outProj, x, y)
    #Z = mnt['ZS']
    mnt_proj = xr.DataArray(
        #data=Z,
        data=mnt.data,
        name='elevation',
        dims=["lat", "lon"],
        coords=dict(lon=X[0], lat=Y[:,0]),
        attrs=dict(description="Elevation",units="m"),
    )
    return mnt_proj

#ds = xr.open_dataset('CUMUL_ANTILOPEH_GrandesRousses_2021073106_2022070106.nc')
#ds = xrp.preprocess(ds)

dem = xr.open_dataset(os.path.join("/home/vernaym/.vortexrc/hack/uget/vernaym/data/", "DEM_GrandesRousses25m_L93.tif"))
dem = dem.band_data
dem = proj_mnt(dem)
dem = dem
vmin = 600
vmax = 4000
cmap = plt.cm.terrain
ratio =14
savename = 'Relief_GrandesRousses25m_Pleiades_2018_2022.pdf'
dem = xrp.preprocess(dem)
# dem = dem.sel({'xx': ds.xx, 'yy': ds.yy})

FondCarte = 'elevation'
#FondCarte = 'ratio'
if FondCarte == 'uncertainty':
    field = xr.open_dataset('/home/vernaym/workdir/ASSIMILATION/RandomSampling/XP27/Observation_error.nc')
    field = field.where((field.lon>=lonmin) & (field.lon<=lonmax) & (field.lat>=latmin) & (field.lat<=latmax), drop=True)
    field = field.Uncertainty
    vmin = 0
    vmax = 26
    cmap = plt.cm.YlOrBr
    ratio = 10
    savename = 'Uncertainty_GrandesRousses1km_Pleiades_2018_2022.pdf'
elif FondCarte == 'ratio':
    field = xr.open_dataset('/home/vernaym/workdir/ASSIMILATION/RandomSampling/XP27/Estimated_ratio.nc')
    field = field.where((field.lon>=lonmin) & (field.lon<=lonmax) & (field.lat>=latmin) & (field.lat<=latmax), drop=True)
    field = field.Ratio
    vmin = 0.7
    vmax = 1.3
    cmap = plt.cm.RdBu_r
    ratio = 10
    savename = 'Ratio_GrandesRousses1km_Pleiades_2018_2022.pdf'
elif FondCarte == 'elevation':
    field = dem
    field = field.rename({'xx': 'lon', 'yy': 'lat'})

#filename = 'Pleiades_20190513.nc'
#io.get(vapp='Pleiades', geometry='Huez250m', xpid='CesarDB_AngeH@vernaym', date='2019051312',
#    kind='SnowObservations', filename=filename)
#pleiades = xr.open_dataarray(filename)
#pleiades.rio.write_crs("EPSG:4326", inplace=True)
#pleiades = pleiades.rio.reproject("EPSG:2154")
#pleiades.data[~np.isnan(pleiades.data)] = 0

#fig, ax = plot2D.plot_field(ds.rr_cumul, vmin=0)
#plot2D.plot_field(dem, vmin=600, vmax=3900)
plt.figure(figsize=(ratio * len(field.lon) / len(field.lat), 11))
#im = plt.contourf(field.xx, field.yy, field.data, cmap=plt.cm.terrain, levels=100, alpha=0.9, antialiased=False)
#im = plot2D.plot_field(field, vmin=vmin, vmax=vmax, dem=dem)
im = field.plot(vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
im.set_edgecolor("face")
ax = plt.gca()

plot2D.add_iso_elevation(dem, ax=ax, levels=[1500, 2000, 2500, 3000, 3500])

plot_elevation.add_landmarks(ax, transform=False)
plot_elevation.add_postes(ax, type_poste='automatic stations')
plot_elevation.add_postes(ax, type_poste='nivometeo stations')

#plot2D.add_rectangle(ax, 'GrandesRousses', linewidth=4)
plot2D.add_quadrilateral(ax, 'Lautaret area (Pleiades 2018)', color='darkviolet', linewidth=7, linestyle='--')
#plot2D.add_quadrilateral(ax, 'Pleiades2019', color='crimson', linewidth=3, linestyle='--')
plot2D.add_quadrilateral(ax, 'Huez area (Pleiades 2022)', color='red', linewidth=7, linestyle='--')

plt.title(None)
plt.legend(ncol=2, loc=(0.05, 1.01))
plt.tight_layout()

#plt.savefig('Relief_GrandesRousses25m_Pleiades_2022.pdf')
plt.savefig(os.path.join('/home/vernaym/These/figures', savename))
