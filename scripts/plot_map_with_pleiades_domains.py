from snowtools.plots.maps import plot2D
import matplotlib.pyplot as plt
import snowtools.tools.xarray_preprocess as xrp
import xarray as xr
import numpy as np
import os

from pyproj import Proj, transform

#from snowtools.scripts.extract.vortex import vortex_get as io

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
#dem = xrp.preprocess(dem.elevation)
#dem = dem.sel({'xx': ds.xx, 'yy': ds.yy})

filename = 'Pleiades_20190513.nc'
#io.get(vapp='Pleiades', geometry='Huez250m', xpid='CesarDB_AngeH@vernaym', date='2019051312',
#    kind='SnowObservations', filename=filename)
#pleiades = xr.open_dataarray(filename)
#pleiades.rio.write_crs("EPSG:4326", inplace=True)
#pleiades = pleiades.rio.reproject("EPSG:2154")
#pleiades.data[~np.isnan(pleiades.data)] = 0

#fig, ax = plot2D.plot_field(ds.rr_cumul, vmin=0)
#plot2D.plot_field(dem, vmin=600, vmax=3900)
plt.figure(figsize=(14 * len(dem.lon) / len(dem.lat), 10))
#im = plt.contourf(dem.xx, dem.yy, dem.data, cmap=plt.cm.terrain, levels=100, alpha=0.9, antialiased=False)
im = dem.plot(vmin=600, vmax=4000, cmap=plt.cm.terrain, rasterized=True)
im.set_edgecolor("face")
ax = plt.gca()

#plot2D.add_iso_elevation(ax, dem, levels=[900, 1700, 2500, 3300])

#plot2D.add_rectangle(ax, 'GrandesRousses', linewidth=4)
plot2D.add_quadrilateral(ax, 'Pleiades2018', color='dimgrey', linewidth=3, linestyle='--')
plot2D.add_quadrilateral(ax, 'Pleiades2019', color='crimson', linewidth=3, linestyle='--')
plot2D.add_quadrilateral(ax, 'Pleiades2022', color='darkviolet', linewidth=3, linestyle='--')

plt.title(None)
plt.legend()
plt.tight_layout()

#plt.savefig('Relief_GrandesRousses25m_Pleiades_2022.pdf')
plt.savefig('Relief_GrandesRousses25m_Pleiades_2018_2019_2022.pdf')
