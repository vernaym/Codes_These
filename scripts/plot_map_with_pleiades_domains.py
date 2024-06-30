from snowtools.plots.maps import plot2D
import matplotlib.pyplot as plt
import snowtools.tools.xarray_preprocess as xrp
import xarray as xr
import numpy as np

from pyproj import Proj, transform

#from snowtools.scripts.extract.vortex import vortex_get as io


ds = xr.open_dataset('CUMUL_ANTILOPEH_GrandesRousses_2021073106_2022070106.nc')
ds = xrp.preprocess(ds)

dem = xr.open_dataset('DEM_ALP_WGS84_1km.nc')
dem = xrp.preprocess(dem.elevation)
dem = dem.sel({'xx': ds.xx, 'yy': ds.yy})

filename = 'Pleiades_20190513.nc'
#io.get(vapp='Pleiades', geometry='Huez250m', xpid='CesarDB_AngeH@vernaym', date='2019051312',
#    kind='SnowObservations', filename=filename)
pleiades = xr.open_dataarray(filename)
#pleiades.rio.write_crs("EPSG:4326", inplace=True)
#pleiades = pleiades.rio.reproject("EPSG:2154")
#pleiades.data[~np.isnan(pleiades.data)] = 0

fig, ax = plot2D.plot_field(ds.rr_cumul, vmin=0)

#pleiades.plot(ax=ax, alpha=0.5)

plot2D.add_iso_elevation(ax, dem, levels=[900, 1700, 2500, 3300])

plot2D.add_rectangle(ax, 'GrandesRousses', linewidth=4)
plot2D.add_quadrilateral(ax, 'Pleiades2018', color='purple', linewidth=3)
plot2D.add_quadrilateral(ax, 'Pleiades2019', color='orange', linewidth=3)
plot2D.add_quadrilateral(ax, 'Pleiades2022', color='red', linewidth=3)

fig.legend()

plt.show()
