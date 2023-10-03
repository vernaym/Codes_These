import xarray as xr
import numpy as np
from pyproj import Proj, transform

mnt250=xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_FRANCE_L93_250m_bilinear.nc')
x,y=np.meshgrid(mnt250.lon.data, mnt250.lat.data)
X,Y=transform(3857, 4326, x, y)
Z = mnt250['elevation']
mnt250_proj = xr.DataArray(data=Z.data, name='elevation', dims=["lat", "lon"], coords=dict(lon=Y[0], lat=X[:,0]), attrs=dict(description="Elevation",units="m"),)

antilope = xr.open_dataset('/home/vernaym/These/DATA/CUMUL_ANTILOPEH_pyr_2021103000_2022060200.nc')

mnt1 = mnt250_proj.interp(lon=antilope.lon, lat=antilope.lat, method='linear')
mnt1.to_netcdf('DEM_PYR_WGS84_1km.nc')
