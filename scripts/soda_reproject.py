import xarray as xr
import rioxarray

soda=xr.open_dataset('SODA_feedback_20220226.nc')
soda=soda.rename({'xx': 'x', 'yy': 'y'})
soda.rio.write_crs("EPSG:2154", inplace=True)
out=soda.rio.reproject("EPSG:4326")
out=out.rename({'x': 'lon', 'y': 'lat'})

antilope=xr.open_dataset('ANTILOPEH_2021080206_2022080106_GrandesRousses.nc')
ant=antilope.sel(lat=slice(44.79, 45.44), lon=slice(5.81, 6.69))

out=out.interp(lon=ant.lon, lat=ant.lat, method='nearest')
out.to_netcdf('SODA_feedback_20220226_1km.nc')

