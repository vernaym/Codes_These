import sys
import xarray as xr

filename = sys.argv[1]

ds = xr.open_dataset(filename)
ds.rio.write_crs("EPSG:4326", inplace=True)
ds  = ds.rename({'lon': 'x', 'lat': 'y'})
out = ds.rio.reproject("EPSG:2154")
outname = f"{filename.split('.')[0]}_L93.nc"
out.to_netcdf(outname)
