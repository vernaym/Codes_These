import geopandas
import xarray as xr

# Open shapefile --> returns geopandas.geodataframe.GeoDataFrame
shp = geopandas.read_file('shapefile.shp')
ds = xr.dataset('CUMUL_ANTILOPEH_alp_2021073106_2021102923.nc')
ds = ds.rio.write_crs(shp.crs)
masked = ds.rio.clip(shp.geometry.values, shp.crs)
