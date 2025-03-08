import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

df  = pd.read_csv('PART.soda.GrandesRousses250m_2022022612.txt', sep=',', header=None)

median = df.apply(lambda x: x.median(), axis=1)
mean = df.apply(lambda x: x.mean(), axis=1)

ds = xr.open_dataset('DEM_GrandesRousses250m.nc')

med = xr.DataArray(
    data   = median.values.reshape(len(ds.y), len(ds.x)),
    name   = 'soda',
    dims   =["y", "x"],
    coords =dict(x=ds.x, y=ds.y),
    attrs  =dict(description="SODA output"),
)
med.to_netcdf('soda_median.nc')
med.plot()
plt.savefig('soda_median.pdf')

men = xr.DataArray(
    data   = mean.values.reshape(len(ds.y), len(ds.x)),
    name   = 'soda',
    dims   =["y", "x"],
    coords =dict(x=ds.x, y=ds.y),
    attrs  =dict(description="SODA output"),
)
men.to_netcdf('soda_mean.nc')
men.plot()
plt.savefig('soda_mean.pdf')
