import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ratio=xr.open_dataset('RATIO_ASANTILOPE_AROME.nc')
soda=xr.open_dataset('SODA_feedback.nc')

tmp=ratio.ratio.where(soda['mean'].notnull())

x = tmp.data.flatten()
x = x[~np.isnan(x)].reshape((-1,1))
y=soda['median'].data.flatten()
y = y[~np.isnan(y)]

reg = LinearRegression().fit(x, y)
z = reg.predict(x)

plt.scatter(x, y, marker='+')
plt.plot(x, z, marker=None, color='red')
plt.savefig('Ratio_vs_soda.pdf')
plt.close()

#out = ( (ratio.ratio - 1) * reg.coef_[0] + 9 ) / 8
out = ( (ratio.ratio - 1) * reg.coef_[0] - (reg.intercept_ + reg.coef_[0]) ) / 8
out.name = 'shift'
out.plot(vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
plt.savefig('SODA_shift_estimated_from_ANTILOPE_vs_AROME_ratio.pdf')
out.to_netcdf('SODA_shift_estimated_from_ANTILOPE_vs_AROME_ratio.nc')
