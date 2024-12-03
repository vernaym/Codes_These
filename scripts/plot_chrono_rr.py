
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

ds  = xr.open_dataset('ANTILOPEH_2021080106_2022080106.nc')
tmp = ds.mean(['lat', 'lon']).resample(time='D').sum(dim='time')
tmp = tmp.rr.rename('Mean daily ANTILOPE precipitation over the Grandes Rousses domain (kg/m²)')

fig, ax = plt.subplots(figsize=(12, 6))

tmp.plot(ax=ax)

ax.axvspan(datetime(2021, 10, 25), datetime(2021, 10, 29), alpha=0.2, color='gray')
ax.axvspan(datetime(2021, 12, 13), datetime(2022, 1, 26), alpha=0.2, color='gray')
ax.axvspan(datetime(2022, 4, 5), datetime(2022, 6, 21), alpha=0.2, color='gray')
ax.axvspan(datetime(2022, 7, 1), datetime(2022, 8, 1), alpha=0.2, color='gray')

ax.set_title(None)

fig.savefig('chrono_rr.pdf')
