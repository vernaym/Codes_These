import os
import numpy as np
import scipy
from scipy.sparse import diags
import xarray as xr
from These.radar import Preprocessing_ANTILOPE

import matplotlib.pyplot as plt

workdir = '/home/mrns/vernaym/workdir/SENSASS'  # sotrtm35-sidev
#workdir = '/home/vernaym/workdir/SENSASS'  # local

max_dist = 0.2

def dynamic_correction(field):
    print('Date=', field.time.data)
    new, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(field.data, pond)
    out = field.copy()
    out.data = new.reshape(out.data.shape)
    out.name = 'wma'
    return out

antilope = xr.open_dataset(os.path.join(workdir, 'ANTILOPEQ_2006070306_2023080106_HauteSavoie.nc'))
#antilope = xr.open_dataset(os.path.join(workdir, 'ANTILOPE_raw_20211204_HauteSavoie.nc'))

# Cut period into 2 sub-periods to avoid memory overflow
ndates = len(antilope.time)
#dates = antilope.time
#dates = antilope.time[5633:5636]  # 2021-12-04
dates = antilope.time[:ndates//2]
#dates = antilope.time[ndates//2:]

antilope = antilope.sel({'time':dates})

ratio = xr.open_dataset(os.path.join(workdir, f"Estimated_ratio.nc"))
ratio = ratio.sel(lat=np.intersect1d(antilope.lat, ratio.lat), lon=np.intersect1d(antilope.lon, ratio.lon))
antilope = antilope.sel(lat=np.intersect1d(antilope.lat, ratio.lat), lon=np.intersect1d(antilope.lon, ratio.lon))
antilope["ratio"] = ratio.Ratio  # Fill missing point with NaNs
var0 = 'rr'

# 1. Static de-biasing :
antilope["rr_debiaise"] = (antilope.rr/antilope.ratio).fillna(antilope.rr)  # Fill NaN values with the original ANTILOPE value
var1 = "rr_debiaise"

# 2. WMA correction
error = xr.open_dataarray(os.path.join(workdir, f'Observation_uncertainty.nc'))
error = error.sel(lat=np.intersect1d(antilope.lat, error.lat), lon=np.intersect1d(antilope.lon, error.lon))
antilope = antilope.sel(lat=np.intersect1d(antilope.lat, error.lat), lon=np.intersect1d(antilope.lon, error.lon))
std = error.data
codist = os.path.join(workdir, f'codistance_max_dist_{max_dist:.2f}.npz')
if not os.path.exists(codist):
    # Compute inter-distances
    coords=[(lon,lat) for lat in error.lat.data for lon in error.lon.data]
    pond = Preprocessing_ANTILOPE.codistances(coords)
    scipy.sparse.save_npz(codist, pond, compressed=False)  # TODO comprendre pourquoi ca ne marche pas pour éviter de recalculer les codistances à chaque fois
else:
    pond = scipy.sparse.load_npz(codist)
pond = pond.dot(diags(1/std.flatten(), 0))  # std is in [1, inf[
obs = antilope[var1].sel({'lat':np.intersect1d(error.lat.data, antilope.lat.data), 'lon':np.intersect1d(error.lon.data, antilope.lon.data)})
out = obs.groupby('time').apply(dynamic_correction)

# 3. Save output
deb = dates.data[0].astype(str)[:10]
end = dates.data[-1].astype(str)[:10]
out.to_netcdf(os.path.join(workdir, f'ANTILOPE_post-processed_HauteSavoie_{deb}_{end}.nc'))

