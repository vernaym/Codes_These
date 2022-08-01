#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from snowtools.utils.prosimu import prosimu
import matplotlib.pyplot as plt
import argparse
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import os, sys

filename = sys.argv[1]

fic  = prosimu(filename)
time = fic.readtime()
altitude = fic.read_var('ZS')
aspect = fic.read_var('aspect')
massif = fic.read_var('massif_number')
# Extraction du massif des Grandes Rousses entre le 01/08/2021 et le 01/7/2022
point = np.where((massif==12) & (aspect==-1))
timeselect = np.where(time<=datetime.datetime(2022,7,1,6,0))
rain = fic.read("Rainf")[timeselect]
snow = fic.read("Snowf")[timeselect]
precip = (rain[:, point] + snow[:, point]).squeeze() * 3600.

rr = xr.DataArray(
    data = precip,
    name = 'rr',
    dims=["time", "elevation"],
    coords=dict(elevation=altitude[point], time=time[timeselect], reference_time=datetime.datetime(2021, 8, 1, 6, 0),),
    attrs=dict(description="Total precipitation",units="mm"),
)

rr.to_netcdf('CUMUL_SAFRAN_GrandesRousses_2021080106_2022070106.nc')
