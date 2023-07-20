#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 20/07/2023

import os, sys
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import xarray as xr
import time



def hourly_to_daily(data):
    """
    Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
    Problem : the xarray tools to do that allows only accumulations between 0h and 23h.
    solution : shift time serie by 7h, compute 24h accumulations and shift back !
    """
    data['time'] = data.time-np.timedelta64(7, 'h')
    t1 = time.time()
    data = data.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
    t2 = time.time()
    print(f'Computing daily antilope took {(t2-t1)*1000}.ms')
    data['time'] = data.time+np.timedelta64(30, 'h')

    return data

