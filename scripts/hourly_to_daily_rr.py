import pandas as pd
import numpy as np

#https://stackoverflow.com/questions/15799162/resampling-within-a-pandas-multiindex

df=pd.read_csv("/home/vernaym/These/DATA/obs_horaires_clim_RR.data", sep=';', parse_dates=['dat'])

newdates=df['dat']-pd.Timedelta(7, 'H')  # Shift time to sum rr over 6h J --> 6h J+1 period
df = df.assign(date=newdates).drop(columns=['dat'])  # Replace date column
df = df.set_index(['num_poste', 'lat', 'lon', 'alti', 'poste', 'reseau_poste', 'date']).sort_index()
level_values = df.index.get_level_values
df = df.groupby([level_values(i) for i in range(len(df.index.names)-1)]+[pd.Grouper(freq='1D', level=-1)]).sum()
df = df.reset_index()
newdates = df['date']+pd.Timedelta(1, 'D')  # Shift dates so that date YYYYMMDD is the cumul between 6h YYYYMM{D-1}06 and YYYYMMDD06
df = df.assign(date=newdates)  # Replace date column
df = df.set_index(['num_poste', 'lat', 'lon', 'alti', 'poste', 'reseau_poste', 'date']).sort_index()

df.to_csv("/home/vernaym/These/DATA/obs_quotidienne_clim_RR.data")



