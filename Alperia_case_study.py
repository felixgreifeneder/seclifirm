import xarray
import pandas as pd
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt

def get_forecast_ts(lon, lat, path, param_name, lead_time):

    # initialize dataset
    ds = xarray.open_dataset(path, decode_times=False)

    # convert date from
    dates_conv = [dt.datetime(1900,1,1,0,0,0) + dt.timedelta(days=np.int(x)) for x in ds.date.values]
    # determine nearest gridcell
    grid_lon = math.trunc(lon)
    grid_lat = math.trunc(lat)
    # extract time series
    lead_time = lead_time - 1
    lon_pos = np.where(ds.longitude == grid_lon)[0][0]
    lat_pos = np.where(ds.latitude == grid_lat)[0][0]
    param_df = pd.DataFrame(ds[param_name][:, lead_time, :, lat_pos, lon_pos].values, index=dates_conv)
    # aggregate ensemble members
    param_df_agg = param_df.aggregate([np.mean, np.std], axis=1)
    param_df_agg['conf_neg'] = param_df_agg['mean'] - param_df_agg['std']
    param_df_agg['conf_pos'] = param_df_agg['mean'] + param_df_agg['std']

    # close dataset
    ds.close()

    return param_df_agg

def read_station_data(path):

    # initialize data variable
    station_df = pd.read_excel(path)

    # define index
    datelist = pd.date_range(dt.datetime(1991, 1, 1), dt.datetime(2017, 12, 31))

    # define series
    ro_series = pd.Series(index=datelist)

    for dix in datelist:
        ro_series[dix] = np.float(station_df[str(dix.year)][dix.dayofyear-1])

    return ro_series

def forecast_vs_insitu():

    # get runoff forecast
    fc = get_forecast_ts(10.98, 46.54,
                         "//projectdata.eurac.edu/projects/seclifirm/seasonal_forecasts/runoff/ECMWF_5_hindcast_monthly_1993_2016_IT.nc",
                         "mrort",
                         6)

    # insitu
    insitu = read_station_data("//projectdata.eurac.edu/projects/seclifirm/Alperia case study/9505-WP3-CS2-1-1-Abflüsse Zoccolo.xlsx")
    insitu_res = insitu.resample('MS').mean()
    insitu_res.name = 'Insitu'

    # merge datasets
    merged = pd.concat([fc, insitu_res], axis=1, join='inner')

    # compute climatologies
    clim = merged.groupby(merged.index.month).mean()
    # expand
    clim_exp = pd.DataFrame(merged, copy=True)
    for i in range(1,13):
        for j in np.where(clim_exp.index.month == i)[0]:
            clim_exp.iloc[j, :] = clim.iloc[i-1, :]

    # compute anomalies
    anomalies = (merged - clim_exp) / clim_exp.std()

    # create plot
    merged[['mean', 'conf_neg', 'conf_pos', 'Insitu']].plot(secondary_y='Insitu')

