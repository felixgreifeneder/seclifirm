import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

def plot_hourly():
    ERA5_runoff_bronzolo = pd.Series(name='ERA5_runoff')
    ERA5_surf_runoff_bronzolo = pd.Series(name='ERA5_surf_runoff')
    for i in range(2000, 2019):
        dataDIR = '//projectdata.eurac.edu/projects/seclifirm/ERA-5/ERA5_runoff_' + str(i) + '_ST.nc'
        DS = xr.open_dataset(dataDIR)
        ERA5_runoff = DS.ro.dropna(dim='time')
        tmp = pd.Series(ERA5_runoff[:,[2,3,3,3,3,3,3,3,4,4,4],[8,2,3,4,5,6,7,8,3,5,6]].sum(
            dim=['longitude', 'latitude']), index=ERA5_runoff.time.values, name='ERA5_runoff')
        #tmp = pd.Series(ERA5_runoff[:, 3, 3], index=ERA5_runoff.time.values, name='ERA5_runoff')
        ERA5_runoff_bronzolo = pd.concat([ERA5_runoff_bronzolo, tmp], axis=0)

        dataDIR2 = '//projectdata.eurac.edu/projects/seclifirm/ERA-5/ERA5_surface_runoff_' + str(i) + '_ST.nc'
        DS2 = xr.open_dataset(dataDIR2)
        ERA5_surf_runoff = DS2.sro.dropna(dim='time')
        tmp = pd.Series(ERA5_surf_runoff[:,[2,3,3,3,3,3,3,3,4,4,4],[8,2,3,4,5,6,7,8,3,5,6]].sum(
            dim=['longitude', 'latitude']), index=ERA5_surf_runoff.time.values, name='ERA5_surf_runoff')
        #tmp = pd.Series(ERA5_surf_runoff[:, 3, 3], index=ERA5_surf_runoff.time.values, name='ERA5_surf_runoff')
        ERA5_surf_runoff_bronzolo = pd.concat([ERA5_surf_runoff_bronzolo, tmp], axis=0)

    to_datetime = lambda d: datetime.strptime(str(d), '%Y%m%d%H%M%S')
    runoff_bronzolo = pd.read_csv("//projectdata.eurac.edu/projects/seclifirm/Alperia case study/Q_UV_Etsch_Branzoll.txt",
                                  sep=' ', header=None, index_col=0, usecols=[0,1],
                                  names=['date', 'discharge'], skiprows=13, parse_dates=True,
                                  date_parser=to_datetime)

    # merge and plot
    ro_vs_insitu = pd.concat([ERA5_runoff_bronzolo, runoff_bronzolo], join='inner', axis=1)
    ERA5_clim = ro_vs_insitu['ERA5_runoff'].groupby(ro_vs_insitu.index.dayofyear).mean()
    insitu_clim = ro_vs_insitu['discharge'].groupby(ro_vs_insitu.index.dayofyear).mean()
    ERA5_anom = pd.Series([ro_vs_insitu['ERA5_runoff'][i] - ERA5_clim[i.dayofyear] for i in ro_vs_insitu.index],
                          index=ro_vs_insitu.index, name='ERA5_anom')
    insitu_anom = pd.Series([ro_vs_insitu['discharge'][i] - insitu_clim[i.dayofyear] for i in ro_vs_insitu.index],
                            index=ro_vs_insitu.index, name='Insitu_anom')
    anoms = pd.concat([ERA5_anom, insitu_anom], axis=1)
    tmp=anoms.corr()
    anoms.plot(kind='scatter', x='Insitu_anom', y='ERA5_anom', title='R=' + str(tmp.iloc[0,1]))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_anom_scatter.png', dpi=600)
    plt.close()
    anoms['2006-01-01':'2007-12-31'].plot(secondary_y='Insitu_anom')
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_anom_ts.png', dpi=600)


    ro_vs_insitu.plot(secondary_y='ERA5_runoff', subplots=True, sharex=True)
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_ts.png", dpi=600)
    plt.close()

    ro_vs_insitu.plot(x='discharge', y='ERA5_runoff', kind='scatter', figsize=(7,7))
    #ax.figsize
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_scatter.png", dpi=600)

    ro_vs_insitu = pd.concat([ERA5_surf_runoff_bronzolo, runoff_bronzolo], join='inner', axis=1)

    ERA5_clim = ro_vs_insitu['ERA5_surf_runoff'].groupby(ro_vs_insitu.index.dayofyear).mean()
    insitu_clim = ro_vs_insitu['discharge'].groupby(ro_vs_insitu.index.dayofyear).mean()
    ERA5_anom = pd.Series([ro_vs_insitu['ERA5_surf_runoff'][i] - ERA5_clim[i.dayofyear] for i in ro_vs_insitu.index],
                          index=ro_vs_insitu.index, name='ERA5_anom')
    insitu_anom = pd.Series([ro_vs_insitu['discharge'][i] - insitu_clim[i.dayofyear] for i in ro_vs_insitu.index],
                            index=ro_vs_insitu.index, name='Insitu_anom')
    anoms = pd.concat([ERA5_anom, insitu_anom], axis=1)
    tmp = anoms.corr()
    anoms.plot(kind='scatter', x='Insitu_anom', y='ERA5_anom', title='R=' + str(tmp.iloc[0, 1]))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_anom_scatter.png', dpi=600)
    plt.close()
    anoms['2006-01-01':'2007-12-31'].plot(secondary_y='Insitu_anom')
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_anom_ts.png', dpi=600)

    ro_vs_insitu.plot(secondary_y='ERA5_surf_runoff', subplots=True, sharex=True)
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_ts.png", dpi=600)
    plt.close()

    ro_vs_insitu.plot(x='discharge', y='ERA5_surf_runoff', kind='scatter', figsize=(7, 7))
    # ax.figsize
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_scatter.png", dpi=600)

def plot_monthly():
    ERA5_runoff_bronzolo = pd.Series(name='ERA5_runoff')
    ERA5_surf_runoff_bronzolo = pd.Series(name='ERA5_surf_runoff')
    for i in range(2000, 2019):
        dataDIR = '//projectdata.eurac.edu/projects/seclifirm/ERA-5/ERA5_runoff_' + str(i) + '_ST.nc'
        DS = xr.open_dataset(dataDIR)
        ERA5_runoff = DS.ro.dropna(dim='time')
        tmp = pd.Series(ERA5_runoff[:, [2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4], [8, 2, 3, 4, 5, 6, 7, 8, 3, 5, 6]].sum(
            dim=['longitude', 'latitude']), index=ERA5_runoff.time.values, name='ERA5_runoff')
        # tmp = pd.Series(ERA5_runoff[:, 3, 3], index=ERA5_runoff.time.values, name='ERA5_runoff')
        ERA5_runoff_bronzolo = pd.concat([ERA5_runoff_bronzolo, tmp], axis=0).resample('M').sum()

        dataDIR2 = '//projectdata.eurac.edu/projects/seclifirm/ERA-5/ERA5_surface_runoff_' + str(i) + '_ST.nc'
        DS2 = xr.open_dataset(dataDIR2)
        ERA5_surf_runoff = DS2.sro.dropna(dim='time')
        tmp = pd.Series(ERA5_surf_runoff[:, [2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4], [8, 2, 3, 4, 5, 6, 7, 8, 3, 5, 6]].sum(
            dim=['longitude', 'latitude']), index=ERA5_surf_runoff.time.values, name='ERA5_surf_runoff')
        # tmp = pd.Series(ERA5_surf_runoff[:, 3, 3], index=ERA5_surf_runoff.time.values, name='ERA5_surf_runoff')
        ERA5_surf_runoff_bronzolo = pd.concat([ERA5_surf_runoff_bronzolo, tmp], axis=0).resample('M').sum()

    to_datetime = lambda d: datetime.strptime(str(d), '%Y%m%d%H%M%S')
    runoff_bronzolo = pd.read_csv(
        "//projectdata.eurac.edu/projects/seclifirm/Alperia case study/Q_UV_Etsch_Branzoll.txt",
        sep=' ', header=None, index_col=0, usecols=[0, 1],
        names=['date', 'discharge'], skiprows=13, parse_dates=True,
        date_parser=to_datetime)
    runoff_bronzolo = runoff_bronzolo.resample('M').mean()

    # merge and plot
    ro_vs_insitu = pd.concat([ERA5_runoff_bronzolo, runoff_bronzolo], join='inner', axis=1)
    ERA5_clim = ro_vs_insitu['ERA5_runoff'].groupby(ro_vs_insitu.index.month).mean()
    insitu_clim = ro_vs_insitu['discharge'].groupby(ro_vs_insitu.index.month).mean()
    ERA5_anom = pd.Series([ro_vs_insitu['ERA5_runoff'][i] - ERA5_clim[i.month] for i in ro_vs_insitu.index],
                          index=ro_vs_insitu.index, name='ERA5_anom')
    insitu_anom = pd.Series([ro_vs_insitu['discharge'][i] - insitu_clim[i.month] for i in ro_vs_insitu.index],
                            index=ro_vs_insitu.index, name='Insitu_anom')
    anoms = pd.concat([ERA5_anom, insitu_anom], axis=1)
    tmp = anoms.corr()
    anoms.plot(kind='scatter', x='Insitu_anom', y='ERA5_anom', title='R=' + str(tmp.iloc[0, 1]), figsize=(5,5),
               xlim=(-150,150),
               ylim=(-20,20))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_monthl_anom_scatter.png', dpi=600)
    plt.close()
    ax = anoms.plot(secondary_y='Insitu_anom')
    ax.right_ax.set_ylim((-150,150))
    ax.set_ylim((-20,20))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_monthl_anom_ts.png', dpi=600)

    ro_vs_insitu.plot(secondary_y='ERA5_runoff', subplots=True, sharex=True)
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_monthl_ts.png", dpi=600)
    plt.close()

    tmp=ro_vs_insitu.corr()
    ro_vs_insitu.plot(x='discharge', y='ERA5_runoff', kind='scatter', title='R=' + str(tmp.iloc[0, 1]), figsize=(5,5))
    # ax.figsize
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_monthl_scatter.png", dpi=600)
    plt.close()

    ro_vs_insitu = pd.concat([ERA5_surf_runoff_bronzolo, runoff_bronzolo], join='inner', axis=1)

    ERA5_clim = ro_vs_insitu['ERA5_surf_runoff'].groupby(ro_vs_insitu.index.month).mean()
    insitu_clim = ro_vs_insitu['discharge'].groupby(ro_vs_insitu.index.month).mean()
    ERA5_anom = pd.Series([ro_vs_insitu['ERA5_surf_runoff'][i] - ERA5_clim[i.month] for i in ro_vs_insitu.index],
                          index=ro_vs_insitu.index, name='ERA5_anom')
    insitu_anom = pd.Series([ro_vs_insitu['discharge'][i] - insitu_clim[i.month] for i in ro_vs_insitu.index],
                            index=ro_vs_insitu.index, name='Insitu_anom')
    anoms = pd.concat([ERA5_anom, insitu_anom], axis=1)
    tmp = anoms.corr()
    anoms.plot(kind='scatter', x='Insitu_anom', y='ERA5_anom', title='R=' + str(tmp.iloc[0, 1]), figsize=(5,5),
               xlim=(-150,150),
               ylim=(-20,20))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_monthl_anom_scatter.png', dpi=600)
    plt.close()
    ax = anoms.plot(secondary_y='Insitu_anom')
    ax.right_ax.set_ylim((-150, 150))
    ax.set_ylim((-20, 20))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_monthl_anom_ts.png', dpi=600)

    ro_vs_insitu.plot(secondary_y='ERA5_surf_runoff', subplots=True, sharex=True)
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_monthl_ts.png", dpi=600)
    plt.close()

    tmp = ro_vs_insitu.corr()
    ro_vs_insitu.plot(x='discharge', y='ERA5_surf_runoff', kind='scatter', title='R=' + str(tmp.iloc[0, 1]), figsize=(5,5))
    # ax.figsize
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_monthl_scatter.png", dpi=600)
    plt.close()


def plot_monthly_ultimo():
    ERA5_runoff_bronzolo = pd.Series(name='ERA5_runoff')
    ERA5_surf_runoff_bronzolo = pd.Series(name='ERA5_surf_runoff')
    for i in range(2000, 2019):
        dataDIR = '//projectdata.eurac.edu/projects/seclifirm/ERA-5/ERA5_runoff_' + str(i) + '_ST.nc'
        DS = xr.open_dataset(dataDIR)
        ERA5_runoff = DS.ro.dropna(dim='time')
        #tmp = pd.Series(ERA5_runoff[:, [2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4], [8, 2, 3, 4, 5, 6, 7, 8, 3, 5, 6]].sum(
        #    dim=['longitude', 'latitude']), index=ERA5_runoff.time.values, name='ERA5_runoff')
        tmp = pd.Series(ERA5_runoff[:, 2, 1], index=ERA5_runoff.time.values, name='ERA5_runoff')
        ERA5_runoff_bronzolo = pd.concat([ERA5_runoff_bronzolo, tmp], axis=0).resample('M').sum()

        dataDIR2 = '//projectdata.eurac.edu/projects/seclifirm/ERA-5/ERA5_surface_runoff_' + str(i) + '_ST.nc'
        DS2 = xr.open_dataset(dataDIR2)
        ERA5_surf_runoff = DS2.sro.dropna(dim='time')
        #tmp = pd.Series(ERA5_surf_runoff[:, [2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4], [8, 2, 3, 4, 5, 6, 7, 8, 3, 5, 6]].sum(
        #    dim=['longitude', 'latitude']), index=ERA5_surf_runoff.time.values, name='ERA5_surf_runoff')
        tmp = pd.Series(ERA5_surf_runoff[:, 2, 1], index=ERA5_surf_runoff.time.values, name='ERA5_surf_runoff')
        ERA5_surf_runoff_bronzolo = pd.concat([ERA5_surf_runoff_bronzolo, tmp], axis=0).resample('M').sum()

    to_datetime = lambda d: datetime.strptime(str(d), '%Y%m%d%H%M%S')
    runoff_bronzolo = read_station_data("//projectdata.eurac.edu/projects/seclifirm/Alperia case study/9505-WP3-CS2-1-1-Abflüsse Zoccolo.xlsx")
    runoff_bronzolo = runoff_bronzolo.resample('M').mean()
    runoff_bronzolo.name = 'discharge'

    # merge and plot
    ro_vs_insitu = pd.concat([ERA5_runoff_bronzolo, runoff_bronzolo], join='inner', axis=1)
    ERA5_clim = ro_vs_insitu['ERA5_runoff'].groupby(ro_vs_insitu.index.month).mean()
    insitu_clim = ro_vs_insitu['discharge'].groupby(ro_vs_insitu.index.month).mean()
    ERA5_anom = pd.Series([ro_vs_insitu['ERA5_runoff'][i] - ERA5_clim[i.month] for i in ro_vs_insitu.index],
                          index=ro_vs_insitu.index, name='ERA5_anom')
    insitu_anom = pd.Series([ro_vs_insitu['discharge'][i] - insitu_clim[i.month] for i in ro_vs_insitu.index],
                            index=ro_vs_insitu.index, name='Insitu_anom')
    anoms = pd.concat([ERA5_anom, insitu_anom], axis=1)
    tmp = anoms.corr()
    anoms.plot(kind='scatter', x='Insitu_anom', y='ERA5_anom', title='R=' + str(tmp.iloc[0, 1]), figsize=(5,5),
               xlim=(-10,10),
               ylim=(-0.2,0.2))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_monthl_anom_scatter.png', dpi=600)
    plt.close()
    ax = anoms.plot(secondary_y='Insitu_anom')
    ax.right_ax.set_ylim((-10,10))
    ax.set_ylim((-0.2,0.2))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_monthl_anom_ts.png', dpi=600)

    ro_vs_insitu.plot(secondary_y='ERA5_runoff', subplots=True, sharex=True)
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_monthl_ts.png", dpi=600)
    plt.close()

    tmp=ro_vs_insitu.corr()
    ro_vs_insitu.plot(x='discharge', y='ERA5_runoff', kind='scatter', title='R=' + str(tmp.iloc[0, 1]), figsize=(5,5))
    # ax.figsize
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/ro_vs_insitu_monthl_scatter.png", dpi=600)
    plt.close()

    ro_vs_insitu = pd.concat([ERA5_surf_runoff_bronzolo, runoff_bronzolo], join='inner', axis=1)

    ERA5_clim = ro_vs_insitu['ERA5_surf_runoff'].groupby(ro_vs_insitu.index.month).mean()
    insitu_clim = ro_vs_insitu['discharge'].groupby(ro_vs_insitu.index.month).mean()
    ERA5_anom = pd.Series([ro_vs_insitu['ERA5_surf_runoff'][i] - ERA5_clim[i.month] for i in ro_vs_insitu.index],
                          index=ro_vs_insitu.index, name='ERA5_anom')
    insitu_anom = pd.Series([ro_vs_insitu['discharge'][i] - insitu_clim[i.month] for i in ro_vs_insitu.index],
                            index=ro_vs_insitu.index, name='Insitu_anom')
    anoms = pd.concat([ERA5_anom, insitu_anom], axis=1)
    tmp = anoms.corr()
    anoms.plot(kind='scatter', x='Insitu_anom', y='ERA5_anom', title='R=' + str(tmp.iloc[0, 1]), figsize=(5,5),
               xlim=(-10,10),
               ylim=(-0.2,0.2))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_monthl_anom_scatter.png', dpi=600)
    plt.close()
    ax = anoms.plot(secondary_y='Insitu_anom')
    ax.right_ax.set_ylim((-10, 10))
    ax.set_ylim((-0.2, 0.2))
    plt.savefig('C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_monthl_anom_ts.png', dpi=600)

    ro_vs_insitu.plot(secondary_y='ERA5_surf_runoff', subplots=True, sharex=True)
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_monthl_ts.png", dpi=600)
    plt.close()

    tmp = ro_vs_insitu.corr()
    ro_vs_insitu.plot(x='discharge', y='ERA5_surf_runoff', kind='scatter', title='R=' + str(tmp.iloc[0, 1]), figsize=(5,5))
    # ax.figsize
    plt.savefig("C:/Users/FGreifeneder/Documents/Temp_Documents/sro_vs_insitu_monthl_scatter.png", dpi=600)
    plt.close()


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