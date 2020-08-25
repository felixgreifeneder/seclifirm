import xarray as xr
import csv
import numpy as np

if __name__ == '__main__':
    dataDIR = 'C:/Users/FGreifeneder/Downloads/ERA5/orography.nc'
    DS = xr.open_dataset(dataDIR)

    with open('C:/Users/FGreifeneder/Downloads/ERA5/elevation.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
        csv_writer.writerow(['lon', 'lat', 'z'])
        for i in range(DS.z.shape[1]):
            for j in range(DS.z.shape[2]):
                csv_writer.writerow([str(DS.z[0,i,j].longitude.values), str(DS.z[0,i,j].latitude.values),
                                     str(DS.z[0,i,j].values)])