import cdsapi
import os

if __name__ == '__main__':
	c = cdsapi.Client()

	years = ['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010',
			 '2011','2012','2013','2014','2015','2016','2017','2018','2019']
	variables = ['2m_temperature','snow_density','snow_depth',
				 'snow_depth_water_equivalent','total_precipitation']
	for iv in variables:
		for iy in years:
			# check if file was already downloaded
			if os.path.isfile('//projectdata.eurac.edu/projects/seclifirm/ERA-5-Land/ERA5_Land_' + iv + '_' + iy + '_ST.nc') == False:
				c.retrieve(
					'reanalysis-era5-land',
					{
						'format':'netcdf',
						'variable':[iv],
						'year':[iy],
						'month':[
							'01','02','03',
							'04','05','06',
							'07','08','09',
							'10','11','12'
						],
						'day':[
							'01','02','03',
							'04','05','06',
							'07','08','09',
							'10','11','12',
							'13','14','15',
							'16','17','18',
							'19','20','21',
							'22','23','24',
							'25','26','27',
							'28','29','30',
							'31'
						],
						'time':[
							'00:00','01:00','02:00',
							'03:00','04:00','05:00',
							'06:00','07:00','08:00',
							'09:00','10:00','11:00',
							'12:00','13:00','14:00',
							'15:00','16:00','17:00',
							'18:00','19:00','20:00',
							'21:00','22:00','23:00'
						],
						'area':[47.2, 10.3, 45.6, 12.6], # North, West, South, East. Default: global
					},
					'//projectdata.eurac.edu/projects/seclifirm/ERA-5-Land/ERA5_Land_' + iv + '_' + iy + '_ST.nc')
			else:
				print('ERA5_' + iv + '_' + iy + '_ST.nc' + ' is already downloaded')