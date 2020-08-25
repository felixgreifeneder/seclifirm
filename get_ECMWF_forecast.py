import cdsapi
import sys

c = cdsapi.Client()

variables = ['runoff', 'snow_depth','2m_temperature', 'total_precipitation']

for i_var in variables:

    outname = '//projectdata.eurac.edu/projects/seclifirm/seasonal_forecasts/' + i_var + \
              '/ECMWF_5_hindcast_monthly_1993_2016_IT_CDFNC.nc'

    c.retrieve(
        'seasonal-monthly-single-levels',
        {
            'format': 'netcdf',
            'originating_centre': 'ecmwf',
            'system': '5',
            'variable': [
                i_var
            ],
            'product_type': [
                'monthly_mean'#,'monthly_standard_deviation'
            ],
            'year': [
                '1993','1994','1995',
                '1996','1997','1998',
                '1999','2000','2001',
                '2002','2003','2004',
                '2005','2006','2007',
                '2008','2009','2010',
                '2011','2012','2013',
                '2014','2015','2016'
                # '2017', '2018', '2019'
            ],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12'
            ],
            'leadtime_month': [
                '1', '2', '3', '4', '5','6'
            ],
            'area': [47.2, 6.5, 36.6, 18.7],  # North, West, South, East. Default: global
        },
        outname)

