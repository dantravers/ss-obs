import numpy as np
import pytest
import dbconnector
import power
import midas
import configparser
import datetime
import os
import datetime

print(os.name)
data_dir = '/home/dtravers/winhome/Documents/GitHub/ss-obs/tests/test_data'
netcdf_dir = os.path.join(data_dir, 'netcdf')
netcdf_file = 'ecmwf2016-01-02T00%3A00%3A00.nc'
file = os.path.join(netcdf_dir, netcdf_file)

time_sep = ':' if ':' in file else '%%3A'
forecast_base = datetime.datetime.strptime(netcdf_file[5:-3], '%Y-%m-%dT%H{0}%M{0}%S'.format(time_sep))

"""config = configparser.ConfigParser()
config.read('/home/dtravers/winhome/Documents/GitHub/ss-obs/src/config/midas.ini')
weather = midas.Midas(2)
s = datetime.datetime(2018, 1, 1).date()
e = datetime.datetime(2018,1, 4).date()
#weather.load_data([842, 676], s, e, goto_db='Never')
print(weather.metadata)"""