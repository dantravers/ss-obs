import configparser
import datetime
import os
config = configparser.ConfigParser()
"""
store_name : str
    Name of HDF5 store used to store the midas data locally.
meta_hdf_key : str
    Key of the object within HDF5 store containing metadata.
obs_hdf_key : str
    Name of the object within HDF5 store containing weather observation data.
default_earliest_date : date
    If no dates are specified for loading, this is the default start date to fall back onto.  The default end date is the date today.
forecast_max_days_ahead : int
    The maximum number of days ahead the weather forecast includes. 
"""
config['hdf5'] = {}
config['hdf5']['store_name'] = "wforecast.h5"
config['hdf5']['store_path'] = "C:\\Users\\DanTravers\\Documents\\dbs\\hdf5"
config['hdf5']['meta_hdf_key'] = 'metadata'
config['hdf5']['obs_hdf_key'] = 'obs'
config['query_settings'] = {}
config['query_settings']['default_earliest_date'] = '2012-01-01'
config['query_settings']['forecast_max_days_ahead'] = '4'
config['query_settings']['path'] = 'C:\\Users\\DanTravers\\Documents\\dbs\\weather\\ecmwf'
config['query_settings']['location_ref_filename'] = "ecmwf2016-01-02T00:00:00.nc"
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config/wforecast.ini'), 'w') as configfile:
    config.write(configfile)

