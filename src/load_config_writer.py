import configparser
import datetime
import os
config = configparser.ConfigParser()
"""
store_name : str
    Name of HDF5 store used to store the midas data locally.
store_path : str
    Path of the HDF5 store.
meta_hdf_key : str
    Key of the object within HDF5 store containing metadata.
obs_hdf_key : str
    Name of the object within HDF5 store containing weather observation data.
default_earliest_date : date
    If no dates are specified for loading, this is the default start date to fall back onto.  The default end date is the date today.
query_settings > file_path : str
    Default path of files to be read into loads.
"""
config['hdf5'] = {}
config['hdf5']['store_name'] = "load.h5"
config['hdf5']['store_path'] = "C:\\Users\\Dan Travers\\Documents\\dbs\\hdf5"
config['hdf5']['meta_hdf_key'] = 'metadata'
config['hdf5']['obs_hdf_key'] = 'obs'
config['query_settings'] = {}
config['query_settings']['default_earliest_date'] = '2018-01-01'
config['query_settings']['site_id_metadata_object_cols'] = 'name, customer, latitude, longitude, kWp' 
config['query_settings']['file_path'] = "C:\\Users\\Dan Travers\\Google Drive\\Bryt\\Modelling\\data"
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config/load.ini'), 'w') as configfile:
    config.write(configfile)