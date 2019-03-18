import configparser
import datetime
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
config['query_settings']['file_path'] = "C:\\Users\\Dan Travers\\Google Drive\\Projects\\Load & PV\\data_files"
with open('C:\\Users\\Dan Travers\\Documents\\GitHub\\ss-obs\\src\\load.ini', 'w') as configfile:
    config.write(configfile)