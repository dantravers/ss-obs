import configparser
import datetime
config = configparser.ConfigParser()
"""
dbc : :obj:'DBConnector'
    DBConnector object used to connect to the database.  
    Configured with the connection information specified in the config.ini file under dbc/mysql_midas_options.
store_name : str
    Name of HDF5 store used to store the midas data locally.
meta_hdf_key : str
    Key of the object within HDF5 store containing metadata.
obs_hdf_key : str
    Name of the object within HDF5 store containing weather observation data.
default_earliest_date : date
    If no dates are specified for loading, this is the default start date to fall back onto.  The default end date is the date today.
"""
config['dbc'] = {}
config['dbc']['mysql_pvstream_options'] = "C:/Users/Dan Travers/Google Drive/Projects/Setup_data/mysql_defaults.ssfdb2.readwrite.pvstream"
config['hdf5'] = {}
config['hdf5']['store_name'] = "power.h5"
config['hdf5']['store_path'] = ""
config['hdf5']['meta_hdf_key'] = 'metadata'
config['hdf5']['obs_hdf_key'] = 'obs'
config['query_settings'] = {}
config['query_settings']['ss_id_metadata_table'] = 'pvstream.view_system_params_grouped_op_at'
config['query_settings']['ss_id_metadata_cols'] = 'ss_id, urn, latitude, longitude, orientation, tilt, orientation_assessed, \
tilt_assessed, kWp, area, array_count, operational_at' # column names to be queried from the database table/view
config['query_settings']['enphase_min'] = '148'
config['query_settings']['enphase_max'] = '2405'
config['query_settings']['ss_id_metadata_object_cols'] = 'ss_id, urn, latitude, longitude, orientation, tilt, orientation_assessed, \
tilt_assessed, kWp, area, array_count, operational_at' # columns saved to dataframe (currently all)
config['query_settings']['default_earliest_date'] = '2010-01-01'
config['query_settings']['ss30_batch_size'] = '30'
with open('pvstream.ini', 'w') as configfile:
    config.write(configfile)

