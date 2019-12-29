import configparser
import datetime
import os
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
ro_distinct_cols : str
    String of the columns to select from table ro_distinct.  String is inserted into the sql query.
ro_distinct_object_cols : str
    Column headings of the irradiance part of weather observations.
wh_cols : str
    String of the columns to select from table wh.  String is inserted into the sql query.
wh_object_cols : str
    Column headings of weather variabes except irradiance in the self.observations dataframe.
metadata_cols : str
    Columns to query from the metadata table (SRCE) in Midas db. 
metadata_object_cols : str
    Column headings of the metadata dataframe in the Midas python object.
version_num : str
    String to be used to select the version to be used in table wh.  String is inserted into the sql query.
met_domain_names : str
    List of the domain names to use in the query of table wh, to avoid querying data from other domain names which sometimes duplicate information.
default_earliest_date : date
    If no dates are specified for loading, this is the default start date to fall back onto.  The default end date is the date today.
"""
config['dbc'] = {}
config['dbc']['mysql_midas_options'] = "C:/Users/DanTravers/Google Drive/Projects/Setup_data/mysql_defaults.ssfdb2.readwrite.midas"
config['hdf5'] = {}
config['hdf5']['store_name'] = "midas.h5"
config['hdf5']['store_path'] = "C:\\Users\\DanTravers\\Documents\\dbs\\hdf5"
config['hdf5']['meta_hdf_key'] = 'metadata'
config['hdf5']['obs_hdf_key'] = 'obs'
config['query_settings'] = {}
config['query_settings']['ro_distinct_cols'] =  'src_id, ob_end_time, glbl_irad_amt'
config['query_settings']['ro_distinct_object_cols'] = 'site_id, ob_end_time, irr'
config['query_settings']['wh_cols'] = 'src_id, ob_time, wind_direction, wind_speed, air_temperature, rltv_hum'
config['query_settings']['wh_object_cols'] = 'site_id, ob_end_time, wind_dir, wind_speed, air_temp, rltv_hum'
config['query_settings']['metadata_cols'] = 'src_id, src_name, high_prcn_lat, high_prcn_lon'
config['query_settings']['metadata_object_cols'] = 'site_id, src_name, lat, long'
config['query_settings']['version_num'] = '1'
config['query_settings']['met_domain_names'] = "'AWSHRLY', 'AWSWIND', 'ESAWWIND', 'HWNDAUTO', 'WINDMEAN', 'SYNOP', 'NCM', 'HSUN3445', 'DLY3208', 'CAWS', 'METAR'"
config['query_settings']['default_earliest_date'] = '2018-01-01'
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config/midas.ini'), 'w') as configfile:
    config.write(configfile)

