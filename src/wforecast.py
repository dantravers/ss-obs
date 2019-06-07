## Classes for getting forecast weather data

# Dan Travers
# 19/04/19
# Inherited from super class SiteData implements nwp data observations
# Potential change would be to store as numpy arrays and convert to dataframe on extraction.

import pandas as pd
import numpy as np
import os
import datetime
import configparser
from datetime import timedelta
import sys
from sitedata import SiteData, check_missing_hours
from midas import Midas
from location_data_utils import read_netcdf_file

class WForecast(Midas): 
    """ Stores numerical weather predictions and relevant meta-data
    
    Default config settings stored in wforecast.ini file, which provides hdf store configurations.
    Config setting can be overridden by passing in dictionary at initialization.

    Attributes
    ----------
    metadata : dataframe
        The metadata dictionary is keyed on site_id and for each site_id contains (latitude, longitude).
    obs : dataframe
        Dataframe with all weather variables from the nwp file.
        Dataframe is indexed by id, fcst_base, datetime, ahead.
        ahead is an integer for the number of days between the forecast base and the datetime.
    """

    def __init__(self, verbose=2, local_config={}):
        """ init initalizes an object with no data.

        Notes 
        -----
        Other columns to query can be selected using the local_config parameter.

        Parameters
        ----------
        config : dict
            A dictionary to override config parameters. Should be a dict of dicts where outer dict
            denotes section in the config file and inner dicts are inidividual parameters.
            Contents are described in the midas_config_writer.py module.
        """
        super(WForecast, self).__init__(verbose)
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config/wforecast.ini'))
        # update config file for any local configs passed in:
        for section in self.config:
            if section in local_config:
                self.config[section].update(local_config[section])
        self.default_earliest_date = datetime.datetime.strptime(self.config['query_settings']['default_earliest_date'], '%Y-%m-%d').date()
        self.forecast_max_days_ahead = int(self.config['query_settings']['forecast_max_days_ahead'])
        self.path = self.config['query_settings']['path']

    def load_data(self, locations, path='C:\\Users\\Dan Travers\\Documents\\dbs\\weather\\ecmwf', \
        start_date=None, end_date=datetime.datetime.now().date(), goto_file='Cache'):
        """ Method to load observations from netCDF files or cache for specified date range into object.
        Method populates data in self.obs dataframe and metadata is derived.
        The method always forces data in memory to be refreshed from the cache if available.  
        If 

        Parameters
        ----------
        locations : DataFrame
            DataFrame containing a row for each location requested.  
            The frame must have either a column named "f_id", or two columns named "latitude" and 
            "longitude".  Where there is a choice, the method looks first for the presence of f_id, 
            and if not uses the latitude and longitudes.
            This frame is the format as output by the function get_fcst_locs (and cross_locations).
        path : str
            Path location of the (netCDF) files if queried from disk.
        start_date : date
            Start date from which to load in data.  Loads data including and above start date.  
            Optional parameter - if no dates are mentioned, all data is loaded.
        end_date : date
            End date from which to load in data.  Loads data up to but not including end date
            Optional parameter - if no dates are mentioned, all data is loaded.
        goto_file : str
            Takes values "File" or "Cache".  If Cache, the values are retrieved from the cache, 
            overwriting all other data in memory.  
        
        Returns
        -------
        Dataframe with all weather variables from the nwp file.
        Dataframe is indexed by id, fcst_base, datetime, ahead.
        ahead is an integer for the number of days between the forecast base and the datetime.
        """
        hdf_config = self.config['hdf5']
        if start_date==None:
            start_date = self.default_earliest_date
        if not isinstance(start_date, datetime.date) or not isinstance(end_date, datetime.date):
            raise TypeError("start_date and end_date must be of type datetime.date.")
        if not 'f_id' in locations.columns:
            locations['f_id'] = locations.latitude.map(str) + ':' + locations.longitude.map(str)
        date_list = [start_date - timedelta(self.forecast_max_days_ahead) + datetime.timedelta(days=x) \
            for x in range(0, (end_date-(start_date-timedelta(self.forecast_max_days_ahead))).days)]
        # ** Should later improve this below check to check for missing dates per id, so you have a 2-d matrix
        if goto_file == 'None':
            pass
        else:
            if len(self.obs)>0: 
                date_list = [x for x in date_list if x not in self.obs.index.get_level_values('fcst_base').unique().date]
            if len(date_list) > 0: 
            # find data in cache and merge with data in-memory
                if goto_file == 'Cache':  
                    temp = pd.DataFrame([])
                    original_len = len(self.obs)
                    id_list = locations.f_id.tolist()
                    with pd.HDFStore(os.path.join(hdf_config['store_path'],hdf_config['store_name']), 'r') as hdf:
                        temp = hdf.select(hdf_config['obs_hdf_key'], where = 'site_id in id_list')    # change the query to be site_id in site_list
                    if len(temp) > 0: 
                        self.obs = self.obs.append(temp)
                        no_dups = np.sum(self.obs.index.duplicated())
                        self.obs = self.obs[~self.obs.index.duplicated(keep='first')]
                    self.myprint('Loaded {} entries from cache, merged with {} rows of data in memory with {} duplicates\
                            for total of {} rows.'.format(len(temp), original_len, no_dups, len(self.obs)), 2)
                # toto file to find data and merge with data in memory    
                elif goto_file == 'File':
                    original_len = len(self.obs)
                    load_count = 0
                    time_sep = '%%3A' if os.name == 'nt' else ':'
                    for file in os.listdir(path):
                        if datetime.datetime.strptime(file[5:-3], '%Y-%m-%dT%H{0}%M{0}%S'.format(time_sep)).date() in date_list: 
                            day = read_netcdf_file(file, path, locations)
                            load_count += len(day)
                            if self.obs.shape[0]==0:
                                self.obs = day.copy()
                            else:
                                self.obs = self.obs.append(day)
                    no_dups = np.sum(self.obs.index.duplicated())
                    self.obs = self.obs[~self.obs.index.duplicated(keep='first')]
                    self.myprint('Loaded {} rows data from files, merged with existing {} rows '\
                        'with {} duplicates for total of {} rows.'.format(load_count, original_len, no_dups, len(self.obs)), 2)
                    if load_count > 0:
                        new_metadata = pd.DataFrame([], index=day.index.get_level_values('site_id').unique())
                        new_metadata['latitude'] = new_metadata.index.to_series().str.split(':').str[0]
                        new_metadata['longitude'] = new_metadata.index.to_series().str.split(':').str[1]
                        self.metadata = self.metadata.append(new_metadata)
                        self.metadata = self.metadata[~self.metadata.index.duplicated(keep='first')]
                else:
                    self.myprint("goto_file parameter must be 'Cache', 'File' or 'None'", 0)


    def get_obs(self, days_ahead = 1, freq='1H'):
        """ Function to return a dataframe of weather forecast data sliced to specified number of days ahead only, 
        and aggregated to requested frequency level.

        Notes
        -----
        Currently the function only extract data stored in 1H frequency.

        Parameters
        ----------
        days_ahead : list:int
            The number of days ahead to fetch the forecast data for.  E.g. 1 means the data for the day ahead
            forecast is collected.
        freq : str
            Frequency at which the data should be returned to the user.  The only supported formats are
            currently 30m and 1H.
        """       

        if freq=='1H':
            if len(self.obs)>0:
                idx = pd.IndexSlice
                df = self.obs.loc[idx[:, :, :, timedelta(days_ahead)], :].reset_index(['fcst_base', 'ahead'], drop=True)
                df.irr = df.irr.diff() # take diff.  This needs to be reconsidered for the forecast 4 days out, where hour steps up in 3's
                # remove the NaN in first entry and the negative values when you to from one forecast base to the next:
                df.loc[df.index.get_level_values('datetime').hour==0, 'irr'] = 0 
                df.loc[df.irr<0] = 0 # remove very small negative values which appear at times.
            else: 
                df = pd.DataFrame([])    
            return(df)
        else: 
            self.myprint('Invalid frequency to request for weather data.', 1) 
            return(None)