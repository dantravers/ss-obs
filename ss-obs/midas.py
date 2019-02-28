## Classes for getting weather data

# Dan Travers
# 29/01/19
# Inherited from super class SiteData implements Midas data observations
# Potential change would be to store as numpy arrays and convert to dataframe on extraction.

import pandas as pd
import numpy as np
import os
import datetime
import configparser
from datetime import timedelta
from dbconnector import DBConnector
import sys
sys.path.append('C:/Users/Dan Travers/Documents/GitHub/ss-obs/ss-obs')
sys.path.append('ss-obs/')
from sitedata import SiteData, check_missing_hours

class Midas(SiteData): 
    """ Stores weather data readings from Midas stations and relevant meta-data
    
    Default config settings stored in midas.ini file, which provides db connection parameter path and columns to query from db.
    Config setting can be overridden by passing in dictionary at initialization.


    Attributes
    ----------
    metadata : dataframe
        The metadata dictionary is keyed on site_id and for each site_id contains (latitude, longitude).
        **We could add to this information such as start and end dates of the readings in db.  And last updated?
        ** do we put all the src_ids in teh dictionary and then just have an "included" list in a separate object?
    obs : dataframe
        Three dimensional dataframe containing the readings of the src locations in the dictionary.
        Multi-indexed by src_id & datetime (hourly) and with the variables as columns.
    """

    def __init__(self, local_config={}):
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
        super(Midas, self).__init__()
        self.config = configparser.ConfigParser()
        self.config.read('midas.ini')
        # update config file for any local configs passed in:
        for section in self.config:
            if section in local_config:
                self.config[section].update(local_config[section])
        self.dbc = DBConnector(self.config['dbc']['mysql_midas_options'], session_tz="UTC")
        self.default_earliest_date = datetime.datetime.strptime(self.config['query_settings']['default_earliest_date'], '%Y-%m-%d').date()

    def load_metadata_db(self, site_list):
        """ loads metadata for midas stations from db and appends to self.metadata

        Function is called after looking for data in cache.  
        The columns and tables to query are supplied in config parameters.

        Parameters
        ----------
        site_list : :obj:'list' of int
            List of src_ids to query from the database
        
        """
        query_config = self.config['query_settings']      
        select_sql = ("SELECT {} FROM midas.SRCE where src_id in ({}) ")\
                    .format(query_config['metadata_cols'], ",".join(map(str, site_list)))
        meta_cols = [x.strip() for x in query_config['metadata_object_cols'].split(',')]
        meta = pd.DataFrame(self.dbc.query(select_sql), columns=meta_cols)
        if len(meta)>0:
            meta = meta.set_index('site_id')
            print('Extracted {} rows of midas metadata from db for src_ids: {}'.format(len(meta), str(site_list).strip('[]'))) # verbosity=2
            self.metadata = self.metadata.append(meta, sort=False)
        else:
            print('No metadata extracted for src_ids: {}'.format(str(site_list).strip('[]'))) #verbosity=2

    def load_obs_db(self, src_id, start_date, end_date, graph=False):
        """ Method to load the midas weather observations from db.

        Notes
        -----
        Data is appended directly to self.obs dataframe.  
        Method load includes all data including the start_date day.
        Method load excludes the data from the end_date
        (except midnight of that day, which is actually the last period of proceeding day.
        
        Parameters
        ----------
        src_id : int
            src_id being queried
        start_date : date
            Start date from which to load in data.  Loads data including and above start date.  
        end_date : date
            End date from which to load in data.  Loads data up to but not including end date
        """

        irr = self.__load_midas_db_irr(src_id, start_date, end_date)  # load data from irradiance table
        wh = self.__load_midas_db_weather_ex_irr(src_id, start_date, end_date)  # load data from weather table
        temp = pd.merge(irr, wh, how='inner', left_on='ob_end_time', right_on='ob_end_time', suffixes=['','wh'])
        temp.set_index(['ob_end_time'], inplace=True)
        temp = temp.tz_localize('UTC')
        temp.drop(['site_idwh'], axis=1, inplace=True)
        check_missing_hours(temp, start_date, end_date, 'From db: src_id: ', src_id) # verbosity=3
        temp = temp.set_index('site_id', append=True).swaplevel()
        self.obs = self.obs.append(temp, sort=False)
        self.obs = self.obs[~self.obs.index.duplicated(keep='last')]
        super(Midas, self).load_obs_db(src_id, start_date, end_date, graph)

    def __load_midas_db_irr(self, src_id, start_date, end_date):
        """ Function returns the midas irradiance observations from db
        
        Parameters
        ----------
        src_id : int
            src_id being queried
        start_date : date
            Start date from which to load in data.  Loads data including and above start date.  
            Optional parameter - if no dates are mentioned, all data is from default_earliest_date.
        end_date : date
            End date from which to load in data.  Loads data up to but not including end date
            Optional parameter - if no dates are mentioned, all data is loaded until today.
        
        Returns
        -------
        Dataframe containing the readings with datetime index and columns for src_id and irradiance.
        """
        query_config = self.config['query_settings']
        select_sql = ("SELECT {} FROM midas.RO_distinct where "\
            "src_id = {} and "\
            "ob_end_time > {} and "\
            "ob_end_time <= {};"\
                    .format(query_config['ro_distinct_cols'], \
                            src_id, \
                            start_date.strftime("'%Y-%m-%d'"), \
                            end_date.strftime("'%Y-%m-%d'")))
        irr_cols = [x.strip() for x in query_config['ro_distinct_object_cols'].split(',')]
        irr = pd.DataFrame(self.dbc.query(select_sql), columns=irr_cols)
        if irr.duplicated([irr_cols[1]]).sum() > 0: 
            if irr.duplicated().sum() > 0: 
                print("{} Duplicate rows removed in query of RO_distinct for src_id = {}".\
                    format(irr.duplicated().sum(), src_id)) # verbosity=3
            else:
                print("***{} Duplicate observation times BUT different data removed in query of RO_distinct for src_id = {}".\
                    format(irr.duplicated().sum(), src_id)) # verbosity=3
            irr = irr[~irr.duplicated([irr_cols[1]])]
        print('Loaded {} rows of irradiance data from RO_distinct'.format(irr.shape[0])) # verbosity=3
        return(irr)

    def __load_midas_db_weather_ex_irr(self, src_id, start_date, end_date):
        """ Function returns the midas weather observations from db except irradiance
        
        Notes
        -----
        Weather variables to be queried and the names of the weather variables in the result are defined
        as config parameters in class variables.

        Parameters
        ----------
        src_id : int
            src_id being queried
        start_date : date
            Start date from which to load in data.  Loads data including and above start date.  
            Optional parameter - if no dates are mentioned, all data is from default_earliest_date.
        end_date : date
            End date from which to load in data.  Loads data up to but not including end date
            Optional parameter - if no dates are mentioned, all data is loaded until today.
        
        Returns
        -------
        Dataframe containing values: datetime index and column for each weather variable observed.
        """
        query_config = self.config['query_settings']
        select_sql = ("SELECT {} FROM midas.WH where "\
            "src_id = {} and "\
            "version_num = {} and "\
            "met_domain_name in ({}) and "\
                    "ob_time > {} and "\
            "ob_time <= {};"\
                    .format(query_config['wh_cols'], \
                            src_id, \
                            query_config['version_num'], \
                            query_config['met_domain_names'], \
                            start_date.strftime("'%Y-%m-%d'"), \
                            end_date.strftime("'%Y-%m-%d'")))
        wh_cols = [x.strip() for x in query_config['wh_object_cols'].split(',')]
        wh = pd.DataFrame(self.dbc.query(select_sql), columns = wh_cols)
        # check for duplicates
        if wh.duplicated([wh_cols[1]]).sum()>0: 
            if wh.duplicated().sum() > 0: 
                print("{} Duplicate rows removed in query of WH for src_id = {}".format(wh.duplicated().sum(), src_id)) # verbosity=3
            else:
                print("***{} Duplicate observation times BUT different data removed in query of WH for src_id = {}".\
                    format(wh.duplicated().sum(), src_id)) # verbosity=3
            wh = wh[~wh.duplicated([wh_cols[1]])]
        print('{} rows of weather data from WH'.format(wh.shape[0])) # verbosity=3
        return(wh)
    
    def get_obs(self, freq='1H'):
        """ function to return a dataframe of observation data aggregated to requested frequency level.
        Function is overridden for the Forecast class, where forecast_hours_ahead is necessary.

        Notes
        -----
        Currently the function is hardcoded to only deal with data stored in 1H frequency.

        Parameters
        ----------
        freq : str
            Frequency at which the data should be returned to the user.  The only supported formats are
            currently 30m and 1H.
        """       
        if freq=='1H':
            return(self.obs)
        else: 
            print('Invalid frequency to request for weather data.') # verbosity=1
            return(None)