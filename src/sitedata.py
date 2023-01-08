## Parent class for sites data objects
# Child classes include Midas and Power classes

# Dan Travers
# 29/1/19
# Potential change would be to store as numpy arrays and convert to dataframe on extraction.

import pandas as pd
import numpy as np
import os
import datetime
import configparser
from datetime import timedelta

class SiteData:
    """ Stores metadata and observation data for sites (PV generation, load sites or weather stations) 
    and manages the extraction of observations for dates required.
    Inherited classes will implement specific extraction routines from db / source files.

    Attributes
        ----------
        metadata : dataframe
            The metadata dictionary is keyed on site_id and for each site_id contains (latitude, longitude).
            **We could add to this information such as start and end dates of the readings in db.  And last updated?
            ** do we put all the site_ids in the dataframe and then just have an "included" list in a separate object?
        obs : dataframe
            Three dimensional dataframe containing the readings of the sites in the metadata.
            Multi-indexed by site_id & datetime with the outturn / load in the single column.
        verbose: int
            Verbosity to control the level of printed output.  
            0 = None
            1 = Errors
            2 = Significant information
            3 = All messages and logging information
    """

    def __init__(self, verbose=2):
        """ init initalizes an object with no data.

        Notes 
        -----
        Other columns to query can be selected using the local_config parameter.
        """

        self.config = configparser.ConfigParser()
        self.periods_per_day = 24
        self.metadata = pd.DataFrame([])
        self.obs = pd.DataFrame([])
        self.default_earliest_date = datetime.datetime(2010, 1, 1).date()
        self.verbose = verbose

    def load_data(self, site_list, start_date=None, end_date=datetime.datetime.now().date()\
    , goto_db=''):
        """ Method to load metadata and observations from db or cache for specified date range into object.
        Method populates data in self.metadata and self.obs dataframes.
        The method always forces data in memory to be refreshed from the cache.  
        Data stored in memory is always overwritten.

        Parameters
        ----------
        site_list : :obj:'list' or :obj:'int'
            List of the site_ids to be populated in the Power object
        start_date : date
            Start date from which to load in data.  Loads data including and above start date.  
            Optional parameter - if no dates are mentioned, all data is loaded.
        end_date : date
            End date from which to load in data.  Loads data up to but not including end date
            Optional parameter - if no dates are mentioned, all data is loaded.
        goto_db : str
            Three-state parameter to indicate whether to go to the database to fetch data.
            Always - indicates always goto db, and don't look in cache.
            Never - indicates only look to cache for data and never query db.
            Empty or other string - default value - goto the db if values not in cache.
        """
        if not isinstance(start_date, datetime.date) or not isinstance(end_date, datetime.date):
            raise TypeError("start_date and end_date must be of type datetime.date.")
        self.load_metadata(site_list, goto_db)
        self.metadata.index.name = 'site_id'
        self.load_observations(site_list, start_date, end_date, goto_db)
        self.obs.index.name = 'site_id'

    def load_metadata(self, site_list, goto_db=''):
        """ Method to populate or append to the metadata object from a list of site_ids.

        Firstly interrogates the cached data and if not available there, goes to the db.

        Parameters
        ----------
        site_list : :obj:'list' or int
            List of the src_ids to add to the metadata of object.
        goto_db : str
            Three-state parameter to indicate whether to go to the database to fetch data.
            Always - indicates always goto db, and don't look in cache.
            Never - indicates only look to cache for data and never query db.
            Empty or other string - default value - goto the db if values not in cache.
        """
        hdf_config = self.config['hdf5']
        # Check for data in memory & remove any src_ids from list which are already in memory
        site_id_short = [site_id for site_id in site_list if site_id not in self.metadata.index.values] # ** replace with numpy.intersect?
        if site_id_short != []:
            # look for data in cache if goto_db parameter is not 'Always'
            if goto_db != 'Always': 
                try:
                    with pd.HDFStore(os.path.join(hdf_config['store_path'],hdf_config['store_name']), 'r') as hdf:
                        temp = hdf.select(hdf_config['meta_hdf_key'], where = 'index = site_id_short')
                    self.metadata = self.metadata.append(temp)
                    site_id_short = [site_id for site_id in site_id_short if site_id not in self.metadata.index.values]
                except KeyError:
                    pass
            # goto db to fetch remaining data if goto_db parameter is not equal to 'Never'
            if (goto_db  != 'Never') & (len(site_id_short) > 0):
                self.load_metadata_db(site_id_short)
            site_list_loaded = [site_id for site_id in site_list if site_id in self.metadata.index.values]
            self.myprint('Loaded metadata for {} / {} locations requested'.format(len(site_list_loaded), len(site_list)), 2)
        else:
            self.myprint("All sites' metadata requested already loaded.", 2)

    def load_metadata_db(self, site_list):
        """ Loads metadata for ss_id stations from db and appends to self.metadata

        Function is overridden for subclasses and not implemented in parent.

        Parameters
        ----------
        site_list : :obj:'list' of int
            List of site_ids to query from the database
        """
        raise NotImplementedError
    
    def load_observations(self, site_list, start_date=None, end_date=datetime.datetime.now().date()\
    , goto_db=''):
        """ Method to load observations from db or cache for specified date range into dataframe.
        Method populates data in self.obs dataframe.
        The method always forces data in memory to be refreshed from the cache.  
        Data stored in memory is always overwritten.

        Parameters
        ----------
        site_list : :obj:'list' of :obj:'int'
            List of the site_ids to be populated in the Power object
        start_date : date
            Start date from which to load in data.  Loads data including and above start date.  
            Optional parameter - if no dates are mentioned, all data is loaded.
        end_date : date
            End date from which to load in data.  Loads data up to but not including end date
            Optional parameter - if no dates are mentioned, all data is loaded.
        goto_db : str
            Three-state parameter to indicate whether to go to the database to fetch data.
            Always - indicates always goto db, and don't look in cache.
            Never - indicates only look to cache for data and never query db.
            Empty or other string - default value - goto the db if values not in cache.
        """
        hdf_config = self.config['hdf5']
        if start_date==None:
            start_date = self.default_earliest_date
        for site_id in site_list:
            site_start, site_end = fetch_start_end_dates(site_id, start_date, end_date)
            self.myprint('--site_ids: {}'.format(site_id), 2)
            # ** for each site look at the first and last dates in the obs df and get more data if required.
            try: 
                first_date = self.obs.loc[site_id, :].index.min().date() # date of first values in obs
                last_date = self.obs.loc[site_id, :].index.max().date()  # date of last values in obs
            except KeyError:
                first_date = last_date = site_end 
            if (site_start < first_date) or (site_end > last_date): # goto cache and/or db if date range not covered
                if goto_db != 'Always': 
                # look for data in cache if goto_db parameter is not 'Always'
                    temp = pd.DataFrame([])
                    with pd.HDFStore(os.path.join(hdf_config['store_path'],hdf_config['store_name']), 'r') as hdf:
                        temp = hdf.select(hdf_config['obs_hdf_key'], where = 'site_id = site_id')
                    if len(temp) > 0: 
                        self.obs = self.obs.append(temp)
                        self.obs = self.obs[~self.obs.index.duplicated(keep='first')]
                        first_date = self.obs.loc[site_id, :].index.min().date() # first_date is date of first values in df
                        last_date = self.obs.loc[site_id, :].index.max().date()  # last_date is date after values in df
                        self.myprint('Found & loaded {} observations: {} days, {} hours from cache.'.\
                        format(len(temp), \
                        np.trunc(len(temp)/self.periods_per_day), \
                        len(temp)-self.periods_per_day*np.trunc(len(temp)/self.periods_per_day)), 2)
                if goto_db  != 'Never':
                # goto db to fetch remaining data if goto_db parameter is not equal to 'Never'
                    if site_start < first_date:
                        self.load_obs_db(site_id, site_start, first_date)
                    if last_date < site_end:
                        self.load_obs_db(site_id, last_date, site_end)
                try:
                    check_missing_hours(self.obs.loc[site_id].tz_localize(None).\
                        loc[site_start.strftime('%Y%m%d'):(site_end-timedelta(1)).strftime('%Y%m%d')],\
                        site_start, site_end, 'From db & cache:', site_id, periods_per_day=self.periods_per_day, verbose_setting=self.verbose)
                except KeyError:
                    self.myprint('Key {} not loaded into observations'.format(site_id), 1)
                # ** The check missing hours here is for the Full data set (cache and db)
                # Doesn't look for small gaps in the data between the first and last dates visible.
                # This is becasue there are often small gaps of 1 day or so - too hard to find.
            
    def load_obs_db(self, site_id, start_date, end_date, graph=False):
        """ Method for loading data from the db for a single site_id.
        Method extended by subclasses to implement querying from db.

        Notes
        -----
        Data is appended directly to self.obs dataframe.  
        Method load includes all data including the start_date day.
        Method load excludes the data from the end_date
        (except midnight of that day, which is actually the last period of proceeding day.
        
        Parameters
        ----------
        site_id : int
            src_id being queried
        start_date : date
            Start date from which to load in data.  Loads data including and above start date.
        end_date : date
            End date from which to load in data.  Loads data up to but not including end date
        graph : Boolean
            Denotes if should generate a graph of the number of observations per day loaded.
            Graph allows user to see where missing observations are.
            When loading many sites however graphing consumes memory and can be confusing.
        """
        # Graph to show the daily data gathered from the db
        #if graph: 
        #    daily_data = self.obs.groupby(pd.Grouper(freq='D')).count()
        #    plt.figure(figsize=(16,1))
        #    plt.plot(daily_data)

    def save_to_hdf(self):
        """ Method to save observation data and metadata to hdf.
        No parameters are passed as all are class attributes.  
        """
        self.__append_to_hdf('meta')
        self.__append_to_hdf('obs')

    def __append_to_hdf(self, type):
        """ Method to save to hdf for each datatype in store.

        Method is called from save_to_hdf for both metadata and observation data in turn.

        Parameters
        ----------
        Type : str
        meta - saves the metadata to the midas store
        obs - saves the observation data to the midas store
        """
        
        hdf_config = self.config['hdf5']
        if type == 'meta':
            hdf_key = hdf_config['meta_hdf_key']
            data = self.metadata.copy()
        else:
            hdf_key = hdf_config['obs_hdf_key']
            data = self.obs.copy()

        with pd.HDFStore(os.path.join(hdf_config['store_path'],hdf_config['store_name']), 'r') as hdf:
            try:
                current = hdf.get(hdf_key)
                data = data.append(current)
                data = data[~data.index.duplicated(keep='last')]
                self.myprint('{} rows already in hdf {} /{}'.format(current.shape[0], os.path.join(hdf_config['store_path'],hdf_config['store_name']), hdf_key), 2)
            except KeyError: 
                self.myprint('     New hdf file {}'.format(hdf_key), 3)
            hdf.close()
        with pd.HDFStore(os.path.join(hdf_config['store_path'],hdf_config['store_name']), 'a') as hdf:    
            hdf.put(hdf_key, data, format='Table', data_columns=True)
            #hdf.get_storer(hdf_key).attrs.time_updated=datetime.datetime.utcnow()
            self.myprint('Saved {} rows to hdf store {} /{}'.format(data.shape[0], os.path.join(hdf_config['store_path'],hdf_config['store_name']), hdf_key), 1)
            hdf.close()

    def get_obs(self, start_date, end_date, freq):
        # function to return a dataframe of data aggregated to appropriate level 
        pass

    def store_summary(self):
        """ Function to return the summary of the observation data stored in the cache.
        The result will contain a row for each site_id and the start and end dates of data stored.
        As usual, data includes the start date, but is all data before the end date.
        """
        hdf_config = self.config['hdf5']
        with pd.HDFStore(os.path.join(hdf_config['store_path'],hdf_config['store_name']), 'r') as hdf:
            index = hdf.select(hdf_config['obs_hdf_key']).index     
        temp = pd.DataFrame(index=index).reset_index(level=1)
        temp.columns = ['datetime']
        return(temp['datetime'].dt.date.groupby('site_id').agg([min, max]))

    def myprint(self, message, message_priority):
        """ Method to print messages filtered by verbosity.
        The verbosity setting is a class attribute and should be in range [0:3]

        parameters
        ----------
        message : str
            Message to be printed (or not if not important enough).
        message_priority : int
            Nothing should have priority 0, so there is a completely queit mode. 
            Errors should be sent with message_priority = 1, 
            Serious information / warning, message_priority = 2, 
            Information, message_priority = 3.
        """

        if message_priority<=self.verbose:
            print(message)

def fetch_start_end_dates(id, start_date, end_date):
    """ function to return start and end dates for specific site based on 
    either the date dictionary or the generic start and end dates 
    """

    start = start_date
    end = end_date
    return(start, end)

def check_missing_hours(df, start_date, end_date, type='-', id='Missing ID', periods_per_day=24, verbose_setting=3):
    """ Utility function to print statements on how many dates have been found
    ** This has a slight bug in that if the dataframe sent through is trimmed by dates, the last
    hh of the final day isn't sent through and it can misleadingly show 1 period missing.
    Not a problem as the data is really there.
    """
    if 3<=verbose_setting: 
        print('{} {}: {} ({} days & {} hours) full observations from {} ({} days) requested'.\
          format(type, id, len(df), np.trunc(len(df)/periods_per_day), len(df)-periods_per_day*np.trunc(len(df)/periods_per_day), periods_per_day*(end_date - start_date).days, \
                 (end_date-start_date).days))
    if 3<=verbose_setting:
        print('     {} {}: Missing {} observations from requested period'.\
              format(type, id, periods_per_day*(end_date-start_date).days - len(df)))
    return

def convert_load_col_names(col):
    col_convert = { ' meter_mpan' : 'site_id', \
        'mpan' : 'site_id', \
        ' read_date' : 'date', \
        'date' : 'date', \
        'settlementdate' : 'date'}
    if type(col)==datetime.time: 
        return('t' + str(int(col.hour*2 + col.minute/30 + 1)))
    elif type(col) == np.float64:
        return('t' + str('%.0f'%(col)))
    elif col.strip()[0:3]=='hh0':
        return('t' + col.strip()[3:4])
    elif col.strip()[0:2]=='hh':
        return('t' + col.strip()[2:4])
    else:
        try:
            float(col)
            return('t' + col)
        except:
            return(col_convert[col.lower()])