## Classes for getting power data inherited from Sites class

# Dan Travers
# 29/1/19
# Implement Power object as child class of SiteData class
# Potential change would be to store as numpy arrays and convert to dataframe on extraction.

import pandas as pd
import numpy as np
import os
import datetime
import configparser
from datetime import timedelta
from dbconnector import DBConnector
import sys
from sitedata import SiteData, check_missing_hours

class Power(SiteData):
    """ Stores power readings from ss_id database.
    
    Parameterization can be used to setup the instance of the object to store 30 min or 5 min data

    Attributes
    ----------
        metadata : dataframe
            The metadata dictionary is keyed on site_id and for each site_id contains (latitude, longitude).
            **We could add to this information such as start and end dates of the readings in db.  And last updated?
            ** do we put all the site_ids in the dataframe and then just have an "included" list in a separate object?
        obs : dataframe
            Three dimensional dataframe containing the readings of the sites in the metadata.
            Multi-indexed by site_id & datetime with the outturn / load in the single column.
    """

    def __init__(self, verbose=3, power_type='30min_PV', local_config={}):
        """ init initalizes an object with no data.

        Notes 
        -----
        Other columns to query can be selected using the local_config parameter.

        Parameters
        ----------
        power_type : str
            Parameter to define which type of power data is to be stored. The default is 30min_PV, 
            which means 30 min data from pvstream db.  Other options are 5min data from pvstream and 
            load data (30min).
        local_config : dict
            A dictionary to override config parameters. Should be a dict of dicts where outer dict
            denotes section in the config file and inner dicts are inidividual parameters.
            Contents are described in the power_config_writer.py module.
        """
        super(Power, self).__init__(verbose)
        self.config.read('C:/Users/Dan Travers/Documents/GitHub/ss-obs/ss-obs/pvstream.ini')
        # update config file for any local configs passed in:
        for section in self.config:
            if section in local_config:
                self.config[section].update(local_config[section])
        self.default_earliest_date = datetime.datetime.strptime(self.config['query_settings']['default_earliest_date'], '%Y-%m-%d').date()
        self.dbc = DBConnector(self.config['dbc']['mysql_pvstream_options'], session_tz="UTC")
        self.power_type = power_type
        self.periods_per_day = (24 * 60) / int(self.power_type[0:2])

    def load_metadata_db(self, site_list):
        """ Overrides function in superclass. 
        Loads metadata for site_list (ss_id stations in the subclass) from db and append to self.metadata

        Function is called after looking for data in cache.  
        The columns and tables to query are supplied in config parameters.

        Parameters
        ----------
        site_list : :obj:'list' of int
            List of site_ids to query from the database
        """

        query_config = self.config['query_settings']      
        select_sql = ("SELECT {} FROM {} where ss_id in ({})")\
                    .format(query_config['ss_id_metadata_cols'], query_config['ss_id_metadata_table'], ",".join(map(str, site_list)))
        meta_cols = [x.strip() for x in query_config['ss_id_metadata_object_cols'].split(',')]
        meta = pd.DataFrame(self.dbc.query(select_sql), columns=meta_cols)
        if len(meta)>0:
            meta[(meta.ss_id >= int(query_config['enphase_min'])) & (meta.ss_id < int(query_config['enphase_max']))]\
            ['orientation'] = meta['orientation_assessed']
            meta[(meta.ss_id >= int(query_config['enphase_min'])) & (meta.ss_id < int(query_config['enphase_max']))]\
            ['tilt'] = meta['tilt_assessed']
            meta = meta.rename(columns={'ss_id' : 'site_id'}).set_index('site_id')
            meta = meta.drop(['orientation_assessed', 'tilt_assessed'], axis=1)
            self.myprint('Extracted {} rows of midas metadata from db for site_ids: {}'.format(len(meta), str(site_list).strip('[]')), 2) 
            self.metadata = self.metadata.append(meta, sort=False)
        else:
            self.myprint('No metadata extracted for site_ids: {}'.format(str(site_list).strip('[]')), 2) 

    def load_obs_db(self, site_id, start_date, end_date, graph=False):
        """ Method for loading data from the db for a single site_id.
        The method calls the appropriate loading db method based on the power_type attribute.
        
        Notes
        -----
        Data is appended directly to self.obs dataframe.  
        Method load includes all data including the start_date day.
        Method load excludes the data from the end_date
        (except midnight of that day, which is actually the last period of proceeding day.
        
        Parameters
        ----------
        site_id : int
            ss_id being queried
        start_date : date
            Start date from which to load in data.  Loads data including and above start date.  
        end_date : date
            End date from which to load in data.  Loads data up to but not including end date
        graph : Boolean
            Denotes if should generate a graph of the number of observations per day loaded.  
            Graph allows user to see where missing observations are.  
            When loading many sites however graphing consumes memory and can be confusing.
        """

        self.myprint("Querying pvstream db for {}.".format(site_id), 3)
        if self.power_type == '30min_PV':
            self.__load_ss30_db(site_id, start_date, end_date)
        else:
            self.myprint('Unsupported power_type in Power object', 1)
        super(Power, self).load_obs_db(site_id, start_date, end_date, graph)

    def get_obs(self, freq='1H'):
        """ function to return a dataframe of observation data aggregated to requested frequency level 

        Notes
        -----
        Currently the function is hardcoded to only deal with data stored in 30m frequency, which can be 
        returned at 30m and 1H frequency.
        Needs some generalization if it is to be expanded to 5 min data storage.

        Parameters
        ----------
        freq : str
            Frequency at which the data should be returned to the user.  The only supported formats are
            currently 30m and 1H.
        """
        
        if freq == '1H': 
            pvhourly = pd.DataFrame(self.obs.reset_index(level='site_id').groupby('site_id')['outturn'].resample('1H',closed='right', loffset='1H').sum())
            # ensure that any NaNs are reflected in the output df, and then dropped (I.e. not hidden in the resampling)
            pvout = pd.merge(pvhourly, self.obs, how='left', left_index=True, right_index=True, suffixes=['', 'hh'])
            pvout.apply(fillnans, axis=1)
            pvout.drop(['outturnhh'], axis=1, inplace=True)
            pvout = pvout.dropna()
        if (freq == '30m') & (self.power_type == '30min_PV'): 
            pvout = self.obs.dropna()
        return(pvout)

    def __load_ss30_db(self, site_id, start_date, end_date):
        """ Function queries data from pvstream reading30compact the data for specific ss_id between 
        date range specified.  Data is returns in half-hourly granularity in a stacked dataframe indexed
        by ss_id and datetime, with just one column of data: "outturn".
        """
        pvs = pd.DataFrame([])
        batch_size = int(self.config['query_settings']['ss30_batch_size'])
        pvcompact_cols = ['date', 'ss_id', 't1', 't2', 't3', 't4', 't5', 't6', 't7','t8', 't9', 't10', 't11', \
                        't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', \
                        't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', \
                        't36', 't37', 't38', 't39', 't40', 't41', 't42', 't43', 't44', 't45', 't46', 't47', 't48']
        # extract raw data from db in wide format:
        batch_end_date = end_date 
        while (batch_end_date > start_date): 
            batch_start_date = max(start_date, batch_end_date - timedelta(days=batch_size))
            select_sql = ("SELECT {} FROM pvstream.reading30compact where "\
                        "ss_id in ({}) and "\
                        "date >= {} and "\
                        "date < {};"\
                        .format(','.join(e for e in pvcompact_cols), site_id, \
                                batch_start_date.strftime("'%Y-%m-%d'"), batch_end_date.strftime("'%Y-%m-%d'")))
            result = self.dbc.query(select_sql)
            df_temp = pd.DataFrame(result, columns=pvcompact_cols)
            df_temp.rename({'ss_id' : 'site_id'}, axis=1,  inplace=True)
            pvs = pvs.append(df_temp, ignore_index=True)
            batch_end_date -= datetime.timedelta(days=batch_size)        
        # stack wide format data:
        if pvs.empty:
            pvflat=pd.DataFrame([])
            self.myprint('{} - no data in db between {} and {}'.\
            format(site_id, start_date.strftime("'%Y-%m-%d'"), end_date.strftime("'%Y-%m-%d'")), 1)
        else:
            pvflat = pvs.set_index(['site_id', 'date'],drop=True).stack().reset_index()
            pvflat['hh'] = pvflat.apply(lambda ser: float(ser['level_2'][1:]), axis=1)
            pvflat['mins'] = pvflat.hh * 30
            pvflat['datetime'] = pvflat.apply(lambda x: x['date'] + datetime.timedelta(minutes = x['mins']), axis=1)
            pvflat.rename(index = {}, columns = {0: 'outturn'}, inplace=True)
            pvflat = pvflat.set_index(['site_id', 'datetime'],drop=True)
            pvflat.drop(['date', 'level_2', 'hh', 'mins'], axis=1, inplace=True)
            pvflat = pvflat.tz_localize('UTC', level=1)
            self.myprint('Loaded {} rows of pv data from pvstream db'.format(pvflat.shape[0]), 3) 
            # report missing hours and append to Power.obs dataframe:
            check_missing_hours(pvflat.loc[site_id, :], start_date, end_date, 'From db:', site_id, periods_per_day=self.periods_per_day, verbose_setting=self.verbose)
            pvflat = pvflat.apply(np.float64)
            self.obs = self.obs.append(pvflat).sort_index()
            self.obs = self.obs[~self.obs.index.duplicated(keep='last')].sort_index()

def fillnans(x):
    if np.isnan(x[1]):
        x[0] = np.NaN 
    return x