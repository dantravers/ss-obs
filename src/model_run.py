# Module containing the ModelRun class
# This class builds up the necessary data and runs a cross validation using chosen ML
# method.  
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
import datetime
from datetime import timedelta
from power import Power
import wforecast
import midas
from midas import Midas
from wforecast import WForecast
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import get_extra_radiation
from model_definition import ModelDefinition
from pv_ml_models import cross_validate_grouped
from stats_funcs import generate_error_stats, scatter_results
from location_data_utils import apply_weekday

class ModelRun:
    """ 
    Configuration and running of machine learning.
    Initializing object collects all data, cleans, creates features.
    Methods such as cross-validate to run model.
    
    Parameters
    ----------
    power_list : obj:'list' of int
        A single ss_id or a set of ss_id's is collected and a weighted sum created (weighted to ensure
        all sites have equal contribution).
    wh_list : obj:'list' of int
        List of integer identifiers of weather locations to be used for the feature set.  
        Could be src_ids, or forecast weather locations.
    power_data : obj:'Power'
        Pass in a reference to instance of the Power object to populate data into. 
    wh_data : obj:'Midas' or 'WForecast'
        Pass in a reference to instance of the Midas object if modelling on actuals, 
        or a 'WForecast' object is modelling on forecast weather.
    model_def : obj:'ModelDefinition'
        Object with the definition of the type and the parameters of the machine learning model and cross-validation.  
        The kwargs contain the parameters which vary between machine learning model.
    start_date : Date
        Start date of analysis
    end_date : Date
        End date of anlaysis
    forecast_days_ahead : int
        Optional attribute.  If forecast weather rather than actual weather is being used, 
        the number of days ahead of the forecast relative to the forecast base.
    power_sum_normalized : Boolean
        If True, the power from the list (assuming len>1) is averaged to create the target.  
        If False, the power from the list (assuming len>1) is summed to create the target.
    observation_freq : 'str'
        Frequency of observations used as input to use in analysis.  Default to 1H. 
    solar_geometry : Three states {'solar', 'both', ''}
        If 'solar' or 'both', create solar geometry features from the sun's location and site metadata.
        If 'solar', don't craete the month & hour features.
        If '', just use the month-hour features.
    feature_list : obj:'list' of str
        List of strings to use as features.  This must be a subset of the complete set of features. 
        The full set of weather features is: ['irr', 'wind_dir', 'wind_speed', 'air_temp', 'rltv_hum'].
        Note that when lagged or rolling average features are requested, the original feature may still be removed.
    lagged_variables : Dictionary
        Dictionary of features and list of lags for each feature. 
        These lags (can be forwards or backwards) are added to the feature set.
        Positive lags means the variable from the hour *before* is used.
    daylight_hours : obj:'list' of int
        If a list of length 2, these are the integer numbers for the first and last hours included in analysis.  
        To include all hours, enter a list of length 2 with entries [0, 25]
        If blank, hours are restricted to 0800 to 1600 UTC.  
        If a list of length 1, this is the number of degrees of solar angle above horizon to start counting from (not currently implemented).
    clean_sigma : int
        Number of standard deviations of the outturn away from the mean for that month-hour to remove.
        Always remove zero outturn values.
        ** this could be done as part of the Power object
    goto_db : str
        Parameter to define where Power and Midas classes should fetch data from.  See class definitions.
    goto_file : str
        Parameter to define where WForecast class should fetch data from.  See class definition. 
    zero_irr_tolerance : float
        Limits amount of irradiance above which any zero measurements of outturn will be removed.
        Expressed as a fraction [0,1] of the maximum irradiance recorded for each site.
    verbose: int
        Verbosity to control the level of printed output.  
        0 = None
        1 = Errors
        2 = Significant information
        3 = All messages and logging information
    
    Attributes
    ----------
    features : DataFrame
        Dataframe indexed by datetime storing all the features to be used in the machine learning.
        The features are created in the method create_features, based on weather and user parameterization.
    target : Series
        Series indexed on datetime indicating the target to be used.
        Either a single outturn / load series, or the sum of a series of outturn / load series.
    target_capacity : float
        The capacity of the target timeseries. This is generated using the same methodology as the target
        series above.  It is used for the normalization of the statistics to percentage values.
    mods : Str
        String of the modifcations made to the data in pre-processing steps.  
        Starts empty at initialization and is populated by data prcessing steps to allow user to 
        see what runtime modifications have been implemented to the data.
    results_ : Series
        Results of the analysis, indexed by datetime.
    stats_ : DataFrame
        Statistics from cross-validation run calculated on each month-hour split.  
        The statistics on the total set (all month-hours) are in the first row.
    """

    zero_irr_tolerance = 0.005
    version = '0.1'

    def __init__(self, power_list, wh_list, power_data, wh_data, model_def, start_date, 
                 end_date=datetime.datetime.today(), forecast_days_ahead=0, power_sum_normalized = True, 
                 observation_freq='1H', solar_geometry=[], feature_list=[], lagged_variables={}, 
                 daylight_hours='', clean_sigma=5, goto_db='', goto_file='Cache', 
                 split_model=[], verbose=2, **kwargs):

        # assign attributes
        self.power_list = power_list        
        self.wh_list = wh_list
        self.power_data = power_data
        self.wh_data = wh_data
        self.model_def = model_def
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_days_ahead = forecast_days_ahead
        self.power_sum_normalized = power_sum_normalized
        self.observation_freq = observation_freq
        self.solar_geometry = solar_geometry
        self.feature_list = feature_list
        self.lagged_variables = lagged_variables
        self.daylight_hours = daylight_hours
        self.clean_sigma = clean_sigma
        self.goto_db = goto_db
        self.goto_file = goto_file
        self.verbose = verbose
        self.mods = ''
        # run methods to populate data:
        self.populate_wh()
        self.populate_power()
        self.power_data.resample()
        self.get_target()
        self.create_features()
        
    def cross_validate(self, capacity_metadata = False):
        """ Function runs the cross validation on the configuration of data provided.  
        Returns the summary metadata and statistics
        """
        timerstart = datetime.datetime.now()
        self.results_ = cross_validate_grouped(self.features, \
                                                 self.target.iloc[:,0], \
                                                 self.model_def).sort_index()
        self.runtime = datetime.datetime.now()-timerstart
        self.timestamp = datetime.datetime.now().replace(microsecond=0)
        self.myprint('Cross validation run completed in {} seconds'.format(self.runtime.seconds), 2)
        if capacity_metadata:
            cap = self.target_capacity * 1000
        else:
            cap = self.target.iloc[:,0].max()
        if self.results_.shape[0] > 0:
            self.stats_ = generate_error_stats(assign_month_hour(self.results_), cap, self.power_data.epex, self.power_data.sbsp, splits=False)
        else: 
            pass

    def populate_wh(self):
        """ Populates the weather data object with requested data."""
        if not isinstance(self.wh_data, (wforecast.WForecast, midas.Midas)):
            print('class neither WForecast nor Midas, but is of type:', type(self.wh_data))
        if isinstance(self.wh_data, (wforecast.WForecast)):
            self.wh_data.load_data(pd.DataFrame(self.wh_list, columns=['f_id']), self.start_date - timedelta(self.forecast_days_ahead+1)\
                , self.end_date - timedelta(self.forecast_days_ahead), goto_file=self.goto_file)
        else:
            print(self.wh_list, self.start_date - timedelta(1), self.end_date, self.goto_db)
            self.wh_data.load_data(self.wh_list, self.start_date - timedelta(1), self.end_date, goto_db=self.goto_db)

    def populate_power(self):
        """ Populates the power data object with requested data. """
        self.power_data.load_data(self.power_list, self.start_date, self.end_date, goto_db = self.goto_db)

    def get_target(self):
        """ Populates the target Series object by either: 
        If single power site, taking the values directly within date range.
        If multiple power sites, and power_sum_normalized is False, sums the figures
        across datetimes where all values exist.
        If multiple power sites, and power_sum_normalized is True, averages the figures, weighted
        by the capacity of the site.  This option ensure that all sites in the model run have equal impact.
        """
        df = self.power_data.get_obs(self.observation_freq).copy()
        if len(self.power_list) < 1:
            self.myprint("No sites in power list.", 1)
        if len(self.power_list) == 1:  # for a single power site, the target is just that series.
            self.target = df.xs(tuple(self.power_list), level='site_id', axis=0)
            self.target_capacity = self.power_data.metadata.loc[self.power_list[0], 'kWp']
            self.lat = self.power_data.metadata.loc[self.power_list[0], 'latitude']
            self.lon = self.power_data.metadata.loc[self.power_list[0], 'longitude']
        else:
            # straight sum of multiple power sites, excluding NaN's:
            if self.power_sum_normalized == False: 
                self.target = df.groupby('datetime').sum()
                self.target_capacity = self.power_data.metadata.loc[self.power_list, 'kWp'].sum()
            # normalized mean of power sites, so each site has same weight:
            if self.power_sum_normalized == True: 
                df.reset_index(inplace=True)
                capacity = self.power_data.metadata['kWp'].copy() # using capacity from metadata
                df['outturn'] = df.apply(lambda x: x.outturn * 1 / (capacity.loc[x.site_id]), axis=1)
                self.target = pd.DataFrame(df.groupby('datetime').mean()['outturn'])
                self.target_capacity = 1
            self.lat = self.power_data.metadata.loc[self.power_list, 'latitude'].mean()
            self.lon = self.power_data.metadata.loc[self.power_list, 'longitude'].mean()

    def create_features(self):
        """ Method where we create the features from weather, add additional features - E.g. lagged variables, 
        drop features not requested, and drop any datetimes without appropriate target values, feature values etc. 

        Notes
        -----
        Method must be run after get_target().
        """

        # copy weather data into features attributes, taking 5 extra days either side for used in lagged features
        # below overly complex line is what I currently have.  Feels like I should be able to do the line above, or 
        # something simpler.
        w_data = self.wh_data.get_obs(self.forecast_days_ahead)
        if len(w_data) == 0: 
            self.myprint("Error: No observations in the weather dataset passed in", 1)
        self.features = w_data.loc[self.wh_list, :].reset_index(level=0).tz_localize(None).\
            loc[(self.start_date-timedelta(5)).strftime('%Y%m%d'): (self.end_date+timedelta(5)).strftime('%Y%m%d')]\
                .set_index('site_id', append=True).swaplevel()
        # find columns to drop first, then add lagged features, then drop columns:
        features_to_drop = [ x for x in self.features.columns.values if x not in self.feature_list]
        self.__add_lagged_data()
        self.features = self.features.drop(features_to_drop, axis=1)
        # create wide format indexed by datetime only, so it can be indexed commonly to the target
        self.features = self.features.swaplevel().unstack() 
        # concetenate the multiindex column into single column index:
        self.features.columns = self.features.columns.map('{0[0]}_{0[1]}'.format)
        # add solar features and/or month & hour features as requested
        self.__add_solar_features()
        # restrict datetimes and all operations on axis 1:
        self.__limit_datetimes()
        # comments and graph: 
        self.myprint("{} features with {} datapoints: {}".format(self.features.shape[1], self.features.shape[0], self.features.columns.values), 3)
        self.mods += "¦ {} rows & {} features ".format(self.features.shape[0], self.features.shape[1])
        if self.verbose >= 3:
            self.__graph_features_and_target()

    def __add_solar_features(self):
        """ Method to add solar geometry related features.
        
        Notes
        -----
        Guided by the self.solar_geometry list parameter.  
        If contains "extra" then use extra-terrestrial irradiance as feature.
        If contains "month" then use month.
        If contains "hour" then use hour.
        If solar_geometry is empty then use month and hour.
        If contains "age" then use age of the system
        """

        if len(self.solar_geometry) == 0:
            self.features = assign_month_hour(self.features)
        if len(self.features) == 0:
            self.myprint("No rows in features dataframe.", 1)
        if 'month' in self.solar_geometry:
            self.features = self.features.assign(month=self.features.index.shift(-1, freq='h').month)
        if 'dayofyear' in self.solar_geometry: # this outperforms the month feature often and can replace it 
            self.features = self.features.assign(dayofyear=self.features.index.dayofyear) # found with load profiles I had to add 10 days
            self.features['dayofyear_sin'] = self.features.dayofyear.apply(lambda x: np.sin(np.pi*2* (x)/ 365))
            self.features['dayofyear_cos'] = self.features.dayofyear.apply(lambda x: np.cos(np.pi*2* (x)/ 365))
            self.features.drop('dayofyear', axis=1, inplace=True)
        if 'hour' in self.solar_geometry:
            self.features = self.features.assign(hour=self.features.index.hour)
        if 'hour_circular' in self.solar_geometry:
            self.features = self.features.assign(hour=self.features.index.hour)            
            self.features['hour_sin'] = self.features.hour.apply(lambda x: np.sin(np.pi*2* (x)/ 365))
            self.features['hour_cos'] = self.features.hour.apply(lambda x: np.cos(np.pi*2* (x)/ 365))
            self.features.drop('hour', axis=1)
        if 'weekday_grouped' in self.solar_geometry:
            self.features = apply_weekday(self.features, 'grouped')
        if 'weekday_individual' in self.solar_geometry:
            self.features = apply_weekday(self.features, 'individual')
        if 'weekday_holiday_individual' in self.solar_geometry:
            self.features = apply_weekday(self.features, 'week_holiday_individual')
        if 'holiday_individual' in self.solar_geometry:
            self.features = apply_weekday(self.features, 'holiday_individual')
        if 'extra' in self.solar_geometry:
            self.features['e_irr'] = get_extra_radiation(self.features.index) * \
                np.sin(np.radians(get_solarposition(self.features.index, self.lat, self.lon).apparent_elevation))\
                    .rolling(window=2).mean()
        if 'clearsky' in self.solar_geometry:
            irr_cols = [col for col in self.features.columns if 'irr' in col]
            self.features['clear_s'] = self.features[irr_cols[0]] / get_extra_radiation(self.features.index) * \
                np.sin(np.radians(get_solarposition(self.features.index, self.lat, self.lon).apparent_elevation))\
                    .rolling(window=2).mean()
        if 'age' in self.solar_geometry: # calculate the age of the system from an arbitrary date 
            self.features = self.features.assign(age=[x.days for x in self.features.index.date - datetime.date(2014, 1, 1)]) 

    def __limit_datetimes(self):
        """ Method which removes the datetimes not requested or with missing data, and
        ensures the target and feature set has common index.  
        Removes rows with hours outside of daytime, either by a simple hour search, or 
        using solar geometry.

        Notes
        -----
        Requires self.features to be in wide format, so the rows are indexed only on datetime.
        """

        # restrict to dates requested and drop NaN's
        self.features = self.features.tz_localize(None).loc[self.start_date.strftime('%Y%m%d'): (self.end_date-timedelta(1)).strftime('%Y%m%d')].dropna(how='any')
        self.target = self.target.tz_localize(None).loc[self.start_date.strftime('%Y%m%d'): (self.end_date-timedelta(1)).strftime('%Y%m%d')].dropna(how='any')
        no_feat = len(self.features)
        no_target = len(self.target)
        # restrict index to common entries of both features and target set by merging on inner.
        # perform following actions on the merged dataset.
        self.features = pd.merge(self.features, self.target, 'inner', left_index=True, right_index=True)
        #self.features = self.features.loc[self.target.index]
        #self.target = self.target.loc[self.features.index]
        no_joined = len(self.features)
        self.myprint('{} weather observations, {} outturn observations, joined for {} observations ({} days and {} hours).'.\
             format(no_feat, no_target, no_joined, \
             np.trunc(no_joined/24), no_joined-24*np.trunc(no_joined/24)), 3)
        # restrict hours to daylight - those requested in analysis
        if self.daylight_hours=='':
            low_hour = 9
            high_hour = 17
            self.features = self.features[self.features.index.hour.isin(range(low_hour, high_hour)) ]
        elif len(self.daylight_hours)==2:
            low_hour, high_hour = self.daylight_hours[0], self.daylight_hours[1]
            self.features = self.features[self.features.index.hour.isin(range(low_hour, high_hour)) ]
        elif len(self.daylight_hours)==1:  # apply mask to filter only hours with sun elevation above threshold
            self.features = self.features\
                [(get_solarposition(self.features.index.shift(-30, freq='min'), self.lat, self.lon).\
                apparent_elevation>self.daylight_hours[0]).values]
        else: 
            self.myprint('Daylight hours attribute is not valid.', 1)
        self.__remove_zero_outturn(.005)
        if self.features['outturn'].sum()>0:
            self.__remove_large_outturn(1.5)
        self.target = pd.DataFrame(self.features['outturn'].sort_index())
        self.features = self.features.drop('outturn', axis=1).sort_index()
        self.myprint("Joined observations: {}. Restricted datetimes: {}".format(no_joined, len(self.features)), 3)

    def __graph_features_and_target(self):
        #graph number of observations per day and number of zeros in dataset
        """         daily_outturn_data = self.target.groupby(pd.Grouper(freq='D')).count()
        daily_weather_data = self.features.groupby(pd.Grouper(freq='D')).count()
        daily_zero_outturn_data = self.target[self.target.outturn==0].groupby(pd.Grouper(freq='D')).count()
        plt.figure(figsize=(16,1))
        plt.plot(daily_weather_data, color='y')
        plt.plot(daily_outturn_data, color='b')
        plt.plot(daily_zero_outturn_data, color='r') """
        pass

    def __remove_zero_outturn(self, fraction=0.005):
        # removes the records from the dataset where the outturn is zero but the irradiance is non-zero
        # keep records with irradiance less than a small % of maximum irradiance (.5% default)
        fraction = self.zero_irr_tolerance
        rows = self.features.shape[0]
        # sum up the irradiance from all weather sites and find on each datetime if the irradiance is greater
        # than the tolerance (default 0.5%) of the maximum irradiance across all datetimes.
        # Remove these rows.  
        # Note the behaviour is slightly problematic when using lagged irradiance, as different runs with different
        # lags remove different datetimes (as lagged irr is added in    ).
        self.features['irr'] = self.features.loc[:,[col for col in self.features.columns if 'irr' in col]].sum(axis=1)
        self.features.drop(self.features[(self.features.outturn==0) & (self.features.irr>self.features.irr.max()*fraction)].index, inplace=True)
        self.features = self.features.drop('irr', axis=1)
        removed = rows - self.features.shape[0]
        self.myprint("Removed {} zeros ({}%)".format(removed, round(100*removed/rows,2)), 3)

    def __remove_large_outturn(self, percent=1.5):
        # removes the largest extreme values
        # removes the records from the dataset where the outturn is greater than the percent times
        # the average of the 10th to 20th largest values.  
        rows = self.features.shape[0]
        too_large = self.features.outturn.nlargest(20)[-10:].mean() * percent
        self.features.drop(self.features[self.features.outturn > too_large].index, inplace=True)
        removed = rows = self.features.shape[0]
        self.myprint("Removed {} very large outturn values ({}%)".format(removed, round(100*removed/rows,2)), 3)        

    def __add_lagged_data(self):
        """ Method to create lagged and rolling average features on weather data.
                
        Notes
        -----
        Dictionary lagged_variables contains first entry: 'lags', which contains a 
        dictionary for the lagged hours for each weather variable string.
        The sign convention is that a lag of 1 means the weather reading from the hour preceding the
        hour in question is added as a feature (E.g. to account for irr being high/ low the
        hour before the heating the cell.)
        The second entry contains 'rolling', which contains an entry for each rolling average number
        of hours to be used.  The lagged_variables parameters is a dictionary like: 
        lagged_data = { 'lags' : {'irr' : [2, 1, -1], 
                                  'air_temp' : [1 ] }, 
                        'rolling' : { 'irr' : [24]} }
        The rolling average does not include the current datetime in the rolling average.
        The created features are labelled: feat_n for a lagged feature by n, and 
        featrn for a rolling average feature of n.
        """

        for feature in self.features.columns.values:
            try:
                for i in self.lagged_variables['lags'][feature]:
                    name = feature + '_' + str(i)
                    self.features[name] = self.features.groupby(level=0)[feature].shift(i)
            except KeyError:
                pass
            try: 
                for i in self.lagged_variables['rolling'][feature]:
                    name = feature + 'r' + str(i)
                    self.features[name] = self.features.groupby(level=0)[feature].apply(lambda x: x.rolling(window=i).mean().shift(1))
            except KeyError:
                pass

    def __get_metadata(self):
        """Function returns the summary of the cross validation run including metadata and statistics.
        """
        dict = { 'model_params' : self.model_def.get_parameters(), \
            'ss_ids' : self.power_list, \
            'src_ids' : self.wh_list, \
            'timestamp' : self.timestamp, \
            'lagged_irr' : self.lagged_variables['lags']['irr'], \
            'lagged_temp' : self.lagged_variables['lags']['air_temp'], \
            'rolling_irr' : self.lagged_variables['rolling']['irr'], \
            'rolling_temp' : self.lagged_variables['rolling']['air_temp']}
        return(dict)

    def myprint(self, message, message_priority):
        """ Method to print messages filtered by verbosity.
        The verbosity setting is a class attribute and should be in range [0:3]

        Parameters
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

def assign_month_hour(df):
    """ Function to assign month and hour columns to dataframe based on df containing datetime index
    """
    
    df = df.assign(hour=df.index.hour)
    df = df.assign(month=df.index.shift(-1, freq='h').month)
    return(df)