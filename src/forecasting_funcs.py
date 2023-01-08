import datetime
import pytz
import pandas as pd
import model_run as mr
import power as pw
import midas as md
import wforecast as wf

def x_val_results(ss_id, w_id, power, weather, model, start, end, forecast_days_ahead, \
          lags, solar, goto_db='Never', goto_file='File', verbose=2):
    """
    Function to return the key statistics of the cross-validation.
    Runs on a single ss_id taking either weather forecast data or actual observed weather data as input.
    Initially tested on just ECMWF weather forecasts.
    
    Parameters
    ----------
    ss_id : int
        Site id of PV observations.  Must be present in the database or cache so a Power object can
        be instantiated.
    w_id : str
        Identifier of the weather forecast location in lat:lon format.  
    power : obj:Power
        Instance of Power class
    weather : obj:Weather
        Instance of the Weather class
    model : obj:Model
        Instance of the Model class
    start : datetime
    end : datetime
    forecast_days_ahead : int
        Number of days ahead to run the cross validation on: 0 indicates the forecast for current day
        (the function assumes the forecast is from overnight), 1 is day ahead etc. 
        -1 indicates to use the actual weather observations (from Midas station).
    lags : dict
        Lags used as input to ModelRun
    solar : list
        The solar geometry fields to use in ModelRun
    goto_db : str
        Wheather to goto db for the Power and Midas observations (input to these classes).  
        Takes values {'Never', 'Always', ''}
    goto_file : str
        Whether to goto file or look in the cache for the weather forecast data.
        Takes values "File" or "Cache".  If Cache, the values are retrieved from the cache, 
        overwriting all other data in memory. 
    verbose : int
        3 is very verbose output, 0 is only errors.
    
    Returns
    -------
    DataFrame with one row with the statistics from the Model Run as columns.
    """
    run= mr.ModelRun([ss_id], 
                   [w_id], 
                   power, weather,
                   model, 
                   start, end, 
                   forecast_days_ahead=forecast_days_ahead, 
                    lagged_variables=lags,
                      solar_geometry=solar,
                      goto_db=goto_db, 
                      goto_file=goto_file,
                     verbose=verbose)
    run.cross_validate(False)
    temp = run.stats_.iloc[0:1].copy()
    temp['model'] = model.ml_model
    temp['lags'] = str(lags['lags']['irr'])
    temp['w_id'] = w_id
    temp['ss_id'] = ss_id
    temp['days_ahead'] = forecast_days_ahead
    return(temp)

def x_val_results_plus(ss_id, w_id, power, weather, model, start, end, forecast_days_ahead, \
          lags, solar, feature_list=[], daylight_hours=[10], goto_db='Never', goto_file='File', verbose=2):
    """
    Function to return the key statistics of the cross-validation AND the results at the timestep level.
    Runs on a single ss_id taking either weather forecast data or actual observed weather data as input.
    Initially tested on just ECMWF weather forecasts.
    
    Parameters
    ----------
    ss_id : int
        Site id of PV observations.  Must be present in the database or cache so a Power object can
        be instantiated.
    w_id : str
        Identifier of the weather forecast location in lat:lon format.  
    power : obj:Power
        Instance of Power class
    weather : obj:Weather
        Instance of the Weather class - midas or w_forecat type.
    model : obj:Model
        Instance of the Model class
    start : datetime
    end : datetime
    forecast_days_ahead : int
        Number of days ahead to run the cross validation on: 0 indicates the forecast for current day
        (the function assumes the forecast is from overnight), 1 is day ahead etc. 
        -1 indicates to use the actual weather observations (from Midas station).
    lags : dict
        Lags used as input to ModelRun
    solar : list
        The solar geometry fields to use in ModelRun, which includes 'dotw' feature for load modelling.
    feature_list : list
        List fieatures to include.
    daylight_hours : obj:'list' of int
        If a list of length 2, these are the integer numbers for the first and last hours included in analysis.  
        To include all hours, enter a list of length 2 with entries [0, 25]
        If blank, hours are restricted to 0800 to 1600 UTC.  
        If a list of length 1, this is the number of degrees of solar angle above horizon to start counting from.
    goto_db : str
        Wheather to goto db for the Power and Midas observations (input to these classes).  
        Takes values {'Never', 'Always', ''}
    goto_file : str
        Whether to goto file or look in the cache for the weather forecast data.
        Takes values "File" or "Cache".  If Cache, the values are retrieved from the cache, 
        overwriting all other data in memory. 
    verbose : int
        3 is very verbose output, 0 is only errors.
    
    Returns
    -------
    DataFrame with one row with the statistics from the Model Run as columns.
    DataFrame with the (hourly) results. 
    """
    try:
        location_slice = power.obs.loc[ss_id, :]
        no_rows = location_slice[ pytz.utc.localize(start) : pytz.utc.localize(end) ].shape[0]
    except:
        no_rows = 0
    if no_rows > 0:
        run= mr.ModelRun([ss_id],
                        [w_id],
                        power, weather,
                        model,
                        start, end,
                        forecast_days_ahead=forecast_days_ahead, 
                        lagged_variables=lags,
                        daylight_hours=daylight_hours,
                        solar_geometry=solar,
                        feature_list=feature_list,
                        goto_db=goto_db, 
                        goto_file=goto_file,
                        verbose=verbose)
        run.cross_validate(False)
        timestamp = datetime.datetime.now().replace(microsecond=0)
        temp = run.stats_.iloc[0:1].copy()
        temp['w_id'] = w_id
        temp['ss_id'] = ss_id
        """temp['model'] = model.ml_model
        for entry in lags:
            for feature in lags[entry]: 
                temp['lags'] = entry[0:1]+'-'+feature[0:3]+'-'+str(lags[entry][feature])
        grpd = ''
        for word in model.grouped_by:
            grpd += word[0:1]
        temp['grouped'] = grpd"""
        temp['days_ahead'] = forecast_days_ahead
        temp['run'] = timestamp
        temp = temp[['ss_id', 'count', 'run', 'MBE', 'MAE', 'RMSE', 'cash_pct', 'hrly_val', 'lg_over', 'lg_under', 'cap_used', 'w_id', 'days_ahead']]
        res = pd.concat([run.results_], keys=[timestamp], names=['run'])[['forecast', 'outturn']]
        res.loc[:, 'ss_id'] = ss_id
        res = res.set_index('ss_id', append=True).reorder_levels([0, 2, 1], axis=0)
    else:
        temp = pd.DataFrame({'ss_id' : ss_id, 'count' : 0, 'run' : datetime.datetime.now().replace(microsecond=0)}, index=[0])
        res = pd.DataFrame([])
    return(temp, res)