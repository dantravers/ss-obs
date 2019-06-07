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
    run1= mr.ModelRun([ss_id], 
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
    run1.cross_validate(False)
    temp = run1.stats_.iloc[0:1].copy()
    temp['model'] = model.ml_model
    temp['lags'] = str(lags['lags']['irr'])
    temp['w_id'] = w_id
    temp['ss_id'] = ss_id
    temp['days_ahead'] = forecast_days_ahead
    return(temp)