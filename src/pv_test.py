import pandas as pd
import sitedata
import power as pw
import midas as md 
import datetime
import path
import os
from model_definition import ModelDefinition
import model_run as mr

#pw = power.Power(1, local_config=local)

#pw.load_data([5902], start_date=datetime.date(2016, 1, 1), end_date=datetime.date(2018,1,1))
#pw.save_to_hdf()
""" start = datetime.date(2016,1,1)
end = datetime.date(2018, 1, 1)
kwargs_empty = {}
lr = ModelDefinition('linear_r', ['month', 'hour'], 10, **kwargs_empty, text='Mth-Hr split')
rf = ModelDefinition('random_f', [], 10, **kwargs_empty)
kwargs_grad = {"n_estimators": 100, "learning_rate" : 0.1, "max_depth" : 5, "random_state" : 0, "loss" : "ls"}
grad = ModelDefinition('g_boost', [], 10, **kwargs_grad)
lag = {}
lag[1] = { 'lags' : {'irr' : [1] }}
lag[22] = { 'lags' : {'irr' : [-1] }, 'rolling' : { 'irr' : [2]} }
lag[3] = { 'lags' : {'irr' : [1] }}
lag[23] = { 'lags' : {'irr' : [-1] }, 'rolling' : { 'irr' : [2]} }
lag[4] = { 'lags' : {'irr' : [1] }}
lag[42] = { 'lags' : {'irr' : [-1] }, 'rolling' : { 'irr' : [2]} }

weather = md.Midas(3) """
power = pw.Power(3)
#weather.load_data([842], start, end, goto_db='')
""" for row in lag:
    for model in[lr, grad]: 
        run1= mr.ModelRun([4784], 
                    [842], 
                    power, weather,
                    model, 
                    start, end, 
                    forecast_hours_ahead=0, 
                    sigma_clean=5, 
                    goto_db='Never',
                    verbose=3)
        try: 
            print('rolling = {}'.format(str(lag[row]['rolling']['irr'])))
        except KeyError:
            print('no rolling')        
        run1.cross_validate() """