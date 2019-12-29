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
start = datetime.date(2016,1,1)
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
lag[16] = { 'lags' : {'irr' : [1, 2, 3, 4, -1, -2, -3]}}

weather = md.Midas(3) 
power = pw.Power(3)
pairings = pd.read_csv('C:\\Users\\Dan Travers\\Google Drive\\Projects\\Explore\\PVLive\\src_ss_id_pairs.csv')
for row in pairings[0:1].iterrows(): 
    src_id = row[1].iloc[0]
    ss_id = row[1].iloc[1]
    for k in [2]: 
        grad = ModelDefinition('g_boost', [], k, **kwargs_grad)
        print(src_id, ss_id, model.ml_model)
        run1= mr.ModelRun([ss_id], 
                       [src_id], 
                       power, weather,
                       model, 
                       start, end, 
                       forecast_hours_ahead=0, 
                        lagged_variables=lag[16],
                       sigma_clean=5, 
                          goto_db='Never', 
                         verbose=3)
        run1.cross_validate(False)
        temp = run1.stats_.iloc[0:1].copy()
        temp['model'] = model.ml_model
        #temp['lags'] = str(lag[16]['lags']['irr'])
        #try: 
        #    temp['rolling'] = str(lag[row]['rolling']['irr'])
        #except KeyError:
        #    temp['rolling'] = 'n_roll'
        temp['src_id'] = src_id
        temp['ss_id'] = ss_id
        temp['vers'] = 2
        temp['k-fold'] = k
        tstats = tstats.append(temp)