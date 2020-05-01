# Script to test doing basic calculation on list of ss_ids to ensure can do parameterization

import datetime
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import configparser

import midas as md
import power as pw
import wforecast as wf
from forecasting_funcs import x_val_results, x_val_results_plus
from location_data_utils import (cross_locations, get_fcst_locs,
                                 get_longest_ss_ids, read_netcdf_file)
from model_definition import ModelDefinition

def main():
    """
    Arguments
    ---------
    arg1 : name of the input.list file - should be a list with ss_ids to query.
    arg2 : name of output file to write results to.
    arg3 : the forecast_day_ahead: two binary digits, each representing (by position) 
            if day 0 or day 1 details are output to file E.g. 01 is output just day 1 ahead details
    arg4 : "wsl", if it is running in wsl, so it picks up correct config files.  Or missing.
    """ 

    # input ss_ids.list and output files
    if len(sys.argv) > 2:
        ss_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        raise(Exception)
    if len(sys.argv) > 3:
        days_ahead = sys.argv[3] + '0000'

    wsl = "" if len(sys.argv) <= 4 else sys.argv[4]

    # netcdf_file locations:
    config = configparser.ConfigParser()
    if wsl=='wsl':
        config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config/wforecast_lx.ini'))
    else:
        config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config/wforecast.ini'))
    netcdf_path = config['query_settings']['path']
    location_ref_filename = config['query_settings']['location_ref_filename']
    location_ref = os.path.join(netcdf_path, location_ref_filename)

    # start / end dates
    s = datetime.datetime(2016, 1, 1).date()
    e = datetime.datetime(2016, 1, 10).date()
    goto_db = ''

    # ml model setup
    kwargs_empty = {}
    lr = ModelDefinition('linear_r', ['month', 'hour'], 5, **kwargs_empty, text='Mth-Hr split')
    kwargs_grad = {"n_estimators": 100, "learning_rate" : 0.1, "max_depth" : 5, "random_state" : 0, "loss" : "ls"}
    grad = ModelDefinition('g_boost', [], 5, 'dayofyear', **kwargs_grad)
    kwargs_xgrad = {"objective" : "reg:squarederror", "eval_metric" : "mae", "learning_rate" : .06, "max_depth" : 6, 
                "colsample_bytree" : .9, "min_split_loss" : 0.0, "colsample_bylevel" : 1, "lambda" : 0.5, 
               "min_child_weight" : 4, "n_estimators" : 130, "gamma" : 0, "subsample" : 0.9, "reg_alpha" : 100}
    xgrad = ModelDefinition('xg_boost', [], 5, 'dayofyear', 0, True, '', **kwargs_xgrad)
    
    with open(ss_file, 'r') as f:
        ss_list = f.readlines()
    ss_list = list(map(lambda x: int(x.strip()), ss_list))
    power = pw.Power(1)
    w_forecast = wf.WForecast(1)
    power.load_metadata(ss_list)
    power.load_data(ss_list, s, e, goto_db=goto_db)

    # find forecast locations, load weather forecast data and assocate with each ss_id
    locations = get_fcst_locs(power.metadata, filename=location_ref, n=1)
    w_forecast.load_data(locations, netcdf_path, s, e, goto_file="File")
    fpairings = pd.DataFrame(ss_list, columns=['ss_id']).merge(locations.astype({'site_id':np.int64}), left_on='ss_id', right_on='site_id', how='inner').drop('site_id', axis=1)

    tstats = pd.DataFrame([])
    tresults0 = pd.DataFrame([])
    tresults1 = pd.DataFrame([])
    # run forecast runs 
    # can modify ML model used in here:
    for index, row in fpairings[:].iterrows(): 
        ss_id = int(row['ss_id'])
        f_id = row['f_id']
        print(ss_id, f_id)
        for model in [xgrad]: 
            lags = { 'lags' : {'irr' : []}} if model==lr else { 'lags' : {'irr' : [1, 2, 3, -1, -2]}}
            for i in range(0, 4): # loop for 4 days ahead.
                print('forecast:', f_id, ss_id, model.ml_model, i)
                temp_stats, temp_results = x_val_results_plus(ss_id, f_id, power, w_forecast, model, s, e, 
                                                                i, \
                                                                lags, \
                                                                ['dayofyear', 'hour'], 
                                                                ['irr', 'u', 'v', 'temp'], # these need to be modified if using Midas vs ECMWF weather
                                                                [10], 
                                                                goto_db, 'None', 2)
                tstats = tstats.append(temp_stats)
                if days_ahead[i]=='1':
                    print(i, days_ahead[i])
                    if i==0:
                        tresults0 = tresults0.append(temp_results)
                    if i==1:
                        tresults1 = tresults1.append(temp_results)
    tstats.to_csv(output_file)
    tresults0.to_csv(output_file.split('.')[0]+'_res0.csv')
    tresults1.to_csv(output_file.split('.')[0]+'_res1.csv')
    print('Finished')

if __name__ == "__main__":
	main()