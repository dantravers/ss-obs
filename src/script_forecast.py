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
from forecasting_funcs import x_val_results
from location_data_utils import (cross_locations, get_fcst_locs,
                                 get_longest_ss_ids, read_netcdf_file)
from model_definition import ModelDefinition


def main():
    """ Function to output the average yield from list of ss_ids over a period of time.
    Arguments
    ---------
    arg1 : name of the input.list file - should be a list with ss_ids to query.
    art2 : name of output file to write results to.
    """

    # input ss_ids.list and output files
    if len(sys.argv) > 2:
        ss_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        raise(Exception)
    # netcdf_file locations:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config/wforecast.ini'))
    netcdf_path = config['query_settings']['path']
    location_ref_filename = config['query_settings']['location_ref_filename']
    if os.name == 'nt':
        location_ref_filename = location_ref_filename.replace(':', '%3A')
    location_ref = os.path.join(netcdf_path, location_ref_filename)
    # start / end dates
    s = datetime.datetime(2016, 1, 1).date()
    e = datetime.datetime(2018,1, 1).date()
    # ml model setup
    kwargs_empty = {}
    lr = ModelDefinition('linear_r', ['month', 'hour'], 5, **kwargs_empty, text='Mth-Hr split')
    kwargs_grad = {"n_estimators": 100, "learning_rate" : 0.1, "max_depth" : 5, "random_state" : 0, "loss" : "ls"}
    grad = ModelDefinition('g_boost', [], 5, **kwargs_grad)
    
    with open(ss_file, 'r') as f:
        ss_list = f.readlines()
    ss_list = list(map(lambda x: int(x.strip()), ss_list))

    power = pw.Power(1)
    w_forecast = wf.WForecast(2)

    # find forecast locations, load weather forecast data and assocate with each ss_id
    power.load_metadata(ss_list)
    locations = get_fcst_locs(power.metadata, filename=location_ref, n=1)
    w_forecast.load_data(locations, netcdf_path, s, e, goto_file="File")
    fpairings = pd.DataFrame(ss_list, columns=['ss_id']).merge(locations.astype({'site_id':np.int64}), left_on='ss_id', right_on='site_id', how='inner').drop('site_id', axis=1)

    tstats = pd.DataFrame([])
    # run forecast runs 
    # can modify ML model used in here:
    for index, row in fpairings[:].iterrows(): 
        ss_id = int(row['ss_id'])
        f_id = row['f_id']
        print(ss_id, f_id)
        for model in [ grad]: 
            lags = { 'lags' : {'irr' : []}} if model==lr else { 'lags' : {'irr' : [1, 2, 3, -1, -2]}}
            for i in range(0, 4): # loop for 4 days ahead.
                print('forecast:', f_id, ss_id, model.ml_model, i)
                tstats = tstats.append(x_val_results(ss_id, f_id, power, w_forecast, model, s, e, i, \
                                                    lags, \
                                                    ['hour', 'month', 'extra'], 'Never', 'None', 2))
    tstats.to_csv(output_file)
    print('Finished')

if __name__ == "__main__":
	main()