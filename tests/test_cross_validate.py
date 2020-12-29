# Test to test the ML running & cross-validating from a set of weather and PV data
# Tests a single site only, with basic cleaning of large results and zeros, 
# but no lagged data or special features.

import os
import numpy as np
import pandas as pd
import datetime 
from datetime import timedelta
import importlib
import model_run as mr
import power as pw
import midas as md
from pv_ml_models import cross_validate_grouped
from model_definition import ModelDefinition

# directory for all test data (hdfs & bench_*.csv)
if os.name == 'nt':
    data_dir = 'C:/Users/DanTravers/Documents/GitHub/ss-obs/tests/test_data'
else:
    data_dir = '/mnt/c/Users/dantravers/Documents/GitHub/ss-obs/tests/test_data'
# setup model definitions: 
kwargs_empty = {}
lr = ModelDefinition('linear_r', ['month', 'hour'], 10, '', **kwargs_empty, text='Mth-Hr split')
kwargs_grad = {"n_estimators": 100, "learning_rate" : 0.1, "max_depth" : 5, "random_state" : 0, "loss" : "ls"}
grad = ModelDefinition('g_boost', [], 10, '', **kwargs_grad)
kwargs_xgrad = {"objective" : "reg:squarederror", "eval_metric" : "mae", "learning_rate" : .09, "max_depth" : 5, 
                "colsample_bytree" : .9, "min_split_loss" : 0.0, "colsample_bylevel" : 1, "lambda" : 0.5}
xgrad = ModelDefinition('xg_boost', [], 5, 'week', 0, True, '', **kwargs_xgrad)
xgrad2 = ModelDefinition('xg_boost', [], 2, 'dayofyear', 0, True, '', **kwargs_xgrad)
ml_model_list = [lr, grad, xgrad, xgrad2]
# setup dates and weather / PV data
start = datetime.date(2016,1,1)
end = datetime.date(2018, 1, 1)
weather = md.Midas(2)
weather.config['hdf5']['store_name'] = os.path.join(data_dir, "midas_test_data.h5")
weather.config['hdf5']['store_path'] = ''
power = pw.Power(2)
power.config['hdf5']['store_name'] = os.path.join(data_dir, "power_test_data.h5")
power.config['hdf5']['store_path'] = ''

def test_cross_validate():
    tstats = pd.DataFrame([])
    # running cross validation through 3 models.
    for model in ml_model_list: 
        run1= mr.ModelRun([4784], 
                      [676], 
                      power, weather,
                      model, 
                      start, end, 
                      forecast_hours_ahead=0, 
                      feature_list=['irr', 'wind_dir', 'wind_speed', 'air_temp', 'rltv_hum'],
                      sigma_clean=5, 
                        verbose=2)
        run1.cross_validate(False)
        # test the results are equal for 2 ML models
        file_name = os.path.join(data_dir, "bench_results_" + model.ml_model + model.cross_val_grp + ".csv" )
        #to replace with new stats file: run1.results_.to_csv(file_name)
        bench_results = pd.read_csv(file_name, index_col=0, parse_dates=True)
        print("Testing model results: {}, {}".format(model.ml_model, model.cross_val_grp))
        run1.results_ = run1.results_.astype( {"forecast": np.float64} )
        pd.testing.assert_frame_equal(run1.results_.iloc[:, [0, 1]], bench_results.iloc[:, [0, 1]])
        # collect stats
        temp = run1.stats_.iloc[0:1].copy()
        temp['model'] = model.ml_model
        tstats = tstats.append(temp)

    # compare stats are equal:
    file_name = os.path.join(data_dir, "bench_stats1.csv" )
    bench_stats = pd.read_csv(file_name, index_col=[0])
    print("Testing stats:")
    print(tstats)
    print(bench_stats)
    #tstats.to_csv('tstats_test.csv')
    #bench_stats.to_csv('bench_stats_test.csv')
    pd.testing.assert_frame_equal(tstats, bench_stats)

def test_add_lagged_features():
    lag1 = { 'lags' : {'irr' : [1, -1]}}
    # initialize ModelRun, which creates features
    run1= mr.ModelRun([4784], 
                  [676], 
                  power, weather,
                  lr, 
                  start, end, 
                  forecast_hours_ahead=0, 
                  feature_list=['irr', 'wind_dir', 'wind_speed', 'air_temp', 'rltv_hum'],
                  lagged_variables=lag1, 
                  sigma_clean=5, 
                    verbose=2)
    # test the results are equal by looking at feature df:
    file_name = os.path.join(data_dir, "bench_lagged_features.csv" )
    bench_results = pd.read_csv(file_name, index_col=0, parse_dates=True)
    print("Testing lagged results:")
    pd.testing.assert_frame_equal(run1.features, bench_results)

def test_add_rolling_features():
    rolling = { 'lags' : {'irr' : [-1, -2] }, 
                'rolling' : { 'irr' : [3]} }
    file_name = os.path.join(data_dir, "bench_rolling_features.csv" )
    bench_results = pd.read_csv(file_name, index_col=0, parse_dates=True)
    # initialize ModelRun, which creates features
    run1= mr.ModelRun([4784], 
                  [676], 
                  power, weather,
                  lr, 
                  start, end, 
                  forecast_hours_ahead=0, 
                  feature_list=['irr', 'wind_dir', 'wind_speed', 'air_temp', 'rltv_hum'],
                  lagged_variables=rolling, 
                  sigma_clean=5, 
                    verbose=2)
    # test the results are equal by looking at feature df:
    print("Testing rolling results:")
    pd.testing.assert_frame_equal(run1.features, bench_results)

def test_remove_features():
    lag = { 'lags' : {'irr' : [1, -1]}}
    feat = ['irr', 'air_temp', 'rltv_hum']
    file_name = os.path.join(data_dir, "bench_remove_features.csv" )
    bench_results = pd.read_csv(file_name, index_col=0, parse_dates=True)
    # initialize ModelRun, which creates features
    run1= mr.ModelRun([4784], 
                  [676], 
                  power, weather,
                  lr, 
                  start, end, 
                  forecast_hours_ahead=0, 
                  lagged_variables=lag,
                  feature_list = feat,  
                  sigma_clean=5, 
                    verbose=2)
    # test the results are equal by looking at feature df:
    print("Testing remove features:")
    pd.testing.assert_frame_equal(run1.features, bench_results)

def test_remove_features_parameter_empty_list():
    file_name = os.path.join(data_dir, "bench_no_features_removed.csv" )
    bench_results = pd.read_csv(file_name, index_col=0, parse_dates=True)
    # initialize ModelRun, which creates features
    run1= mr.ModelRun([4784], 
                  [676], 
                  power, weather,
                  lr, 
                  start, end, 
                  forecast_hours_ahead=0, 
                  feature_list=['irr', 'wind_dir', 'wind_speed', 'air_temp', 'rltv_hum'],
                  sigma_clean=5, 
                  verbose=2)
    # test the results are equal by looking at feature df:
    print("Testing rolling results:")
    pd.testing.assert_frame_equal(run1.features, bench_results)

def test_extra_terrestrial_irradiance():
    file_name = os.path.join(data_dir, "bench_solar_feat_extra_terrestrial_irr.csv" )
    bench_results = pd.read_csv(file_name, index_col=0, parse_dates=True)
    # initialize ModelRun, which creates features
    run1= mr.ModelRun([4784], 
                  [676], 
                  power, weather,
                  lr, 
                  start, end, 
                  forecast_hours_ahead=0, 
                  solar_geometry=['month', 'extra'],
                  feature_list=['irr', 'wind_dir', 'wind_speed', 'air_temp', 'rltv_hum'],
                  sigma_clean=5, 
                  verbose=2)
    # test the results are equal by looking at feature df:
    print("Testing extra-terrestrial irradiation feature:")
    pd.testing.assert_frame_equal(run1.features, bench_results) 