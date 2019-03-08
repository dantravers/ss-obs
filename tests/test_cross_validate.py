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
from pv_ml_models import cross_validate_grouped, coef_lr_grouped
from model_definition import ModelDefinition

def test_cross_validate():
    data_dir = 'C:/Users/Dan Travers/Documents/GitHub/ss-obs/tests'
    # setup model definitions: 
    kwargs_empty = {}
    lr = ModelDefinition('linear_r', ['month', 'hour'], 10, **kwargs_empty, text='Mth-Hr split')
    kwargs_grad = {"n_estimators": 100, "learning_rate" : 0.1, "max_depth" : 5, "random_state" : 0, "loss" : "ls"}
    grad = ModelDefinition('g_boost', [], 10, **kwargs_grad)
    ml_model_list = [lr, grad]
    # setup dates and weather / PV data
    start = datetime.date(2016,1,1)
    end = datetime.date(2018, 1, 1)
    weather = md.Midas(2)
    weather.config['hdf5']['store_name'] = os.path.join(data_dir, "midas_test_data.h5")
    power = pw.Power(2)
    power.config['hdf5']['store_name'] = os.path.join(data_dir, "power_test_data.h5")
    tstats = pd.DataFrame([])
    # running cross validation through 3 models.
    for model in ml_model_list: 
        run1= mr.ModelRun([4784], 
                      [676], 
                      power, weather,
                      model, 
                      start, end, 
                      forecast_hours_ahead=0, 
                      sigma_clean=5, 
                        verbose=2)
        run1.cross_validate(False)
        # test the results are equal for 3 ML models
        file_name = os.path.join(data_dir, "bench_results_" + model.ml_model + ".csv" )
        bench_results = pd.read_csv(file_name, index_col=0, parse_dates=True)
        print("Testing model results: {}".format(model.ml_model))
        pd.testing.assert_frame_equal(run1.results_.iloc[:, [2, 3]], bench_results.iloc[:, [2, 3]])
        # collect stats
        temp = run1.stats_.iloc[0:1].copy()
        temp['model'] = model.ml_model
        tstats = tstats.append(temp)

    # compare stats are equal:
    file_name = os.path.join(data_dir, "bench_stats1.csv" )
    bench_stats = pd.read_csv(file_name, index_col=[0])
    print("Testing stats:")
    pd.testing.assert_frame_equal(tstats, bench_stats)
