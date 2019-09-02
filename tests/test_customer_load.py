# Test to test the customer load data modelling is working
# Derived class from Power class

import os
import numpy as np
import pandas as pd
import datetime 
import power as pw

# directory for all test data (hdfs & bench_*.csv)
data_dir = 'C:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/test_data'
# setup dates and Load data
start = datetime.date(2016,1,1)
end = datetime.date(2018, 1, 1)

#load.config['hdf5']['store_name'] = os.path.join(data_dir, "load_test_data.h5")
xls_filename = "customer_load_test.xlsx"
csv_filename = "112233445566.csv"

def test_customer_excel_load():
    load = pw.Load(2)    
    load.load_from_file(os.path.join(data_dir, xls_filename))
    # test metadata is correct
    # test observation data is correct
    bench = pd.read_csv(os.path.join(data_dir, 'bench_customer_load_obs.csv'), index_col=[0, 1], parse_dates=True)
    pd.testing.assert_frame_equal(load.obs, bench)

def test_customer_csv_load():
    load = pw.Load(2)    
    load.load_from_file(os.path.join(data_dir, csv_filename))
    # test metadata is correct
    # test observation data is correct
    bench = pd.read_csv(os.path.join(data_dir, 'bench_customer_112233445566_obs.csv'), index_col=[0, 1], parse_dates=True)
    pd.testing.assert_frame_equal(load.obs, bench)
