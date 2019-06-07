# check the observations and metadata loaded from midas database are equal to the reference data stored in file.
# 26/2/19

import datetime
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

import midas
import power as pw
import wforecast as wf
from location_data_utils import (cross_locations, get_fcst_locs,
                                 get_longest_ss_ids, read_netcdf_file)

# setup common data
data_dir = 'C:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/test_data'
netcdf_dir = os.path.join(data_dir, 'netcdf')
netcdf_file = 'ecmwf2016-01-02T00%3A00%3A00.nc'

def test_load_midas_obs_from_db():
    s = datetime.datetime(2017, 1, 1).date() 
    e = datetime.datetime(2017,1, 3).date()
    hdf_midas_bench = os.path.join(data_dir, 'midas_2sites_2days.h5')
    with pd.HDFStore(hdf_midas_bench, 'r') as hdf:
        weather_obs_bench = hdf.select('obs')
    with pd.HDFStore(hdf_midas_bench, 'r') as hdf:
        weather_meta_bench = hdf.select('metadata')
    weather = midas.Midas(1)
    weather.load_data([842, 676], s, e,  goto_db='Always')

    pd.testing.assert_frame_equal(weather.metadata, weather_meta_bench)
    pd.testing.assert_frame_equal(weather.obs, weather_obs_bench)

def test_load_pvstream_obs_from_db():
    s = datetime.datetime(2017, 1, 1).date() 
    e = datetime.datetime(2017,1, 3).date()
    hdf_power_bench = os.path.join(data_dir, 'power_2sites_2days.h5')
    with pd.HDFStore(hdf_power_bench, 'r') as hdf:
        power_obs_bench = hdf.select('obs')
    with pd.HDFStore(hdf_power_bench, 'r') as hdf:
        power_metadata_bench = hdf.select('metadata')  
    power = pw.Power(1)
    power.load_data([3111, 6730], s, e,  goto_db='Always')

    pd.testing.assert_frame_equal(power.metadata, power_metadata_bench)
    pd.testing.assert_frame_equal(power.obs, power_obs_bench)

def test_load_additional_midas_obs_from_db():
    s = datetime.datetime(2016, 12, 31).date() 
    e = datetime.datetime(2017,1, 4).date()
    orig = os.path.join(data_dir, 'midas_2sites_2days.h5')
    hdf_scratch = os.path.join(data_dir, 'scratch.h5')
    shutil.copy2(orig, hdf_scratch)
    hdf_bench = os.path.join(data_dir, 'midas_3sites_4days.h5')
    with pd.HDFStore(hdf_bench, 'r') as hdf:
        weather_obs_bench = hdf.select('obs')
    with pd.HDFStore(hdf_bench, 'r') as hdf:
        weather_metadata_bench = hdf.select('metadata')

    weather = midas.Midas(1)
    weather.config['hdf5']['store_name'] = hdf_scratch
    weather.load_data([842, 676, 384], s, e,  goto_db='')

    #test results are as expected from initial load:
    pd.testing.assert_frame_equal(weather.metadata.sort_index(), weather_metadata_bench.sort_index())
    pd.testing.assert_frame_equal(weather.obs.sort_index(), weather_obs_bench.sort_index())

    # save results to hdf, then create new object and lift results from hdf
    weather.save_to_hdf()
    weather2 = midas.Midas(1)
    weather2.config['hdf5']['store_name'] = hdf_scratch
    weather2.load_data([842, 676, 384], s, e,  goto_db='Never')
    pd.testing.assert_frame_equal(weather2.metadata.sort_index(level=0), weather_metadata_bench.sort_index(level=0))
    pd.testing.assert_frame_equal(weather2.obs.sort_index(level=0), weather_obs_bench.sort_index(level=0))

    os.remove(hdf_scratch)

def test_load_additional_pvstream_obs_from_db():
    s = datetime.datetime(2016, 12, 31).date() 
    e = datetime.datetime(2017,1, 4).date()
    orig = os.path.join(data_dir, 'power_2sites_2days.h5')
    hdf_scratch = os.path.join(data_dir, 'scratch.h5')
    shutil.copy2(orig, hdf_scratch)
    hdf_bench = os.path.join(data_dir, 'power_3sites_4days.h5')
    with pd.HDFStore(hdf_bench, 'r') as hdf:
        power_obs_bench = hdf.select('obs')
    with pd.HDFStore(hdf_bench, 'r') as hdf:
        power_metadata_bench = hdf.select('metadata')  

    power = pw.Power(1)
    power.config['hdf5']['store_name'] = hdf_scratch
    power.load_data([3111, 6730, 8441], s, e,  goto_db='')

    # test results are as expected from initial load:
    pd.testing.assert_frame_equal(power.metadata.sort_index(level=0), power_metadata_bench.sort_index(level=0))
    pd.testing.assert_frame_equal(power.obs.sort_index(), power_obs_bench.sort_index())

    # save results to hdf, then create new object and lift results from hdf
    power.save_to_hdf()
    power2 = pw.Power(1)
    power2.config['hdf5']['store_name'] = hdf_scratch
    power2.load_data([3111, 6730, 8441], s, e,  goto_db='Never')
    pd.testing.assert_frame_equal(power2.metadata.sort_index(level=0), power_metadata_bench.sort_index(level=0))
    pd.testing.assert_frame_equal(power2.obs.sort_index(level=0), power_obs_bench.sort_index(level=0))

    os.remove(hdf_scratch)

def test_locations_fn_and_read_netcdf_file():
    site_list = pd.read_pickle(os.path.join(data_dir, 'ss_list.pickle'))
    locations = get_fcst_locs(site_list, os.path.join(netcdf_dir, netcdf_file), 1)
    day = read_netcdf_file(netcdf_file, netcdf_dir, locations)
    day_bench = pd.read_pickle(os.path.join(data_dir, 'day_bench.pickle'))
    pd.testing.assert_frame_equal(day.sort_index(level=[0, 1, 2, 3]), day_bench.sort_index(level=[0, 1, 2, 3]))

def test_wforecast_load_from_file():
    s = datetime.datetime(2016, 1, 1).date() 
    e = datetime.datetime(2016,1, 5).date()
    with open(os.path.join(data_dir, 'wforecast_bench.pickle'), 'rb') as f:
        wforecast_bench = pickle.load(f)
    site_list = pd.read_pickle(os.path.join(data_dir, 'ss_list.pickle'))
    locations = get_fcst_locs(site_list, os.path.join(netcdf_dir, netcdf_file), 1)
    
    wforecast = wf.WForecast(3)
    wforecast.load_data(locations, netcdf_dir, s, e, 'File')

    pd.testing.assert_frame_equal(wforecast.obs.sort_index(level=[0, 1, 2]), wforecast_bench.obs.sort_index(level=[0, 1, 2]))
    pd.testing.assert_frame_equal(wforecast.metadata.sort_index(level=[0]), wforecast_bench.metadata.sort_index(level=[0]))

def test_wforecast_get_obs_function():
    with open(os.path.join(data_dir, 'wforecast_bench.pickle'), 'rb') as f:
        wforecast = pickle.load(f)
    obs_bench = pd.read_pickle(os.path.join(data_dir, 'obs_bench.pickle'))

    pd.testing.assert_frame_equal(wforecast.get_obs(3), obs_bench)
    