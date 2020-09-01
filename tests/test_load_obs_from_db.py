# check the observations and metadata loaded from midas database are equal to the reference data stored in file.
# 26/2/19

import datetime
import os
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
if os.name == 'nt':
    data_dir = 'C:/Users/DanTravers/Documents/GitHub/ss-obs/tests/test_data'
else:
    data_dir = '/home/dtravers/winhome/Documents/GitHub/ss-obs/tests/test_data'
netcdf_dir = os.path.join(data_dir, 'netcdf')
netcdf_file = 'ecmwf2016-01-02T00.nc'
site_list = pd.read_csv(os.path.join(data_dir, 'site_list.csv'))

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

    pd.testing.assert_frame_equal(power.metadata.sort_index(), power_metadata_bench.sort_index())
    pd.testing.assert_frame_equal(power.obs.sort_index(), power_obs_bench.sort_index())

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
    locations = get_fcst_locs(site_list, os.path.join(netcdf_dir, netcdf_file), 1)
    day = read_netcdf_file(netcdf_file, netcdf_dir, locations)
    day.to_csv(os.path.join(data_dir, 'day_bench.csv'))
    day_bench = pd.read_csv(os.path.join(data_dir, 'day_bench.csv'), \
        dtype={'irr':np.float32, 'temp':np.float32, 'u':np.float32, 'v':np.float32}, parse_dates=[1, 2], dayfirst=True)
    day_bench.ahead = pd.to_timedelta(day_bench.ahead)
    day_bench.set_index(['site_id', 'fcst_base', 'datetime', 'ahead'], inplace=True)
    pd.testing.assert_frame_equal(day.sort_index(level=[0, 1, 2, 3]), day_bench.sort_index(level=[0, 1, 2, 3]), check_names=False)

def test_wforecast_load_from_file():
    s = datetime.datetime(2016, 1, 1).date() 
    e = datetime.datetime(2016,1, 5).date()
    locations = get_fcst_locs(site_list, os.path.join(netcdf_dir, netcdf_file), 1)
    
    wforecast = wf.WForecast(3)
    wforecast.load_data(locations, netcdf_dir, s, e, 'File')
    wforecast.metadata.latitude = wforecast.metadata.latitude.astype(np.float64)
    wforecast.metadata.longitude = wforecast.metadata.longitude.astype(np.float64)
    #wforecast.metadata.to_csv('wforecast_bench_meta.csv')
    #wforecast.obs.to_csv('wforecast_bench_obs.csv')

    wforecast_meta_bench = pd.read_csv(os.path.join(data_dir, 'wforecast_meta_bench.csv'), index_col=0)

    #read obs file and parse dates & timedeltas:
    wforecast_obs_bench = pd.read_csv(os.path.join(data_dir, 'wforecast_obs_bench.csv'), \
        dtype={'irr':np.float32, 'temp':np.float32, 'u':np.float32, 'v':np.float32}, parse_dates=[1, 2], dayfirst=True)
    wforecast_obs_bench.ahead = pd.to_timedelta(wforecast_obs_bench.ahead)
    wforecast_obs_bench.set_index(['site_id', 'fcst_base', 'datetime', 'ahead'], inplace=True)
    #      wforecast_obs_bench.irr = wforecast_obs_bench.irr.astype(np.float32)
    print(wforecast.obs.sort_index(level=[0, 1, 2]).head(2))
    print(wforecast_obs_bench.sort_index(level=[0, 1, 2]).head(2))
    pd.testing.assert_frame_equal(wforecast.obs.sort_index(level=[0, 1, 2]), wforecast_obs_bench.sort_index(level=[0, 1, 2]), check_names=False)
    print(wforecast.metadata.sort_index(level=[0]).head(2))
    print(wforecast_meta_bench.sort_index(level=[0]).head(2))
    pd.testing.assert_frame_equal(wforecast.metadata.sort_index(level=[0]), wforecast_meta_bench.sort_index(level=[0]))
