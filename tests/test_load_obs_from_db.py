# check the observations and metadata loaded from midas database are equal to the reference data stored in file.
# 26/2/19

import numpy as np
import pandas as pd
import sys
import os
import shutil
import datetime
import midas
import power as pw

# setup common data
current_dir = 'C:/Users/Dan Travers/Documents/GitHub/ss-obs/tests'

def test_load_midas_obs_from_db():
    s = datetime.datetime(2017, 1, 1).date() 
    e = datetime.datetime(2017,1, 3).date()
    with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/midas_2sites_2days.h5', 'r') as hdf:
        weather_obs_bench = hdf.select('obs')
    with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/midas_2sites_2days.h5', 'r') as hdf:
        weather_meta_bench = hdf.select('metadata')
    weather = midas.Midas(1)
    weather.load_data([842, 676], s, e,  goto_db='Always')

    pd.testing.assert_frame_equal(weather.metadata, weather_meta_bench)
    pd.testing.assert_frame_equal(weather.obs, weather_obs_bench)

def test_load_pvstream_obs_from_db():
    s = datetime.datetime(2017, 1, 1).date() 
    e = datetime.datetime(2017,1, 3).date()
    with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/power_2sites_2days.h5', 'r') as hdf:
        power_obs_bench = hdf.select('obs')
    with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/power_2sites_2days.h5', 'r') as hdf:
        power_metadata_bench = hdf.select('metadata')  
    power = pw.Power(1)
    power.load_data([3111, 6730], s, e,  goto_db='Always')

    pd.testing.assert_frame_equal(power.metadata, power_metadata_bench)
    pd.testing.assert_frame_equal(power.obs, power_obs_bench)

def test_load_additional_midas_obs_from_db():
    s = datetime.datetime(2016, 12, 31).date() 
    e = datetime.datetime(2017,1, 4).date()
    orig = os.path.join(current_dir, 'midas_2sites_2days.h5')
    hdf_scratch = os.path.join(current_dir, 'scratch.h5')
    shutil.copy2(orig, hdf_scratch)
    hdf_bench = os.path.join(current_dir, 'midas_3sites_4days.h5')
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
    orig = os.path.join(current_dir, 'power_2sites_2days.h5')
    hdf_scratch = os.path.join(current_dir, 'scratch.h5')
    shutil.copy2(orig, hdf_scratch)
    hdf_bench = os.path.join(current_dir, 'power_3sites_4days.h5')
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