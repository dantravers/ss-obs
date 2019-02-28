# check the observations and metadata loaded from a mix of hdf5 store and db are 
# all loaded into Midas object correctly and completely.
# We start with a hdf store (write_test_copy.h5) with 2 days of data for 2 sites already populated.
# Additional data for 1 site and 2 days should be loaded from the database and integrated to hdf correctly.
# Second test saves the results of object to the hdf store and then retrieves and check they are the same.
# 26/2/19

import shutil
import os
import numpy as np
import pandas as pd
import sys
import datetime
import midas

s = datetime.datetime(2016, 12, 31).date() 
e = datetime.datetime(2017,1, 4).date()
current_dir = 'C:/Users/Dan Travers/Documents/GitHub/ss-obs/tests'
orig = os.path.join(current_dir, 'midas_2sites_2days.h5')
hdf_scratch = os.path.join(current_dir, 'scratch.h5')
shutil.copy2(orig, hdf_scratch)
hdf_bench = os.path.join(current_dir, 'midas_3sites_4days.h5')
with pd.HDFStore(hdf_bench, 'r') as hdf:
    weather_obs_bench = hdf.select('obs')
with pd.HDFStore(hdf_bench, 'r') as hdf:
    weather_metadata_bench = hdf.select('metadata')

weather = midas.Midas()
weather.config['hdf5']['store_name'] = hdf_scratch
weather.load_data([842, 676, 384], s, e,  goto_db='')

#test results are as expected from initial load:
pd.testing.assert_frame_equal(weather.metadata.sort_index(), weather_metadata_bench.sort_index())
pd.testing.assert_frame_equal(weather.obs.sort_index(), weather_obs_bench.sort_index())

# save results to hdf, then create new object and lift results from hdf
weather.save_to_hdf()
weather2 = midas.Midas()
weather2.config['hdf5']['store_name'] = hdf_scratch
weather2.load_data([842, 676, 384], s, e,  goto_db='Never')
pd.testing.assert_frame_equal(weather2.metadata.sort_index(level=0), weather_metadata_bench.sort_index(level=0))
pd.testing.assert_frame_equal(weather2.obs.sort_index(level=0), weather_obs_bench.sort_index(level=0))

os.remove(hdf_scratch)