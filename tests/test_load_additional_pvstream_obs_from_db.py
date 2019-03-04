# check the observations and metadata loaded from a mix of hdf5 store and db are 
# all loaded into Power object correctly and completely.
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
import power

s = datetime.datetime(2016, 12, 31).date() 
e = datetime.datetime(2017,1, 4).date()
current_dir = 'C:/Users/Dan Travers/Documents/GitHub/ss-obs/tests'
orig = os.path.join(current_dir, 'power_2sites_2days.h5')
hdf_scratch = os.path.join(current_dir, 'scratch.h5')
shutil.copy2(orig, hdf_scratch)
hdf_bench = os.path.join(current_dir, 'power_3sites_4days.h5')
with pd.HDFStore(hdf_bench, 'r') as hdf:
    power_obs_bench = hdf.select('obs')
with pd.HDFStore(hdf_bench, 'r') as hdf:
    power_metadata_bench = hdf.select('metadata')  

pw = power.Power(1)
pw.config['hdf5']['store_name'] = hdf_scratch
pw.load_data([3111, 6730, 8441], s, e,  goto_db='')

# test results are as expected from initial load:
pd.testing.assert_frame_equal(pw.metadata.sort_index(level=0), power_metadata_bench.sort_index(level=0))
pd.testing.assert_frame_equal(pw.obs.sort_index(), power_obs_bench.sort_index())

# save results to hdf, then create new object and lift results from hdf
pw.save_to_hdf()
pw2 = power.Power(1)
pw2.config['hdf5']['store_name'] = hdf_scratch
pw2.load_data([3111, 6730, 8441], s, e,  goto_db='Never')
pd.testing.assert_frame_equal(pw2.metadata.sort_index(level=0), power_metadata_bench.sort_index(level=0))
pd.testing.assert_frame_equal(pw2.obs.sort_index(level=0), power_obs_bench.sort_index(level=0))

os.remove(hdf_scratch)