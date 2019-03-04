# check the observations and metadata loaded from pvstream database are equal to the reference data stored in file.
# 26/2/19

import numpy as np
import pandas as pd
import sys
import datetime
import power

s = datetime.datetime(2017, 1, 1).date() 
e = datetime.datetime(2017,1, 3).date()
with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/power_2sites_2days.h5', 'r') as hdf:
    power_obs_bench = hdf.select('obs')
with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/power_2sites_2days.h5', 'r') as hdf:
    power_metadata_bench = hdf.select('metadata')  
power = power.Power(1)
power.load_data([3111, 6730], s, e,  goto_db='Always')

pd.testing.assert_frame_equal(power.metadata, power_metadata_bench)
pd.testing.assert_frame_equal(power.obs, power_obs_bench)
