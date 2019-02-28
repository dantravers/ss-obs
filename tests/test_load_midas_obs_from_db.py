# check the observations and metadata loaded from midas database are equal to the reference data stored in file.
# 26/2/19

import numpy as np
import pandas as pd
import sys
import datetime
import midas

s = datetime.datetime(2017, 1, 1).date() 
e = datetime.datetime(2017,1, 3).date()
with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/midas_2sites_2days.h5', 'r') as hdf:
    weather_obs_bench = hdf.select('obs')
with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/midas_2sites_2days.h5', 'r') as hdf:
    weather_meta_bench = hdf.select('metadata')
weather = midas.Midas()
weather.load_data([842, 676], s, e,  goto_db='Always')

pd.testing.assert_frame_equal(weather.metadata, weather_meta_bench)
pd.testing.assert_frame_equal(weather.obs, weather_obs_bench)