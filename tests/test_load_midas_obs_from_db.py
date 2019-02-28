# check the observations loaded from database are equal to the reference observations stored in file.
# 26/2/19

import numpy as np
import pandas as pd
import sys
import datetime
sys.path.append('C:/Users/Dan Travers/Documents/GitHub/ss-obs/ss-obs')
#sys.path.append('ss-obs/')
import midas

s = datetime.datetime(2017, 1, 1).date() 
e = datetime.datetime(2017,1, 3).date()
with pd.HDFStore('c:/Users/Dan Travers/Documents/GitHub/ss-obs/tests/test_data.h5', 'r') as hdf:
    weather_bench = hdf.select('weather_test_from_db_842_676')
weather = midas.Midas()
weather.load_data([842, 676], s, e,  goto_db='Always')

pd.testing.assert_frame_equal(weather.obs, weather_bench)