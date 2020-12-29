import os
import pandas as pd
from location_data_utils import create_grid

testdir = '/mnt/c/Users/dantravers/Documents/GitHub/ss-obs/tests/test_data/'

def test_create_grid():
    res = pd.read_csv(os.path.join(testdir, 'create_grid_raw_results.csv'), dayfirst=True, index_col=0)
    bench = pd.read_csv(os.path.join(testdir, 'create_grid_output.csv'), index_col=[0, 1])
    uk_grid = create_grid(50, 51, -6, -4, 20, 20, res, 'MAE', 40, 1.2)

    pd.testing.assert_frame_equal(uk_grid, bench)