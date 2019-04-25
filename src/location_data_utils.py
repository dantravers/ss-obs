## Location functions and tools

import os
from pytz import timezone
import datetime
import pandas as pd
import power as pw
import xarray as xr
from ss_utilities.generic_tools import haversine_np

def get_fcst_locs(filename, ss_list, n=4):
    """Function to find the lat/lon of the closest n forecast grid points to every site in ss_list.
    """
    data = xr.open_dataset(filename)
    f_loc = pd.DataFrame([], index = pd.MultiIndex.from_product((data.latitude.data, data.longitude.data), names=['latitude', 'longitude']))
    return(cross_locations(ss_list, f_loc, n))

def cross_locations(primary, secondary, n=4):
    """ Function to find the n closest secondary locations to each primary location.

    Parameters
    ----------
    primary : DataFrame
        Contains a row for each site to be used, and must have column or index names: site_id, latitude, longitude.
    secondary : DataFrame
        Should have index or columns: latitude, longitude.
    n : int
        The number of secondary locations to find close to each primary location.
    """

    primary['dummy'] = secondary['dummy'] = 1
    cross = pd.merge(primary.reset_index(), secondary.reset_index(), on='dummy', suffixes=('', '_sec'))\
    .drop('dummy', axis=1)
    cross['dist'] = haversine_np(cross.latitude.values, cross.longitude.values, \
                                 cross.latitude_sec.values, cross.longitude_sec.values)
    cross.set_index(['latitude_sec', 'longitude_sec'], inplace=True)
    cross.index.rename([ 'latitude', 'longitude'], inplace=True)
    primary.drop('dummy', axis=1, inplace=True)
    secondary.drop('dummy', axis=1, inplace=True)
    return(cross.groupby('site_id').dist.nsmallest(n))

def read_netcdf_file(filename, filepath, locations):
    """ Read the netCDF file provided and return the weather variables on the locations requested.
    
    Notes
    -----
    Returns a dataframe indexed on latitude, longitude, (forecast) time, base time (unique entry), weather variable.  
    Weather variables supported in forecast are: 
    irradiance (integrated joules, divided by 3600), temperature (C), u and v wind components (m/s)

    Parameters
    ----------
    filename : str
        The netCDF filename from ECMWF (this has a datetime encoded in the format).
    filepath : str
        Folder path containing the file.
    locations : Series
        Series containing a row for each location requested.  
        This is the format as output by the function cross_locations.
        The lat / lon in index levels 1, 2 (index level 0 is the pv farm site_id from cross_locations).  
        The data is not used.

    Returns
    -------
    Dataframe with all weather variables from the nwp file.
    Dataframe is indexed by id, fcst_base, datetime, ahead.
    ahead is an integer for the number of days between the forecast base and the datetime.
    """

    file = os.path.join(filepath, filename)
    forecast_base = timezone('UTC').localize(datetime.datetime.strptime(filename[5:-3], '%Y-%m-%dT%H%%3A%M%%3A%S'))
    loc_dic = { 'latitude' : locations.index.get_level_values(1).unique(), \
               'longitude' : locations.index.get_level_values(2).unique()}
    DS = xr.open_dataset(file)
    dsel = DS.sel(loc_dic)
    di = dsel.ssrd * 0.000277778
    dt = dsel.t2m - 273.15
    du = dsel.u10
    dv = dsel.v10
    df = xr.concat([di, dt, du, dv], pd.Index(['irr', 'temp', 'u', 'v'], name='variable')).to_series()
    # tidy up to ensure merge works:
    loc_temp = pd.DataFrame(locations).reset_index(0, drop=True).drop('dist', axis=1) # create df with just the index being lat/lon and no columns.
    loc_temp.columns = pd.MultiIndex.from_product([loc_temp.columns, ['']]) # add additional column level so it is multi-index with levels=2
    out_frame = pd.merge(df.unstack(0).unstack(0), loc_temp, how='inner', left_index=True, right_index=True).stack()  # merge to select only the values passed into locations
    #dloc['irr'] = dloc.groupby(['latitude', 'longitude']).irr.diff()
    # append a index with the forecast base (just one value for the file), and make tz_aware for future dates.
    out_frame = pd.concat([out_frame], keys=[forecast_base], names=['fcst_base']).tz_localize(('UTC'), level=-1)
    # create id by concatenating lat:lon
    out_frame['id'] = out_frame.index.get_level_values(1).map(str) + ':' + out_frame.index.get_level_values(2).map(str)
    out_frame.set_index('id', inplace=True, append=True)
    out_frame.reset_index(level=[1, 2], drop=True, inplace=True)
    # create a day ahead measure for referencing later
    out_frame['ahead'] = out_frame.index.get_level_values(1).date - out_frame.index.get_level_values(0).date
    out_frame = out_frame.reorder_levels([2, 0, 1])
    out_frame.set_index('ahead', append=True, inplace=True)
    out_frame.index.rename('datetime', 2, inplace=True)
    return(out_frame)

def get_longest_ss_ids(n=3):
    power = pw.Power()
    ss_list = power.store_summary()
    ss_list['diff'] = ss_list['max'] -ss_list['min']
    # find n sites with the largest history in the hdf store:
    ss_selected = ss_list.sort_values('diff', ascending=False).head(n).index.get_values()
    power.load_metadata(ss_selected)
    return(power.metadata.loc[ss_selected])