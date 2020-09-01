## Location functions and tools

import os
from pytz import timezone
import datetime
import numpy as np
import pandas as pd
import power as pw
import xarray as xr
import json
import holidays
from ss_utilities.generic_tools import haversine_np
from functools import partial
import multiprocessing as mp
import fiona
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.prepared import prep

def get_fcst_locs(site_list, filename='/home/dtravers/winhome/Documents/dbs/weather/ecmwf_new/ecmwf2016-01-02T00.nc', n=1):
    """Function to find the lat/lon of the closest n forecast grid points to every site in ss_list. # 

    Parameters
    ----------
    site_list : DataFrame
        Contains a row for each site to be used, and must have column or index names: site_id, latitude, longitude.
    filename : str
        Name (including path) of the netcdf file which is used to find the set of all possible forecast locations to search through.
    n : int
        Number of forecast locations to find per row in site_list.  

    Returns
    -------
    Dataframe containing a row for each forecast location found (should be equal to number of sites multiplied by n).
    Dataframe has columns: site_id, latitude, longitude and dist (distance between the site_id and the lat/lon). 
    """

    data = xr.open_dataset(filename)
    f_loc = pd.DataFrame([], index = pd.MultiIndex.from_product((data.latitude.data, data.longitude.data), names=['latitude', 'longitude']))
    temp_loc = cross_locations(site_list, f_loc, n)
    temp_loc['f_id'] = temp_loc.latitude.map(str) + ':' + temp_loc.longitude.map(str)
    return(temp_loc)

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

    Returns
    -------
    Dataframe containing a row for each forecast location found (should be equal to number of sites multiplied by n).
    Dataframe has columns: site_id, latitude, longitude and dist (distance between the site_id and the lat/lon). 
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
    return(cross.groupby('site_id').dist.nsmallest(n).reset_index())

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
    locations : DataFrame
        DataFrame containing a row for each location requested.  
        The frame must have two columns named "latitude" and "longitude".  
        This frame is the format as output by the function get_fcst_locs (and cross_locations).

    Returns
    -------
    Dataframe with all weather variables from the nwp file.
    Dataframe is indexed by id, fcst_base, datetime, ahead.
    ahead is an integer for the number of days between the forecast base and the datetime.
    """

    file = os.path.join(filepath, filename)
    forecast_base = timezone('UTC').localize(datetime.datetime.strptime(filename[5:-3], '%Y-%m-%dT%H'))

    loc_dic = { 'latitude' : locations.latitude.unique(), \
               'longitude' : locations.longitude.unique()}
    DS = xr.open_dataset(file)
    if os.name == 'nt':
        dsel = DS.sel(loc_dic)
    else:
        dsel = DS.sel(**loc_dic)
    di = dsel.ssrd * 0.00027777778
    dt = dsel.t2m - 273.15
    du = dsel.u10
    dv = dsel.v10
    df = xr.concat([di, dt, du, dv], pd.Index(['irr', 'temp', 'u', 'v'], name='variable')).copy().to_series()
    # tidy up to ensure merge works:
    loc_temp = locations.set_index(['latitude', 'longitude'])
    loc_temp.drop(loc_temp.iloc[:, :], axis=1, inplace=True) # create df with just the index being lat/lon and no columns.
    loc_temp.columns = pd.MultiIndex.from_product([loc_temp.columns, ['']]) # add additional *column* level so it is multi-index columns with levels=2
    out_frame = pd.merge(df.unstack(0).unstack(0), loc_temp, how='inner', left_index=True, right_index=True).stack()  # merge to select only the values passed into locations
    #dloc['irr'] = dloc.groupby(['latitude', 'longitude']).irr.diff()
    # append a index with the forecast base (just one value for the file), and make tz_aware for future dates.
    out_frame = pd.concat([out_frame], keys=[forecast_base], names=['fcst_base']).tz_localize(('UTC'), level=-1)
    # create id by concatenating lat:lon
    out_frame['site_id'] = out_frame.index.get_level_values(1).map(str) + ':' + out_frame.index.get_level_values(2).map(str)
    out_frame.set_index('site_id', inplace=True, append=True)
    out_frame.reset_index(level=[1, 2], drop=True, inplace=True)
    # create a day ahead measure for referencing later
    out_frame['ahead'] = out_frame.index.get_level_values(1).date - out_frame.index.get_level_values(0).date
    out_frame = out_frame.reorder_levels([2, 0, 1])
    out_frame.set_index('ahead', append=True, inplace=True)
    out_frame.index.rename('datetime', 2, inplace=True)
    return(out_frame)

def populate_generic_dno_licence_region_locations(meta_df):
    meta_df.reset_index(inplace=True)
    with open(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config') , 'dno_latlon.json'), 'r') as jf:
        dno_latlon = json.load(jf)
    meta_df.loc[:, 'latitude'] = meta_df.site_id.map(lambda x: dno_latlon[str(x)[0:2]][0])
    meta_df.loc[:, 'longitude'] = meta_df.site_id.map(lambda x: dno_latlon[str(x)[0:2]][1])
    meta_df.set_index(['site_id'], drop=True, inplace=True)
    return(meta_df)

def get_longest_ss_ids(n=3):
    """ Function used for test runs, so not production quality or tested as such.
    Finds the n sites with the maximum number of days between the highest and lowest dates
    from the default Power hdf store.  
    """

    power = pw.Power()
    ss_list = power.store_summary()
    ss_list['diff'] = ss_list['max'] - ss_list['min']
    # find n sites with the largest history in the hdf store:
    ss_selected = ss_list.sort_values('diff', ascending=False).head(n).index.get_values()
    power.load_metadata(ss_selected)
    return(power.metadata.loc[ss_selected])

def apply_weekday(df, type='grouped', dummies=True):
    """ Returns dataframe with dummy columns for the weekdays requested.
    'grouped' groups all working days as 0, and Sundays and English holidays together.
    'individual' returns each daytype separately, and holidays as a Sunday daytype (=6).
    'holiday_individual' returns all working day as 0, and holidays separate from Sundays.
    'week_holiday_individual' returns each daytype separately and holidays separately (=7).
    """
    uk_hols = holidays.England()
    df = df.assign(weekday=df.index.shift(-1, freq='h').weekday)
    if (type=='week_holiday_individual') or (type=='holiday_individual'):
        holiday_day = 7
    else:
        holiday_day = 6
    df.loc[df.index.map(lambda x: (x + datetime.timedelta(hours=-1)).date() in uk_hols), 'weekday'] = holiday_day
    if (type == 'grouped') or (type == 'holiday_individual'):
        df.loc[df.weekday < 5, 'weekday'] = 0
    if dummies==True:
        df  = pd.concat([df, pd.get_dummies(df.weekday)], axis=1)
        df.drop('weekday', axis=1, inplace=True)
    df.columns = df.columns.astype(str)
    return(df)

def hdist(lat1, long1, lat2, long2):
    """
    Calculates haversine distance
    """
    R = 6371 #distances in km
    la1 = lat1 * np.pi / 180
    la2 = lat2 * np.pi / 180
    lo1 = long1 * np.pi / 180
    lo2 = long2 * np.pi / 180
    return np.arccos( np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo2 - lo1)) * R

def delta_lat(d): # calculates the number of degrees of latitude given distance in km
    R = 6371
    return(np.arcsin(d / R) * 180 / np.pi)

def delta_lon(d, lat): # calculates the number of degrees of longitude given the distance in km, and latitude.
    R = 6371
    return(np.arcsin(d / (R * np.cos(lat * np.pi / 180))) * 180 / np.pi)

def geoavg(lat, lon, data, radius, p=1):
    """ 
    Function to return the inverse-distance weighted average of the observations
    at a specified lat-lon point within a circle of radius radius.
    Parameters
    ----------
    lat : float
    lon : float
    data : obj:Series
        Series indexed by lat, lon, containing the observations
    radius : float
        Circle size (rather than square) within which we gather points    
    p : integer
        The inverse power - 1 is linear, 2 quadratic...
    """
    max_lat = lat + delta_lat(radius)
    min_lat = lat - delta_lat(radius)
    max_lon = lon + delta_lon(radius, lat)
    min_lon = lon - delta_lon(radius, lat)
    subset = pd.DataFrame(data.loc[min_lat : max_lat, min_lon : max_lon]).reset_index()
    data_name = data.name
    if len(subset) == 0:
        avg = np.nan
    else:
        subset['dist'] = subset.apply(lambda ser: (hdist(ser[0], ser[1], lat, lon)), axis=1, raw=True)
        subset['inv_dist'] = subset['dist']**(-1*p)
        subset = subset[subset['dist'] < radius]
        if len(subset) == 0:
            avg = np.nan
        else:
            avg = np.average(subset[data_name], weights = subset['inv_dist'])
    return(avg)

def geo_tuple(loc, data, radius, p):
    return(geoavg(loc[0], loc[1], data, radius, p))

# define function to determine if point is on land:
geoms = fiona.open(shpreader.natural_earth(resolution='50m', category='physical', name='land'))
land_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry']) for geom in geoms])
land = prep(land_geom)

def is_land(lon, lat):
    return land.contains(sgeom.Point(lon, lat))

def create_grid(lat_min, lat_max, lon_min, lon_max, lat_squares, lon_squares, results, metric, radius, power):
    """
    Function creates a grid of points across the bounded area and
    interpolates the metric from a unevenly distributed sample set onto the grid.
    User inverse-distance weighted average.
    ----------
    lat_min : float
        Bound of the gridded area.
    lat_max : float
    lon_min : float
    lon_max : float
    lat_squares: int
        Number of squares across the latitude.
    lon_squares : int
        Number of squares acrsoss the longitude.
    results : obj:DataFrame
        DataFrame containing columns "lat", "lon" and the metric (named next parameter).
        DataFrame contains all the raw result observations, and do not lie on a grid.
    metric : str
        The name of the metric - must be one of the column names of 'results'.
    radius : 
        Circle size within which we average the points    
    p : integer
        The inverse power - 1 is linear, 2 quadratic...
    """    
    t0 = datetime.datetime.now()
    results = results.set_index(['lat', 'lon'], drop=True)[metric]
    results.sort_index(level=['lat','lon'], inplace=True)
    uk_idx = pd.MultiIndex.from_product([np.linspace(lat_min, lat_max, np.int(lat_squares)), np.linspace(lon_min, lon_max, np.int(lon_squares))], names=['lat','lon'])
    uk_grid = pd.DataFrame([], index=uk_idx)
    uk_grid = uk_grid.sort_index()
    uk_grid = uk_grid.reset_index()
    uk_grid['land'] = uk_grid.apply(lambda x: is_land(x.lon, x.lat), axis=1) # vectorized land apply.
    f_part = partial(geo_tuple, data=results, radius=radius, p=power)
    with mp.Pool(12) as p:
        uk_grid.loc[uk_grid.land, 'PV'] = np.array(p.map(f_part, uk_grid.loc[uk_grid.land].values))
    uk_grid = uk_grid.set_index(['lat', 'lon'])
    #uk_grid = uk_grid.interpolate()
    uk_grid.loc[uk_grid.land == False, 'PV'] = np.nan 
    print(datetime.datetime.now() - t0)
    return(uk_grid)