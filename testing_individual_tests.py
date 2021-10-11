import inspect
import numpy as np
import xarray as xr
import pandas as pd
from ioos_qc.config import Config
from ioos_qc import qartod
from ioos_qc.utils import load_config_as_dict as loadconfig
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import timedelta


# read in file and profile time
f = './test_files/maracoos_02-20210716T1814/maracoos_02_20210716T190208Z_dbd.nc'
ds = xr.open_dataset(f)
profile_time = pd.to_datetime(ds.profile_time.values)
profile_lon = ds.profile_lon.values
profile_lat = ds.profile_lat.values
times = ds.time.values

# actually probably want to start with deployment-specific config as default,
# if it exists, but figure this out later

# start with global config as default
config_file = './config/global_configs.yml'
c_global = Config(config_file)

c = c_global.config['contexts'][0]

# pull in regional boundaries
# Ross Sea boundaries I have set up cross over +180 border - issue to put off until later, but something to keep in mind
region_bounds=loadconfig('./config/regional_boundaries.yml') # maybe this file goes somewhere else, putting it with configs for now
best_region = {'priority': 100, 'region': 'global'}
# look for highest priority region containing profile lon and lat, with existing config
for region in region_bounds['regions']:
    if region['priority'] >= best_region['priority']:
        continue
    if not os.path.exists(os.path.join('./config/',region['region']+'_configs.yml')):
        continue
    if Polygon(list(zip(region['boundaries']['longitude'], region['boundaries']['latitude']))).contains(Point(profile_lon, profile_lat)):
        best_region = region

if best_region['region'] != 'global':
    # pull in regional config
    config_file = os.path.join('./config/',best_region['region']+'_configs.yml')
    c_region = Config(config_file)

    # loop through different time ranges and replace existing
    # config 'c' if an appropriate time window with higher
    # priority is found
    for c0 in c_region.config['contexts']:
        if c0['priority'] >= c['priority']:
            continue
        t0 = c0['window']['starting'].replace(year = profile_time.year)
        t1 = c0['window']['ending'].replace(year = profile_time.year)
        if np.logical_and(profile_time >= t0, profile_time <= t1):
            c = c0

c['window']['starting'] = c['window']['starting'].replace(year = profile_time.year)
c['window']['ending'] = c['window']['ending'].replace(year = profile_time.year)

# loop through streams
for sensor, config_info in c['streams'].items():
    if sensor not in ds.data_vars:
        continue
    # grab data for sensor
    data = ds[sensor].values
    # identify where not nan
    non_nan_ind = np.invert(np.isnan(data))
    # get locations of non-nans
    non_nan_i = np.where(non_nan_ind)[0]
    # get time interval (s) between non-nan points
    tdiff = np.diff(times[non_nan_ind]).astype('timedelta64[s]').astype(float)
    # locate time intervals > 5 min
    tdiff_long = np.where(tdiff > 60*5)[0]
    # original locations of where time interval is long
    tdiff_long_i = np.append(non_nan_i[tdiff_long], non_nan_i[tdiff_long+1])

    for test, cinfo in config_info['qartod'].items():
        if test == 'pressure_test':
            # pressure test
            flag_vals = 2 * np.ones(np.shape(data))
            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING
            flag_vals[non_nan_ind] = qartod.pressure_test(inp = data[non_nan_ind],
                                                          tinp = times[non_nan_ind],
                                                          **cinfo)
            # write to nc with attributes
        elif test == 'climatology_test':
            # climatology test
            climatology_settings = {'tspan': [c['window']['starting']-timedelta(days=2), c['window']['ending']+timedelta(days=2)],
                                    'fspan': None, 'vspan': None, 'zspan': None}

            # if no set depth range, apply thresholds to full profile depth
            if 'depth_range' not in cinfo.keys():
                climatology_settings['zspan'] = [0, np.nanmax(ds.depth.values)]
                if 'suspect_span' in cinfo.keys():
                    climatology_settings['vspan'] = cinfo['suspect_span']
                if 'fail_span' in cinfo.keys():
                    climatology_settings['fspan'] = cinfo['fail_span']
                climatology_config = qartod.ClimatologyConfig()
                climatology_config.add(**climatology_settings)
                flag_vals = qartod.climatology_test(config=climatology_config,
                                                    inp=data,
                                                    tinp=times,
                                                    zinp=ds.depth.values)
            else:
                # if one depth range provided, apply thresholds only to that depth range
                if len(np.shape(cinfo['depth_range'])) == 1:
                    climatology_settings = {'tspan': [c['window']['starting']-timedelta(days=2), c['window']['ending']+timedelta(days=2)],
                                            'fspan': cinfo['depth_range'],
                                            'vspan': None, 'zspan': None}
                    if 'suspect_span' in cinfo.keys():
                        climatology_settings['vspan'] = cinfo['suspect_span']
                    if 'fail_span' in cinfo.keys():
                        climatology_settings['fspan'] = cinfo['fail_span']
                    climatology_config = qartod.ClimatologyConfig()
                    climatology_config.add(**climatology_settings)
                    flag_vals = qartod.climatology_test(config=climatology_config,
                                                        inp=data,
                                                        tinp=times,
                                                        zinp=ds.depth.values)
                else: # if different thresholds for multiple depth ranges, loop through each
                    flag_vals = 2 * np.ones(np.shape(data))
                    for z_int in len(cinfo['depth_range']):
                        climatology_settings = {'tspan': [c['window']['starting']-timedelta(days=2), c['window']['ending']+timedelta(days=2)],
                                                'fspan': cinfo['depth_range'][z_int],
                                                'vspan': None, 'zspan': None}
                        if 'suspect_span' in cinfo.keys():
                            climatology_settings['vspan'] = cinfo['suspect_span'][z_int]
                        if 'fail_span' in cinfo.keys():
                            climatology_settings['fspan'] = cinfo['fail_span'][z_int]
                        climatology_config = qartod.ClimatologyConfig()
                        climatology_config.add(**climatology_settings)
                        z_ind = np.logical_and(ds.depth.values > cinfo['depth_range'][z_int][0],
                                               ds.depth.values <= cinfo['depth_range'][z_int][1])
                        flag_vals[z_ind] = qartod.climatology_test(config=climatology_config,
                                                            inp=data[z_ind],
                                                            tinp=times[z_ind],
                                                            zinp=ds.depth.values[z_ind])
            # write to nc with attributes
        elif test == 'spike_test':
            # spike test
            spike_settings = {'suspect_threshold': None, 'fail_threshold': None}
            # convert original threshold from units/s to units/average-timestep
            if 'suspect_threshold' in cinfo.keys():
                spike_settings['suspect_threshold'] = cinfo['suspect_threshold'] * np.nanmedian(tdiff)
            if 'fail_threshold' in cinfo.keys():
                spike_settings['fail_threshold'] = cinfo['fail_threshold'] * np.nanmedian(tdiff)
            flag_vals = 2 * np.ones(np.shape(data))
            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING
            flag_vals[non_nan_ind] = qartod.spike_test(inp=data[non_nan_ind],
                                                          **spike_settings)
            # flag as unknown on either end of long time gap
            flag_vals[tdiff_long_i] = qartod.QartodFlags.UNKNOWN
            # write to nc with attributes
        elif test == 'rate_of_change_test':
            # rate of change test
            flag_vals = 2 * np.ones(np.shape(data))
            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING
            flag_vals[non_nan_ind] = qartod.rate_of_change_test(inp=data[non_nan_ind],
                                                          tinp=times[non_nan_ind],
                                                          **cinfo)
            # write to nc with attributes