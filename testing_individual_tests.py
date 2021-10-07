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

if best_region['region'] is not 'global':
    # pull in regional config
    config_file = os.path.join('./config/',best_region['region']+'_configs.yml')
    c_region = Config(config_file)

    # loop through different time ranges and replace existing
    # config 'c' if an appropriate time window with higher
    # priority is found
    for c0 in c_region['contexts']:
        if c0['priority'] >= c['priority']:
            continue
        t0 = c0['window']['starting'].replace(year = profile_time.year)
        t1 = c0['window']['ending'].replace(year = profile_time.year)
        if np.logical_and(profile_time >= t0, profile_time <= t1):
            c = c0

# loop through streams
for sensor in c['streams']:
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

    for test in c['streams'][sensor]['qartod']:
        if test == 'pressure_test':
            # pressure test
            flag_vals = 2 * np.ones(np.shape(data))
            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING
            flag_vals[non_nan_ind] = qartod.pressure_test(inp = data[non_nan_ind],
                                                          tinp = times[non_nan_ind],
                                                          **c['streams'][sensor]['qartod'][test])
            # write to nc with attributes
        elif test == 'climatology_test':
            # climatology test
            climatology_settings = {'fspan': None, 'vspan': None, 'zspan': None}

            # if no set depth range, apply thresholds to full profile depth
            if 'depth_range' not in c['streams'][sensor]['qartod'][test].keys():
                climatology_settings['zspan'] = [0, np.max(ds.depth.values)]
                if 'suspect_span' in c['streams'][sensor]['qartod'][test].keys():
                    climatology_settings['vspan'] = c['streams'][sensor]['qartod'][test]['suspect_span']
                if 'fail_span' in c['streams'][sensor]['qartod'][test].keys():
                    climatology_settings['fspan'] = c['streams'][sensor]['qartod'][test]['fail_span']
                climatology_config = ClimatologyConfig()
                climatology_config.add(**climatology_settings)
                flag_vals = qartod.climatology_test(config=climatology_config,
                                                    inp=data,
                                                    tinp=times,
                                                    zinp=ds.depth.values)
            else:
                # if one depth range provided, apply thresholds only to that depth range
                if len(np.shape(c['streams'][sensor]['qartod'][test]['depth_range'])) == 1:
                    climatology_settings = {'fspan': c['streams'][sensor]['qartod'][test]['depth_range'],
                                            'vspan': None, 'zspan': None}
                    if 'suspect_span' in c['streams'][sensor]['qartod'][test].keys():
                        climatology_settings['vspan'] = c['streams'][sensor]['qartod'][test]['suspect_span']
                    if 'fail_span' in c['streams'][sensor]['qartod'][test].keys():
                        climatology_settings['fspan'] = c['streams'][sensor]['qartod'][test]['fail_span']
                    climatology_config = ClimatologyConfig()
                    climatology_config.add(**climatology_settings)
                    flag_vals = qartod.climatology_test(config=climatology_config,
                                                        inp=data,
                                                        tinp=times,
                                                        zinp=ds.depth.values)
                else: # if different thresholds for multiple depth ranges, loop through each
                    flag_vals = 2 * np.ones(np.shape(data))
                    for z_int in len(c['streams'][sensor]['qartod'][test]['depth_range']):
                        climatology_settings = {'fspan': c['streams'][sensor]['qartod'][test]['depth_range'][z_int],
                                                'vspan': None, 'zspan': None}
                        if 'suspect_span' in c['streams'][sensor]['qartod'][test].keys():
                            climatology_settings['vspan'] = c['streams'][sensor]['qartod'][test]['suspect_span'][z_int]
                        if 'fail_span' in c['streams'][sensor]['qartod'][test].keys():
                            climatology_settings['fspan'] = c['streams'][sensor]['qartod'][test]['fail_span'][z_int]
                        climatology_config = ClimatologyConfig()
                        climatology_config.add(**climatology_settings)
                        z_ind = np.logical_and(ds.depth.values > c['streams'][sensor]['qartod'][test]['depth_range'][z_int][0],
                                               ds.depth.values <= c['streams'][sensor]['qartod'][test]['depth_range'][z_int][1])
                        flag_vals[z_ind] = qartod.climatology_test(config=climatology_config,
                                                            inp=data[z_ind],
                                                            tinp=times[z_ind],
                                                            zinp=ds.depth.values[z_ind])
            # write to nc with attributes
        elif test == 'spike_test':
            # spike test
            spike_settings = {'suspect_threshold': None, 'fail_threshold': None}
            # convert original threshold from units/s to units/average-timestep
            if 'suspect_threshold' in c['streams'][sensor]['qartod'][test].keys():
                spike_settings['suspect_threshold'] = c['streams'][sensor]['qartod'][test]['suspect_threshold'] * np.nanmedian(tdiff)
            if 'fail_threshold' in c['streams'][sensor]['qartod'][test].keys():
                spike_settings['fail_threshold'] = c['streams'][sensor]['qartod'][test]['fail_threshold'] * np.nanmedian(tdiff)
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
                                                          **c['streams'][sensor]['qartod'][test])
            # write to nc with attributes