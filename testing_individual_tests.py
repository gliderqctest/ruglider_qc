import inspect
import numpy as np
import xarray as xr
import pandas as pd
from ioos_qc.config import Config
from ioos_qc import qartod


# read in file and profile time
f = './test_files/maracoos_02-20210716T1814/maracoos_02_20210716T190208Z_dbd.nc'
ds = xr.open_dataset(f)
profile_time = pd.to_datetime(ds.profile_time.values)
times = ds.time.values

# start with global config as default
config_file = './config/global_configs.yml'
c_global = Config(config_file)

c = c_global.config['contexts'][0]

# pull in regional config
config_file = './config/mab_configs.yml'
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
    data = ds[sensor].values
    non_nan_ind = np.invert(np.isnan(data))
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
            flag_vals = 2 * np.ones(np.shape(data))
            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING
            flag_vals[non_nan_ind] = qartod.spike_test(inp=data[non_nan_ind],
                                                          **c['streams'][sensor]['qartod'][test])
            # write to nc with attributes
        elif test == 'rate_of_change_test':
            # rate of change test
            flag_vals = 2 * np.ones(np.shape(data))
            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING
            flag_vals[non_nan_ind] = qartod.rate_of_change_test(inp=data[non_nan_ind],
                                                          tinp=times[non_nan_ind],
                                                          **c['streams'][sensor]['qartod'][test])
            # write to nc with attributes