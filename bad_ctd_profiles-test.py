#!/usr/bin/env python

import os
import logging
import argparse
import sys
import pytz
from dateutil import parser
from datetime import timedelta
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from shapely.geometry import Polygon
from shapely.ops import polygonize
from ioos_qc import qartod
np.set_printoptions(suppress=True)


def apply_qartod_qc(dataset):
    # make a copy of conductivity and apply QARTOD QC flags
    cond_copy = dataset.conductivity.copy()
    for qv in [x for x in dataset.data_vars if 'conductivity_qartod' in x]:
        qv_idx = np.where(np.logical_or(dataset[qv].values == 3, dataset[qv].values == 4))[0]
        cond_copy[qv_idx] = np.nan
    return cond_copy


def initialize_flags(dataset):
    # start with flag values UNKNOWN (2)
    flags = 2 * np.ones(np.shape(dataset.conductivity.values))

    # identify where not nan
    non_nan_ind = np.invert(np.isnan(dataset.conductivity.values))
    # get locations of non-nans
    non_nan_i = np.where(non_nan_ind)[0]

    # flag the missing values
    flags[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING

    # identify where pressure is not nan
    press_non_nan_ind = np.where(np.invert(np.isnan(dataset.pressure.values)))[0]

    return non_nan_i, press_non_nan_ind, flags


def save_ds(dataset, flag_array, attributes, save_file):
    # Add QC variable to the original dataset
    da = xr.DataArray(flag_array, coords=dataset['conductivity'].coords, dims=dataset['conductivity'].dims,
                      name='ctd_profile_test', attrs=attributes)
    dataset['ctd_profile_test'] = da

    # Save the resulting netcdf file with QC variable
    dataset.to_netcdf(save_file)


def set_qc_attrs(test, sensor, thresholds=None):
    """
    Define the QC variable attributes
    :param test: QC test
    :param sensor: sensor variable name (e.g. conductivity)
    :param thresholds: flag thresholds from QC configuration files
    """
    thresholds = thresholds or None

    flag_meanings = 'GOOD UNKNOWN SUSPECT FAIL MISSING'
    flag_values = [1, 2, 3, 4, 9]
    standard_name = f'{test}_quality_flag'  # 'flat_line_test_quality_flag'
    if test == 'ctd_profile_test':
        long_name = 'CTD Profile Test Quality Flag'
    else:
        long_name = f'{" ".join([x.capitalize() for x in test.split("_")])} Quality Flag'

    # Defining gross/flatline QC variable attributes
    attrs = {
        'standard_name': standard_name,
        'long_name': long_name,
        'flag_values': np.byte(flag_values),
        'flag_meanings': flag_meanings,
        'valid_min': np.byte(min(flag_values)),
        'valid_max': np.byte(max(flag_values)),
        'qc_target': sensor,
    }

    if thresholds:
        attrs['flag_configurations'] = str(thresholds)

    return attrs


# def main(args):
def main(deployments, mode, cdm_data_type, loglevel, dataset_type):
    """
    Run ioos_qc QARTOD tests on processed slocum glider netcdf files,
    and append the results to the original netcdf file.
    Note: This currently works for sci-profile netcdf files only
    """
    status = 0

    # Set up the logger
    # log_level = getattr(logging, args.loglevel.upper())
    log_level = getattr(logging, loglevel.upper())
    log_format = '%(asctime)s%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    # cdm_data_type = args.cdm_data_type
    # mode = args.mode
    # dataset_type = args.level

    # Find the glider deployments root directory
    data_home = os.getenv('GLIDER_DATA_HOME')
    if not data_home:
        logging.error('GLIDER_DATA_HOME not set')
        return 1
    elif not os.path.isdir(data_home):
        logging.error('Invalid GLIDER_DATA_HOME: {:s}'.format(data_home))
        return 1

    deployments_root = os.path.join(data_home, 'deployments')
    if not os.path.isdir(deployments_root):
        logging.warning('Invalid deployments root: {:s}'.format(deployments_root))
        return 1
    logging.info('Deployments root: {:s}'.format(deployments_root))

    # for deployment in args.deployments:
    for deployment in [deployments]:

        logging.info('Checking deployment {:s}'.format(deployment))

        try:
            (glider, trajectory) = deployment.split('-')
        except ValueError as e:
            logging.error('Error parsing invalid deployment name {:s}: {:}'.format(deployment, e))
            status = 1
            continue

        try:
            trajectory_dt = parser.parse(trajectory).replace(tzinfo=pytz.UTC)
        except ValueError as e:
            logging.error('Error parsing trajectory date {:s}: {:}'.format(trajectory, e))
            status = 1
            continue

        trajectory = '{:s}-{:s}'.format(glider, trajectory_dt.strftime('%Y%m%dT%H%M'))
        deployment_name = os.path.join('{:0.0f}'.format(trajectory_dt.year), trajectory)

        # Create fully-qualified path to the deployment location
        deployment_location = os.path.join(data_home, 'deployments', deployment_name)
        logging.info('Deployment location: {:s}'.format(deployment_location))
        if not os.path.isdir(deployment_location):
            logging.warning('Deployment location does not exist: {:s}'.format(deployment_location))
            status = 1
            continue

        # Set the deployment netcdf data path
        data_path = os.path.join(deployment_location, 'data', 'out', 'nc',
                                 '{:s}-{:s}/{:s}'.format(dataset_type, cdm_data_type, mode))

        if not os.path.isdir(data_path):
            logging.warning('{:s} data directory not found: {:s}'.format(trajectory, data_path))
            status = 1
            continue

        # List the netcdf files
        ncfiles = sorted(glob.glob(os.path.join(data_path, 'queue', '*.nc')))

        # Iterate through files
        skip = 0
        for i, f in enumerate(ncfiles):
            i += skip
            try:
                with xr.open_dataset(ncfiles[i]) as ds:
                    ds = ds.load()
                    print(f'\nds1: {ncfiles[i]}')
            except OSError as e:
                logging.error('Error reading file {:s} ({:})'.format(ncfiles[i], e))
                status = 1
                continue
            except IndexError:
                continue

            try:
                ds.conductivity
            except AttributeError:
                logging.error('conductivity variable not found in file {:s})'.format(ncfiles[i]))
                status = 1
                continue

            qc_varname = 'ctd_profile_test'
            attrs = set_qc_attrs(qc_varname, 'conductivity')
            data_idx, pressure_idx, flag_vals = initialize_flags(ds)

            if len(data_idx) == 0:
                logging.error('conductivity data not found in file {:s})'.format(ncfiles[i]))
                status = 1
                continue

            # determine if profile is up or down
            if ds.pressure.values[pressure_idx][0] > ds.pressure.values[pressure_idx][-1]:
                # if profile is up, test can't be run because you need a down profile paired with an up profile
                # leave flag values as UNKNOWN (2), set the attributes and save the .nc file
                save_ds(ds, flag_vals, attrs, ncfiles[i])
            else:  # first profile is down, check the next file
                try:
                    f2 = ncfiles[i + 1]
                    with xr.open_dataset(f2) as ds2:
                        ds2 = ds2.load()
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f2, e))
                    status = 1
                    skip += 1
                except IndexError:
                    # if there are no more files, leave flag values on the first file as UNKNOWN (2)
                    # set the attributes and save the first .nc file
                    save_ds(ds, flag_vals, attrs, ncfiles[i])
                    continue

                print(f'ds2: {f2}')
                try:
                    ds2.conductivity
                except AttributeError:
                    logging.error('conductivity variable not found in file {:s})'.format(f2))
                    status = 1
                    # TODO should we be checking the next file? example ru30_20210510T015902Z_sbd.nc
                    # leave flag values on the first file as UNKNOWN (2), set the attributes and save the first .nc file
                    save_ds(ds, flag_vals, attrs, ncfiles[i])
                    continue

                data_idx2, pressure_idx2, flag_vals2 = initialize_flags(ds2)

                # determine if second profile is up or down
                if ds2.pressure.values[pressure_idx2][0] < ds2.pressure.values[pressure_idx2][-1]:
                    # if second profile is also down, test can't be run on the first file
                    # leave flag values on the first file as UNKNOWN (2), set the attributes and save the first .nc file
                    # but don't skip because this second file will now be the first file in the next loop
                    save_ds(ds, flag_vals, attrs, ncfiles[i])
                else:
                    # first profile is down and second profile is up
                    # determine if the end/start timestamps are < 5 minutes apart,
                    # indicating a paired yo (down-up profile pair)
                    if ds2.time.values[0] - ds.time.values[-1] < np.timedelta64(5, 'm'):

                        # make a copy of conductivity and apply QARTOD QC flags
                        conductivity_copy = apply_qartod_qc(ds)
                        conductivity_copy2 = apply_qartod_qc(ds2)

                        # both yos must have data remaining after QARTOD flags are applied,
                        # otherwise, test can't be run and leave the flag values as UNKNOWN (2)
                        if np.logical_and(np.sum(~np.isnan(conductivity_copy)) > 0, np.sum(~np.isnan(conductivity_copy2)) > 0):
                            # calculate the area between the two profiles
                            df = conductivity_copy.to_dataframe().merge(ds.pressure.to_dataframe(), on='time')
                            df2 = conductivity_copy2.to_dataframe().merge(ds2.pressure.to_dataframe(), on='time')
                            df = df.append(df2)
                            df = df.dropna(subset=['pressure', 'conductivity'])
                            polygon_points = df.values.tolist()
                            polygon_points.append(polygon_points[0])
                            polygon = Polygon(polygon_points)
                            polygon_lines = polygon.exterior
                            polygon_crossovers = polygon_lines.intersection(polygon_lines)
                            polygons = polygonize(polygon_crossovers)
                            valid_polygons = shapely.geometry.MultiPolygon(polygons)

                            # normalize area between the profiles and data range to pressure range
                            pressure_range = (np.nanmax(df.pressure.values) - np.nanmin(df.pressure.values))
                            area = valid_polygons.area / pressure_range
                            data_range = (np.nanmax(df.conductivity.values) - np.nanmin(df.conductivity.values)) / pressure_range

                            # If the normalized area between the profiles is greater than an order of magnitude more
                            # than the normalized data range, flag both profiles as fail.
                            if area > data_range * 10:
                                flag = qartod.QartodFlags.FAIL
                            # If the normalized area between the profiles is greater than 5x more than the normalized
                            # data range, flag both profiles as suspect
                            elif area > data_range * 5:
                                flag = qartod.QartodFlags.SUSPECT
                            # Otherwise, both profiles are good
                            else:
                                flag = qartod.QartodFlags.GOOD
                            flag_vals[data_idx] = flag
                            flag_vals2[data_idx2] = flag

                            t0str = pd.to_datetime(np.nanmin(df.index.values)).strftime('%Y-%m-%dT%H:%M:%S')
                            tfstr = pd.to_datetime(np.nanmax(df.index.values)).strftime('%Y-%m-%dT%H:%M:%S')
                            fig, ax = plt.subplots(figsize=(8, 10))
                            ax.plot(df.conductivity.values, df.pressure.values, color='k')  # plot lines
                            ax.scatter(df.conductivity.values, df.pressure.values, color='k', s=30)  # plot points
                            ax.invert_yaxis()
                            ax.set_ylabel('Pressure (dbar)')
                            ax.set_xlabel('Conductivity')
                            ttl = '{} to {}\nNormalized Area = {}, Normalized Data Range = {}'.format(t0str, tfstr,
                                                                                                      np.round(area, 4),
                                                                                                      str(np.round(data_range, 4)))
                            ax.set_title(ttl)

                            # Iterate through unknown (2), suspect (3), and fail (4) flags
                            flag_defs = dict(unknown=dict(value=2, color='cyan'),
                                             suspect=dict(value=3, color='orange'),
                                             fail=dict(value=4, color='red'))

                            for fd, info in flag_defs.items():
                                idx = np.where(flag_vals == info['value'])
                                if len(idx[0]) > 0:
                                    ax.scatter(ds.conductivity.values[idx], ds.pressure.values[idx], color=info['color'],
                                               s=40, label=f'{qc_varname}-{fd}', zorder=10)
                                idx2 = np.where(flag_vals2 == info['value'])
                                if len(idx2[0]) > 0:
                                    ax.scatter(ds2.conductivity.values[idx2], ds2.pressure.values[idx2], color=info['color'],
                                               s=40, label=f'{qc_varname}-{fd}', zorder=10)

                            # add legend if necessary
                            handles, labels = plt.gca().get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            if len(handles) > 0:
                                ax.legend(by_label.values(), by_label.keys(), loc='best')

                            plt_name = f'{ncfiles[i].split("/")[-1].split(".nc")[0]}_{f2.split("/")[-1].split(".nc")[0]}_qc.png'
                            sfile = os.path.join(data_path, plt_name)
                            plt.savefig(sfile, dpi=300)
                            plt.close()

                        # save both .nc files with QC applied
                        save_ds(ds, flag_vals, attrs, ncfiles[i])
                        save_ds(ds2, flag_vals2, attrs, f2)
                        skip += 1

                    else:
                        # if timestamps are too far apart they're likely not from the same profile pair
                        # leave flag values as UNKNOWN (2), set the attributes and save the .nc files
                        save_ds(ds, flag_vals, attrs, ncfiles[i])
                        save_ds(ds2, flag_vals2, attrs, f2)
                        skip += 1

    return status


if __name__ == '__main__':
    deploy = 'ru30-20210503T1929'  # maracoos_02-20210716T1814 ru34-20200729T1430 ru33-20201014T1746 ru33-20200715T1558  ru32-20190102T1317  ru30-20210503T1929
    mode = 'rt'
    d = 'profile'
    ll = 'info'
    level = 'sci'
    main(deploy, mode, d, ll, level)
    # arg_parser = argparse.ArgumentParser(description=main.__doc__,
    #                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # arg_parser.add_argument('deployments',
    #                         nargs='+',
    #                         help='Glider deployment name(s) formatted as glider-YYYYmmddTHHMM')
    #
    # arg_parser.add_argument('-m', '--mode',
    #                         help='Deployment dataset status <Default=rt>',
    #                         choices=['rt', 'delayed'],
    #                         default='rt')
    #
    # arg_parser.add_argument('--level',
    #                         choices=['raw', 'sci', 'ngdac'],
    #                         default='sci',
    #                         help='Dataset type')
    #
    # arg_parser.add_argument('-d', '--cdm_data_type',
    #                         help='Dataset type <default=profile>',
    #                         choices=['trajectory', 'profile'],
    #                         default='profile')
    #
    # arg_parser.add_argument('-l', '--loglevel',
    #                         help='Verbosity level <Default=warning>',
    #                         type=str,
    #                         choices=['debug', 'info', 'warning', 'error', 'critical'],
    #                         default='info')
    #
    # parsed_args = arg_parser.parse_args()
    #
    # sys.exit(main(parsed_args))