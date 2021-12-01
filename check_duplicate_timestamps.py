#!/usr/bin/env python

import os
import logging
import argparse
import sys
import pytz
from dateutil import parser
import glob
import numpy as np
import xarray as xr


def main(args):
#def main(deployments, mode, cdm_data_type, loglevel, dataset_type):
    """
    Check two consecutive .nc files for duplicated timestamps and rename files that are full duplicates of all or part
    of another file.
    """
    status = 0

    # Set up the logger
    log_level = getattr(logging, args.loglevel.upper())
    #log_level = getattr(logging, loglevel.upper())
    log_format = '%(asctime)s%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
    logging.basicConfig(format=log_format, level=log_level)

    cdm_data_type = args.cdm_data_type
    mode = args.mode
    dataset_type = args.level

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

    for deployment in args.deployments:
    #for deployment in [deployments]:

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

        # List the netcdf files in queue
        ncfiles = sorted(glob.glob(os.path.join(data_path, 'queue', '*.nc')))

        # Iterate through files and find duplicated timestamps
        duplicates = 0
        for i, f in enumerate(ncfiles):
            try:
                ds = xr.open_dataset(f)
            except OSError as e:
                logging.error('Error reading file {:s} ({:})'.format(ncfiles[i], e))
                status = 1
                continue

            # find the next file and compare timestamps
            try:
                f2 = ncfiles[i + 1]
                ds2 = xr.open_dataset(f2)
            except OSError as e:
                logging.error('Error reading file {:s} ({:})'.format(ncfiles[i + 1], e))
                status = 1
                continue
            except IndexError:
                continue

            # find the unique timestamps between the two datasets
            unique_timestamps = list(set(ds.time.values).symmetric_difference(set(ds2.time.values)))

            # find the unique timestamps in each dataset
            check_ds = [t for t in ds.time.values if t in unique_timestamps]
            check_ds2 = [t for t in ds2.time.values if t in unique_timestamps]

            # if the unique timestamps aren't found in either dataset (i.e. timestamps are exactly the same)
            # rename the second dataset
            if np.logical_and(len(check_ds) == 0, len(check_ds2) == 0):
                os.rename(f2, f'{f2}.duplicate')
                logging.info('Duplicated timestamps found in file: {:s}'.format(f2))
                duplicates += 1
            # if the unique timestamps aren't found in the second dataset, rename it
            elif np.logical_and(len(check_ds) > 0, len(check_ds2) == 0):
                os.rename(f2, f'{f2}.duplicate')
                logging.info('Duplicated timestamps found in file: {:s}'.format(f2))
                duplicates += 1
            # if the unique timestamps aren't found in the first dataset, rename it
            elif np.logical_and(len(check_ds) == 0, len(check_ds2) > 0):
                try:
                    os.rename(f, f'{f}.duplicate')
                    logging.info('Duplicated timestamps found in file: {:s}'.format(f))
                    duplicates += 1
                except FileNotFoundError:  # file has already been identified as a duplicate
                    continue
            else:
                continue

    logging.info(' {:} duplicated files found (of {:} total files)'.format(duplicates, len(ncfiles)))
    return status


if __name__ == '__main__':
    # deploy = 'maracoos_02-20210716T1814'  # maracoos_02-20210716T1814 ru34-20200729T1430 ru33-20201014T1746 ru33-20200715T1558  ru32-20190102T1317 ru30-20210503T1929
    # mode = 'delayed'
    # d = 'profile'
    # ll = 'info'
    # level = 'sci'
    # main(deploy, mode, d, ll, level)
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('deployments',
                            nargs='+',
                            help='Glider deployment name(s) formatted as glider-YYYYmmddTHHMM')

    arg_parser.add_argument('-m', '--mode',
                            help='Deployment dataset status <Default=rt>',
                            choices=['rt', 'delayed'],
                            default='rt')

    arg_parser.add_argument('--level',
                            choices=['raw', 'sci', 'ngdac'],
                            default='sci',
                            help='Dataset type')

    arg_parser.add_argument('-d', '--cdm_data_type',
                            help='Dataset type <default=profile>',
                            choices=['trajectory', 'profile'],
                            default='profile')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level <Default=warning>',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
