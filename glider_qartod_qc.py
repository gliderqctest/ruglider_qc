#!/usr/bin/env python

import os
import logging
import argparse
import sys
import pytz
from dateutil import parser
import glob
import inspect
import numpy as np
import xarray as xr
from ioos_qc.config import Config
from ioos_qc.streams import XarrayStream
from ioos_qc.results import collect_results


def define_config(sensor_name, model_name):
    if sensor_name == 'instrument_ctd':
        config_filename = f'{model_name.split(" ")[0].lower()}_ctd_gross_flatline.yaml'
    elif sensor_name == 'instrument_optode':
        config_filename = f'optode{model_name.split(" ")[-1].lower()}_gross_flatline.yaml'
    else:
        config_filename = 'no_filename_specified'

    return config_filename


def run_gross_flatline(ds, qc_config_root):
    """
    Run ioos_qc gross range and flat line QARTOD tests
    """
    # Set the path for the gross range and flat line configuration files
    qc_config_root = os.path.join(qc_config_root, 'gross_flatline')

    # List the sensors in the netcdf file
    sensors = [x for x in list(ds.data_vars) if 'instrument_' in x]

    for sensor in sensors:
        # Define the sensor make/model
        try:
            model = ds[sensor].make_model
        except AttributeError:
            logging.error('Sensor make_model not specified {:s}'.format(sensor))

        # Build the configuration filename based on the sensor, make and model
        qc_config_filename = define_config(sensor, model)

        qc_config_file = os.path.join(qc_config_root, qc_config_filename)

        if not os.path.isfile(qc_config_file):
            logging.warning('Missing QC configuration file: {:s} {:s}'.format(sensor, qc_config_file))
            continue
        logging.info('QC configuration file: {:s}'.format(qc_config_file))

        # Run ioos_qc tests based on the QC configuration file
        c = Config(qc_config_file)
        xs = XarrayStream(ds, time='time', lat='latitude', lon='longitude')
        qc_results = xs.run(c)
        collected_list = collect_results(qc_results, how='list')

        # Parse each QC result
        for cl in collected_list:
            varname = cl.stream_id
            qc_varname = f'{varname}_{cl.package}_{cl.test}'
            logging.info('Parsing QC results: {:s}'.format(qc_varname))
            flag_results = cl.results.data

            # Getting QC variable attributes
            standard_name = getattr(cl.function, 'standard_name', 'quality_flag')
            long_name = getattr(cl.function, 'long_name', 'Quality Flag')
            flags = getattr(inspect.getmodule(cl.function), 'FLAGS')
            varflagnames = [d for d in flags.__dict__ if not d.startswith('__')]
            varflagvalues = [getattr(flags, d) for d in varflagnames]
            thresholds = c.config[varname]['qartod'][cl.test]

            # Defining QC variable attributes
            attrs = {
                'standard_name': standard_name,
                'long_name': long_name,
                'flag_values': np.byte(varflagvalues),
                'flag_meanings': ' '.join(varflagnames),
                'flag_configurations': str(thresholds),
                'valid_min': np.byte(min(varflagvalues)),
                'valid_max': np.byte(max(varflagvalues)),
                'ioos_qc_module': cl.package,
                'ioos_qc_test': cl.test,
                'ioos_qc_target': cl.stream_id,
            }

            # Add QC variable to the original dataset
            da = xr.DataArray(flag_results, coords=ds[varname].coords, dims=ds[varname].dims,
                              name=qc_varname,
                              attrs=attrs)
            ds[qc_varname] = da


def main(args):
# def main(deployments, mode, cdm_data_type, loglevel, dataset_type):
    """
    Run ioos_qc QARTOD tests on processed slocum glider netcdf files,
    and append the results to the original netcdf file.
    Note: This currently works for sci-profile netcdf files only
    """
    status = 0

    # Set up the logger
    log_level = getattr(logging, args.loglevel.upper())
    # log_level = getattr(logging, loglevel.upper())
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
    # for deployment in [deployments]:

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
        ncfiles = glob.glob(os.path.join(data_path, 'queue', '*.nc'))

        # Iterate through files and apply QC
        for f in ncfiles:
            nc_filename = f.split('/')[-1]
            try:
                ds = xr.open_dataset(f)
            except OSError as e:
                logging.error('Error reading file {:s} ({:})'.format(f, e))
                status = 1
                continue

            # Set the qc configuration path
            qc_config_root = os.path.join(data_home, 'qc', 'config')
            if not os.path.isdir(qc_config_root):
                logging.warning('Invalid QC config root: {:s}'.format(qc_config_root))
                return 1

            run_gross_flatline(ds, qc_config_root)

            # TODO add other tests

            # Save the resulting netcdf file with QC variables
            output_netcdf = os.path.join(data_path, nc_filename)
            ds.to_netcdf(output_netcdf)

    return status


if __name__ == '__main__':
    # deploy = 'maracoos_02-20210716T1814'
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
