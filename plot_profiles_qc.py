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
import ast
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


def main(args):
# def main(deployments, mode, cdm_data_type, loglevel, dataset_type):
    """
    plot profiles
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

        # Set the QC images save file directory
        save_path = os.path.join(deployment_location, 'qc', 'images',
                                 '{:s}-{:s}/{:s}'.format(dataset_type, cdm_data_type, mode))
        if not os.path.isdir(save_path):
            logging.warning('{:s} QC imagery directory not found: {:s}'.format(trajectory, save_path))
            status = 1
            continue

        # List the netcdf files to plot
        ncfiles = glob.glob(os.path.join(data_path, '*.nc'))

        # Iterate through files and plot profiles
        for f in ncfiles:
            nc_filename = f.split('/')[-1]
            try:
                ds = xr.open_dataset(f)
            except OSError as e:
                logging.error('Error reading file {:s} ({:})'.format(f, e))
                status = 1
                continue

            # Iterate through each QC variable to plot the data with the flags applied
            qcvarnames = [x for x in list(ds.data_vars) if '_qartod_' in x]
            for qv in qcvarnames:
                title = f'{ds[qv].long_name.split(" Test")[0]}: {ds[qv].flag_configurations}'

                v = qv.split('_qartod')[0]
                if v not in ['oxygen_concentration', 'oxygen_saturation', 'pressure']:  # skip for now
                    fig, ax = plt.subplots(figsize=(8, 10))

                    # Plot data
                    xdata = ds[v].values
                    ydata = ds['depth'].values
                    xmask = ~np.isnan(xdata)  # get rid of nans so the lines are continuous
                    ax.plot(xdata[xmask], ydata[xmask], color='k')  # plot lines
                    ax.scatter(xdata[xmask], ydata[xmask], color='k', s=30)  # plot points
                    ylims = ax.get_ylim()
                    xlims = ax.get_xlim()

                    # Get the flag thresholds
                    flag_config = ast.literal_eval(ds[qv].flag_configurations)
                    suspect_key = [x for x in flag_config.keys() if 'suspect' in x]
                    suspect_threshold = flag_config[suspect_key[0]]
                    fail_key = [x for x in flag_config.keys() if 'fail' in x]
                    fail_threshold = flag_config[fail_key[0]]

                    # Iterate through unknown (2) suspect (3) and fail (4) flags
                    flag_defs = dict(unknown=dict(value=2, color='cyan'),
                                     suspect=dict(value=3, color='orange'),
                                     fail=dict(value=4, color='red'))

                    for fd, info in flag_defs.items():
                        flag = ds[qv].values
                        idx = np.where(flag == info['value'])
                        if len(idx[0]) > 0:
                            ax.scatter(xdata[idx], ydata[idx], color=info['color'], s=40, label=f'{qv}-{fd}',
                                       zorder=2)
                    if 'gross' in qv:
                        # Plot horizontal lines for the suspect and fail thresholds for gross_range
                        ax.vlines(suspect_threshold, ylims[0], ylims[1], colors='orange')
                        ax.vlines(fail_threshold, ylims[0], ylims[1], colors='red')

                    # set the x limits to the limits of the data
                    ax.set_xlim(xlims)

                    # add legend if necessary
                    handles, labels = plt.gca().get_legend_handles_labels()
                    if len(handles) > 0:
                        ax.legend()

                    ax.set_ylim(ylims)
                    ax.invert_yaxis()
                    ax.set_ylabel('Depth (m)')
                    ax.set_xlabel(f'{v} ({ds[v].units})')
                    ax.set_title(title)

                    sfile = os.path.join(save_path, f'{qv}_{nc_filename.split(".nc")[0]}_qc.png')
                    plt.savefig(sfile, dpi=300)
                    plt.close()

    return status


if __name__ == '__main__':
    # deploy = 'maracoos_02-20210716T1814'  # maracoos_02-20210716T1814 ru34-20200729T1430
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
