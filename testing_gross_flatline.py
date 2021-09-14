import inspect
import numpy as np
import xarray as xr
from ioos_qc.config import Config
from ioos_qc.streams import XarrayStream
from ioos_qc.results import collect_results


def find_config(dataset, sensor_name):
    model = dataset[sensor_name].make_model
    if sensor_name == 'instrument_ctd':
        config_filename = f'./config/{model.split(" ")[0].lower()}_ctd_gross_flatline.yaml'
    elif sensor_name == 'instrument_optode':
        config_filename = f'./config/optode{model.split(" ")[-1].lower()}_gross_flatline.yaml'
    else:
        config_filename = None

    return config_filename


f = './test_files/maracoos_02-20210716T1814/maracoos_02_20210716T190208Z_dbd.nc'
ds = xr.open_dataset(f)
sensors = [x for x in list(ds.data_vars) if 'instrument_' in x]
for sensor in sensors:
    config_file = find_config(ds, sensor)
    if config_file:
        c = Config(config_file)
        print(c.config)
        xs = XarrayStream(ds, time='time', lat='latitude', lon='longitude')
        qc_results = xs.run(c)
        collected_list = collect_results(qc_results, how='list')

        for cl in collected_list:
            varname = cl.stream_id
            variable_data = cl.data
            column_name = f'{varname}_{cl.package}_{cl.test}'
            flag_results = cl.results.data
            standard_name = getattr(cl.function, 'standard_name', 'quality_flag')
            long_name = getattr(cl.function, 'long_name', 'Quality Flag')
            flags = getattr(inspect.getmodule(cl.function), 'FLAGS')
            varflagnames = [d for d in flags.__dict__ if not d.startswith('__')]
            varflagvalues = [getattr(flags, d) for d in varflagnames]
            thresholds = c.config[varname]['qartod'][cl.test]

            # Set QC variable attributes  TODO: modify format for flag_config attribute?
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
            # Add QC variable to dataset - should the QC values be floats? right now they're uint8
            da = xr.DataArray(flag_results, coords=ds[varname].coords, dims=ds[varname].dims, name=column_name,
                              attrs=attrs)
            ds[column_name] = da

ds.to_netcdf(f'{f.split(".nc")[0]}_gross_flatline_qc.nc')
