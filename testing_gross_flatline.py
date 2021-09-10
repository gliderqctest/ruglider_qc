import inspect
import numpy as np
import xarray as xr
from ioos_qc.config import Config
from ioos_qc.streams import XarrayStream
from ioos_qc.results import collect_results

f = './test_files/maracoos_02-20210716T1814/maracoos_02_20210716T190208Z_dbd.nc'
ds = xr.open_dataset(f)

c = Config('./config/seabird_ctd_gross_flatline.yaml')
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
