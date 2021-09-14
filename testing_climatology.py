import numpy as np
import xarray as xr
from ioos_qc.qartod import ClimatologyConfig, climatology_test

# test on conductivity

f = './test_files/maracoos_02-20210716T1814/maracoos_02_20210716T190208Z_dbd.nc'
varnames = ['conductivity']
varflagvalues = [1, 2, 3, 4, 9]
ds = xr.open_dataset(f)

for varname in varnames:
    c = ClimatologyConfig()
    c.add(tspan=['2021-06-01', '2021-09-01'],
          vspan=[3.4, 5],
          zspan=[0, 1000])
    results = climatology_test(config=c,
                               inp=ds[varname].values,
                               tinp=ds.time.values,
                               zinp=ds.depth.values)

    # Set QC variable attributes  TODO: add spans from config file for suspect and fail
    attrs = {
        'standard_name': 'climatology_test_quality_flag',
        'long_name': 'Climatology Test Quality Flag',
        'flag_values': np.byte(varflagvalues),
        'flag_meanings': 'GOOD UNKNOWN SUSPECT FAIL MISSING',
        'flag_configurations': 'add config info',
        'valid_min': np.byte(min(varflagvalues)),
        'valid_max': np.byte(max(varflagvalues)),
        'ioos_qc_module': 'qartod',
        'ioos_qc_test': 'climatology_test',
        'ioos_qc_target': varname,
    }
    qcname = f'{varname}_qartod_climatology_test'
    da = xr.DataArray(results.data, coords=ds[varname].coords, dims=ds[varname].dims, name=qcname,
                      attrs=attrs)
    ds[qcname] = da

ds.to_netcdf(f'{f.split(".nc")[0]}_climatology_qc.nc')
