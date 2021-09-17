import inspect
import numpy as np
import xarray as xr
from ioos_qc.config import Config
from ioos_qc.streams import XarrayStream
from ioos_qc.results import collect_results

baseDir='/Users/nazzaro/Documents/GitHub/ruglider_qc/'
f = baseDir+'/test_files/maracoos_02-20210716T1814/maracoos_02_20210716T190208Z_dbd.nc'
ds = xr.open_dataset(f)

config_file=baseDir+'config/mara02test_all.yml'

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
    #thresholds = c.config[varname]['qartod'][cl.test]