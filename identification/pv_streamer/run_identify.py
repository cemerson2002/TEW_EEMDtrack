# Usage Example

from feature.pipeline import FeaturePipeline
from feature.identification._proto_gen import pv_streamer_pb2
from feature.identification.pv_streamer import PVIdentification, PVWernliSprenger2007
from feature.identification.pv_streamer.plotting import generate_2d_plots
import os

pipeline = FeaturePipeline(pv_streamer_pb2, processing_mode='3d')

# init PVIdentification strategy, can take different parameters
# specify unit as 'pv' or 'pvu' depending on data unit
i_strat = PVIdentification(unit='pvu') # ,out_type='ll' , mode_2d_layer=330, theta_range=(300, 380), extract_containing_layer=330)

# or strategy by Wernli and Sprenger (2007). Note: Fortran required and manual adaptation of scripts needed, as well as their source code.
# change contour.f level too, also if data in PV/PVU adapt param.
# also pretty restrictive filewise.
# i_strat = PVWernliSprenger2007(unit='pv')

# t_strat = OverlapTracking()
# t_strat = EmptyTr()

pipeline.set_identification_strategy(i_strat)
pipeline.set_tracking_strategy(None)

# set data path
# data_path = os.path.join(os.path.expanduser("~"), 'w2w/workshops/EMS_2022/feature/data/pv/4m_12d_pvavg_200_500_1998.nc')
data_path = os.path.join(os.path.expanduser("~"), 'w2w/workshops/EMS_2022/feature/data/pv/th2016-09-07T00.nc')

pipeline.set_data_path(data_path)

# execute pipeline
pipeline.execute()

if not pipeline.is_data_3d():
    generate_2d_plots(pipeline.get_data())

out_netcdf_path = data_path + '_streamers.nc'
out_json_path = data_path + '_streamers_desc.json'
pipeline.save_result(description_type='json', description_path=out_json_path, dataset_path=out_netcdf_path,) #  save_proj=



