# Usage Example

from feature.pipeline import FeaturePipeline
from feature.identification.storm import StormIdentification
from feature.tracking.overlap_tracking import OverlapTracking
from feature.tracking.proximity_tracking import ProximityTracking
from feature.identification._proto_gen import storm_pb2
from os.path import expanduser
from feature.identification.storm.util import plot_timestep, plot_track

import os
from datetime import timedelta

from feature.util.graph import DataGraph

# set the pb_reference to the compiled pb2.py file (see proto_gen directory)
pipeline = FeaturePipeline(storm_pb2, processing_mode='2d')

i_strat = StormIdentification() # set the Identification strategy

# Tracking, either proximity of low pressure centers, or overlap of low pressure areas
"""
t_strat = ProximityTracking('min_pressure_pos.lon', # where lon and lat values are in proto
                            'min_pressure_pos.lat',
                            speed_range_ms=[0,40], min_duration=timedelta(hours=12)) # proximity tracking for the storm centers
"""
t_strat = OverlapTracking(field_name='storm_areas', min_duration=timedelta(hours=12)) # overlap tracking of the storm areas


pipeline.set_identification_strategy(i_strat)
pipeline.set_tracking_strategy(t_strat) # or None as argument if no tracking

# data is expected in Pa, not hPa here. Adjust in identification strategy accordingly.
path = '/project/meteo/w2w/workshops/EMS_2022/feature/data/storm/zeynep.nc'

plot_dir = os.path.expanduser('~/plots/')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

desc_dir = os.path.expanduser('~/feature_desc/')
if not os.path.exists(desc_dir):
    os.makedirs(desc_dir)

pipeline.set_data_path(path)

# execute pipeline
pipeline.execute()

od = pipeline.get_feature_desc()
reanalysis_set = od.sets[0]

for ts in reanalysis_set.timesteps:
    ds_t = pipeline.get_data().sel(time=ts.time)
    plot_timestep(ds_t, plot_dir + str(ts.time)[:13] + '.png')

for track_id, track in enumerate(reanalysis_set.tracks):
    plot_track(track, plot_dir + 'track' + str(track_id) + '.png', pipeline.get_data().storm_areas)


out_json_path = desc_dir + 'desc.json'
out_ds_path = desc_dir + 'desc.nc'

pipeline.save_result(description_type='json', description_path=out_json_path, dataset_path=out_ds_path)


