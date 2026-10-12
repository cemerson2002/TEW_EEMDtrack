from feature.pipeline import FeaturePipeline
from feature.identification.threshold import DoubleThresholdIdentification,SingleThresholdIdentification
from feature.tracking.overlap_tracking import OverlapTracking
from feature.identification._proto_gen import threshold_pb2
from os.path import expanduser

# set the pb_reference to the compiled pb2.py file (see proto_gen directory)
pipeline = FeaturePipeline(threshold_pb2, processing_mode='3d', workers=1)

# change this to an identification strategy that actually does something: existing one or implement your own

path = '/project/meteo/w2w/workshops/EMS_2022/feature/data/storm/zeynep.nc'
pipeline.set_data_path(path)

i_strat = DoubleThresholdIdentification("msl", 97500, 97000, "<" ,processing_mode='2d') # set the Identification strategy
t_strat = OverlapTracking(field_name="feature_field") # set the tracking strategy

pipeline.set_identification_strategy(i_strat)
pipeline.set_tracking_strategy(t_strat) # or None as argument if no tracking

# execute pipeline
pipeline.execute()

out_json_path = path[:-3] + '_thrdesc.json'
out_nc_path = path[:-3] + '_thr.nc'
out_graph_path = path[:-3] + '_thrgraph.json'
print(out_json_path)
od = pipeline.get_feature_desc() # identified features, and graph-structure if tracking done

pipeline.save_result(description_type='json', description_path=out_json_path, dataset_path=out_nc_path) # dataset_path=...
