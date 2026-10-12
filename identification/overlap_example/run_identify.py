# Usage Example

from feature.pipeline import FeaturePipeline
from feature.identification.overlap_example import OverlapIdentificationExample
from feature.tracking.template_feature_compare import TrackingCompareTemplate
from feature.tracking.overlap_tracking import OverlapTracking
from feature.identification._proto_gen import overlap_example_pb2
from os.path import expanduser
from feature.util.graph import DataGraph
from datetime import timedelta

# set the pb_reference to the compiled pb2.py file (see proto_gen directory)
pipeline = FeaturePipeline(overlap_example_pb2, processing_mode='3d')

# change this to an identification strategy that actually does something: existing one or implement your own
i_strat = OverlapIdentificationExample()  # set the Identification strategy
t_strat = OverlapTracking(field_name='identified_areas', min_duration=timedelta(hours=12))  # set the tracking strategy

pipeline.set_identification_strategy(i_strat)
pipeline.set_tracking_strategy(t_strat)  # or None as argument if no tracking

path = expanduser("~") + '/w2w/C3/fischer/phd_all/data/tracked/feature/enstools_feature_test.nc'  # ERA5/vietnam/t0_glob.nc' #  set data path(s) here
out_data_path = path[:-3] + '_id.nc'
out_graph_path = path[:-3] + '_graph.json'
pipeline.set_data_path(path)

# execute pipeline
pipeline.execute()

od = pipeline.get_feature_desc()  # identified features, and graph-structure if tracking done


pipeline.save_result(description_type='json', description_path=out_graph_path,
                     dataset_path=out_data_path)  # dataset_path=...
