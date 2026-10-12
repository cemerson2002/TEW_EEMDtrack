# Usage Example

from feature.pipeline import FeaturePipeline
from feature.identification.template import IdentificationTemplate
from feature.tracking.template_feature_compare import TrackingCompareTemplate
from feature.tracking.template import TrackingTemplate
from feature.identification._proto_gen import template_pb2
from os.path import expanduser
from feature.util.graph import DataGraph

# set the pb_reference to the compiled pb2.py file (see proto_gen directory)
pipeline = FeaturePipeline(template_pb2, processing_mode='2d', workers=16)

# change this to an identification strategy that actually does something: existing one or implement your own
i_strat = IdentificationTemplate(some_parameter='foo') # set the Identification strategy
t_strat = TrackingCompareTemplate() # set the tracking strategy
t_strat.set_threads(16)

pipeline.set_identification_strategy(i_strat)
pipeline.set_tracking_strategy(t_strat, generate_tracks=True) # or None as argument if no tracking
path = expanduser("~") + '/w2w/C3/fischer/phd_all/data/tracked/feature/enstools_feature_test.nc' #  ERA5/vietnam/t0_glob.nc' #  set data path(s) here
pipeline.set_data_path(path)

# execute pipeline
pipeline.execute()

od = pipeline.get_feature_desc()

for trackable_set in od.sets:
    # can perform operations on tracks and nodes...
    graph = DataGraph.from_pb2(trackable_set, t_strat)
    
    track = graph.tracks[0]

    # get earliest and latest nodes of that track
    earliest_nodes = track.get_earliest_nodes()
    print("First nodes:")
    print([en.time + " ID " + str(en.feature.id) for en in earliest_nodes])
    latest_nodes = track.get_latest_nodes()
    print("Latest nodes:")
    print([ln.time + " ID " + str(ln.feature.id) for ln in latest_nodes])

    if len(latest_nodes):
        parents = track.get_parents(latest_nodes[0])
        print("Parents of a latest node:")
        print([p.time + " ID " + str(p.feature.id) for p in parents])
        children = track.get_children(latest_nodes[0])
        print("Children of latest nodes:")
        print([c.time + " ID " + str(c.feature.id) for c in children])


    for track_id, track in enumerate(graph.tracks):
        print("Track " + str(track_id) + " has " + str(len(track.edges)) + " nodes.")


out_json_path = path[:-3] + '_desc.json'
out_graph_path = path[:-3] + '_graph.json'

pipeline.save_result(description_type='json', description_path=out_json_path) # dataset_path=...


