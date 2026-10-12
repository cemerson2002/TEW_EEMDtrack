#!/usr/bin/env python

# Usage Example
from feature.pipeline import FeaturePipeline
from feature.identification.aew_pv import AEWPVIdentification
from datetime import timedelta, datetime
from feature.identification._proto_gen import aew_pv_pb2
from os.path import expanduser, join
from feature.util.graph import DataGraph
import feature.identification.aew_pv.configuration as cfg
import os, sys, glob, shutil
from feature.util.data_utils import get_subset_by_description
import xarray as xr
xr.set_options(keep_attrs=True)
import numpy as np
import dask

pipeline = FeaturePipeline(aew_pv_pb2, processing_mode='3d', workers=16)
dask.config.set(**{'array.slicing.split_large_chunks': False})

# in_files_all_cv_data = cfg.cv_data_ex

if len(sys.argv) > 2 and sys.argv[1] == '-y':
    y = sys.argv[2]
    year_dir = cfg.in_data_era5 + y + "/"
    
    cfg.in_file = [year_dir + "pv" + y + ".nc", year_dir + "uv" + y + "_fft.nc", year_dir + "wts" + y + ".nc"]
    cfg.in_wts = year_dir + "wts" + y + ".json"
    
    # ell_mode = sys.argv[3]
    # cfg.ell_mode = ell_mode
    
    if len(sys.argv) > 3:
        vsplit_mode = sys.argv[3]
    else:
        vsplit_mode = None
    cfg.vsplit_mode = vsplit_mode
    
    """
    if ell_mode == 'pc':
        cfg.out_json_path = year_dir + "pv" + y + "_pc_desc.json"
        cfg.out_dataset_path = year_dir + "pv" + y + "_pc_desc.nc"
    """
    # all modes
    cfg.out_json_path = year_dir + (cfg.vsplit_mode + "_" if cfg.vsplit_mode is not None else "") + "pv" + y + "_desc.json"
    cfg.out_dataset_path = year_dir + (cfg.vsplit_mode + "_" if cfg.vsplit_mode is not None else "") + "pv" + y + "_desc.nc"
    
    cfg.start_date = None
    cfg.end_date = None
    
else:
    cfg.in_file = [cfg.in_data_pv, cfg.in_data_fft, cfg.in_data_wts]

pipeline.set_data_path(cfg.in_file)

# init AEWIdentification strategy, can take different parameters
i_strat = AEWPVIdentification(cfg)
# t_strat = SomeTracking()

pipeline.set_identification_strategy(i_strat)
# pipeline.set_tracking_strategy(t_strat)

# execute pipeline
pipeline.execute()

od = pipeline.get_feature_desc()

"""
all_tracks = []
for set_id, trackable_set in enumerate(od.sets):

    # generate graph out of tracked data
    # g = DataGraph(trackable_set, t_strat)
    # g.generate_tracks(apply_filter=True)
    # tracks = g.set_desc.tracks

    # plot tracks

    # for track_id, track in enumerate(tracks):
    #     plot_track(track, "track" + "{:03d}".format(set_id) + "_" + "{:03d}".format(track_id))

    ds = pipeline.get_data()
    ds_set = get_subset_by_description(ds, trackable_set, '3d')

    all_tracks.extend([])

"""

# ds_set = get_subset_by_description(ds, trackable_set, '2d')
    
ds = pipeline.get_data()

ds.level.attrs['units'] = 'hPa'
ds.level.attrs['positive'] = 'down'

# plot kitweather mode
# if len(sys.argv) > 1 and sys.argv[1] == '-kw':
#     plot_kw(all_tracks, ds=ds, tp=rain_ds.tp, plot_prefix=(plot_dir + run + '/' + plot_file_prefix), ens_mode=ens_mode)



# no out dataset here.
pipeline.save_result(description_type='json', description_path=cfg.out_json_path, dataset_path=cfg.out_dataset_path) # dataset_path=out_dataset_path,
