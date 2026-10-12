#!/usr/bin/env python

# Usage Example
from feature.pipeline import FeaturePipeline
from feature.identification.aew_vortices import AEWVortexIdentification
from feature.tracking.aew_vortices import AEWVortexTracking
from datetime import timedelta, datetime
import json, sys

# from feature.identification._proto_gen import african_easterly_waves_pb2
from feature.identification._proto_gen import aew_vortices_pb2
from os.path import expanduser, join
from feature.util.graph import DataGraph
import feature.identification.aew_vortices.configuration as cfg
import os, sys, glob, shutil
from feature.util.data_utils import get_subset_by_description
import xarray as xr
xr.set_options(keep_attrs=True)
import numpy as np
from feature.identification.aew_vortices.plotting import plot_kw_style
from types import SimpleNamespace

pipeline = FeaturePipeline(aew_vortices_pb2, processing_mode='2d')

if len(sys.argv) >= 3 and sys.argv[1] == '-y':
    year = sys.argv[2]
    
    cfg.start_date = f'{year}-06-01T00:00'
    cfg.end_date = f'{year}-10-31T23:00'
    
    method = cfg.method
    if method == 'critpts':
        cfg.in_data = f"/project/meteo/w2w/C3/fischer/data/aew_new/{year}/uv{year}.nc"
    elif method == 'hodges':
        cfg.in_data = f"/project/meteo/w2w/C3/fischer/data/aew_new/{year}/F32.nc"
        # custom params for low res
        # cfg.max_dist_deg_per_hr_lon = 1 # allow 2 cell jumpy per 6h?
        # cfg.max_dist_deg_per_hr_lat = 0.5 # allow 1px 2.5 deg jump
    cfg.in_data_fft = None
    cfg.u_dim = None # auto infer
    cfg.v_dim = None # auto infer
    field = cfg.field
    bpf = cfg.bpf
    cfg.out_json_path = f'/project/meteo/w2w/C3/fischer/data/aew_new/{year}/easterly_vortices_{method}_{field}_{bpf}.json'
    
elif len(sys.argv) >= 2 and sys.argv[1] == '-m':
    # marlon mode
    year = sys.argv[2]
    member_str = sys.argv[3]
    
    cfg.start_date = f'{year}-06-01T00:00'
    cfg.end_date = f'{year}-10-31T23:00'
    cfg.levels = [70000] # in Pa...
    cfg.u_dim = None # auto infer grib id
    cfg.v_dim = None # auto infer grib id

    cfg.in_data = f"/project/meteo/w2w/C3/fischer/data/era5_ens/data_elda_uv700_{year}_member_{member_str}.nc"
    cfg.in_wts = f"/project/meteo/w2w/C3/fischer/data/era5_ens/wts_{year}_member_{member_str}.json" # "/project/meteo/w2w/C3/fischer/data/era5_ens/wts.json"
    cfg.in_wts_rea = f"/project/meteo/w2w/C3/fischer/data/aew_new/{year}/wts{year}.json"
    cfg.in_data_fft = None
    cfg.in_data_rain = None
    field = cfg.field
    bpf = cfg.bpf
    cfg.out_json_path = f'/project/meteo/w2w/C3/fischer/data/era5_ens/vortices_{year}_member_{member_str}.json'
    cfg.fig_dir = f'/project/meteo/w2w/C3/fischer/data/era5_ens/figs/{year}_{member_str}/'
    
# in_files_all_cv_data = cfg.cv_data_ex

in_file = cfg.in_data
in_desc = cfg.in_wts
pipeline.set_data_path(in_file)

# init AEWIdentification strategy, can take different parameters
i_strat = AEWVortexIdentification()
t_strat = AEWVortexTracking()

pipeline.set_identification_strategy(i_strat)
pipeline.set_tracking_strategy(t_strat)

# execute pipeline
pipeline.execute()

od = pipeline.get_feature_desc()
ds = pipeline.get_data()

pipeline.save_result(description_type='json', description_path=cfg.out_json_path) #, dataset_path=out_dataset_path) # dataset_path=out_dataset_path,
    
wts_json = json.load(open(in_desc), object_hook= lambda x:  SimpleNamespace(**x))
if hasattr(cfg, 'in_wts_rea'):
    wts_rea_json = json.load(open(cfg.in_wts_rea), object_hook= lambda x:  SimpleNamespace(**x))
    plot_kw_style(ds, od.sets[0], wts_json.sets[0], cfg, rea_wts_set=wts_rea_json.sets[0], lev=70000) # 70000  # wts_rea_json.sets[0]
    exit()


# plot_kw_style(ds, od.sets[0], wts_json.sets[0], cfg, rea_wts_set=None, lev=850) # 70000  # wts_rea_json.sets[0]
# plot_kw_style(ds, od.sets[1], wts_json.sets[1], cfg, lev=85000)
# plot_kw_style(ds, od.sets[1], wts_json.sets[0], cfg, lev=850)


print(ds)
# dataset.to_netcdf("/home/ws/he7273/phd_all/data/coll_oper/jjaso2021_test.nc")
exit()
