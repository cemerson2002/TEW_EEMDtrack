#!/usr/bin/env python

# Usage Example
from feature.pipeline import FeaturePipeline
from feature.identification.african_easterly_waves import AEWIdentification
from feature.tracking.african_easterly_waves import AEWTracking
from datetime import timedelta, datetime
from feature.identification._proto_gen import african_easterly_waves_pb2
from os.path import join
from feature.identification.african_easterly_waves.plotting import plot_kw, plot_differences, plot_track, plot_track_in_ts, plot_timesteps_from_desc, plot_tracks_from_desc
import feature.identification.african_easterly_waves.configuration as cfg
import os, sys, glob, shutil
from feature.identification.african_easterly_waves.processing import interpolate_wts, add_wts_to_ds
from feature.util.data_utils import get_subset_by_description
import xarray as xr
xr.set_options(keep_attrs=True)
import numpy as np
from pprint import pprint

pipeline = FeaturePipeline(african_easterly_waves_pb2, processing_mode='2d', workers=1)

if len(sys.argv) >= 3 and sys.argv[1] == '-kw':
    kw_mode = True
    from kwutil import *
    print("Executing in kitweather mode...")
    # kitweather: use last 7 days of analysis and the ecmwf forecast
    
    run = None
    if len(sys.argv) == 4:
        run = sys.argv[3]

    if sys.argv[2] == 'ecmwf_fc':
        ds, run = get_ecmwf_forecast(run=run, include_analysis_delta=cfg.timedelta_ana, add_prec_rate=True)
        plot_dir = cfg.plot_dir_ecmwf
        plot_file_prefix = cfg.plot_dir_ecmwf_prefix
        ens_mode = False
    elif sys.argv[2] == 'icon_fc':
        ds, run = get_icon_forecast(run=run, include_analysis_delta=cfg.timedelta_ana, add_prec_rate=True)
        plot_dir = cfg.plot_dir_icon
        plot_file_prefix = cfg.plot_dir_icon_prefix
        ens_mode = False
    elif sys.argv[2] == 'gfs_fc':
        ds, run = get_gfs_forecast(run=run, include_analysis_delta=cfg.timedelta_ana, add_prec_rate=True)
        plot_dir = cfg.plot_dir_gfs
        plot_file_prefix = cfg.plot_dir_gfs_prefix
        ens_mode = False
    elif sys.argv[2] == 'ecmwf_ens':
        ds, run = get_ecmwf_ensemble(run=run, add_prec_rate=True)
        plot_dir = cfg.plot_dir_ecmwf_ens
        plot_file_prefix = cfg.plot_dir_ecmwf_ens_prefix
        ens_mode = True
    elif sys.argv[2] == 'gfs_ens':
        ds, run = get_gfs_ensemble(run=run, add_prec_rate=True)
        plot_dir = cfg.plot_dir_gfs_ens
        plot_file_prefix = cfg.plot_dir_gfs_ens_prefix
        ens_mode = True
    else:
        print("Unknown command line parameter " + sys.argv[2] + ", expected ecmwf_fc or icon_fc.")
        exit(1)

    data_ds = ds[0]
    rain_ds = None if len(ds) == 1 else ds[1]
    print("Done collecting files.")

    pipeline.set_data(data_ds)
elif len(sys.argv) >= 3 and sys.argv[1] == '-y':
    kw_mode = False
    year = sys.argv[2]
    in_file = cfg.main_data_dir + year + '/uv' + year + '.nc'
    out_dir = cfg.main_data_dir + year + '/'
    
    out_json_path = out_dir + 'wts' + year + '.json'
    out_data_path = out_dir + 'wts' + year + '.nc'
    
    pipeline.set_data_path(in_file)

elif len(sys.argv) >= 2 and sys.argv[1] == '-m': # marlon
    kw_mode = False
    
    # marlon mode
    year = sys.argv[2]
    member_str = sys.argv[3]
    
    in_files = cfg.era5_ens_dir + f"data_elda_uv700_{year}_member_{member_str}.nc"
    
    out_json_path = cfg.era5_ens_dir + f'wts_{year}_member_{member_str}.json'
    out_data_path = cfg.era5_ens_dir + f'wts_field_{year}_member_{member_str}.nc'
    cfg.u_dim = None # infer from file (grib)
    cfg.v_dim = None
    cfg.levels = 70000 # in Pa in files
    
    pipeline.set_data_path(in_files)
    
else:
    kw_mode = False
    in_file = cfg.in_files
    out_dir = cfg.out_dir
    out_json_path = cfg.out_json_path
    out_data_path = cfg.out_data_path
    pipeline.set_data_path(in_file)

# init AEWIdentification strategy, can take different parameters
enable_out = (not kw_mode) and cfg.generate_output
i_strat = AEWIdentification(wt_out_file=enable_out, cv='cv') # , year_summer=proc_summer_of_year, month=proc_month_of_year)
t_strat = AEWTracking()
t_strat.set_threads(32)

pipeline.set_identification_strategy(i_strat)
pipeline.set_tracking_strategy(t_strat, generate_tracks=True)

# execute pipeline
pipeline.execute()
print("Execute done.")


all_tracks = []

ds = pipeline.get_data()

# plot kitweather mode
if len(sys.argv) > 1 and sys.argv[1] == '-kw':
    plot_kw(all_tracks, ds=ds, tp=rain_ds.tp, plot_prefix=(plot_dir + run + '/' + plot_file_prefix), ens_mode=ens_mode)



# no out data besides plots on kitweather
if len(sys.argv) > 1 and sys.argv[1] == '-kw':
    # delete old plots
    subdirs = [dI for dI in os.listdir(plot_dir) if os.path.isdir(os.path.join(plot_dir,dI))]
    for sd in subdirs: # for each subdir in plot dir

        if not sd == run and datetime.fromtimestamp(os.path.getmtime(os.path.join(plot_dir, sd))) < datetime.now() - timedelta(days=7): # not touched in a week? delete it.
            print("Removing directory " + str(os.path.join(plot_dir, sd)))
            shutil.rmtree(os.path.join(plot_dir, sd))


    # All done. Update text file containing time of latest finished run.
    if os.path.exists(cfg.plot_dir_ecmwf + run) and os.path.exists(cfg.plot_dir_icon + run) and os.path.exists(cfg.plot_dir_gfs + run) and not ens_mode:
        print("Run " + run + " done for ECMWF, ICON, GFS. Update latest run info file.") 
        yyyymmddhh = run[4:]

        latest_run_file_path = cfg.latest_run_dir + cfg.latest_run_info_file_name

        with open(latest_run_file_path, 'w+') as info_file:
            info_file.write(yyyymmddhh)
        
    # All done. scp data over to webserver.
    """
    path_webserver = '/home/iconeps/Data3/plots/ecmwf/aew_prediction_maps/'
    print('scp -r ' + cfg.plot_dir + time_dir + '/ ' + 'iconeps@imk-tss-web.imk-tro.kit.edu:' + path_webserver)
    os.system('scp -r ' + cfg.plot_dir + time_dir + '/ ' + 'iconeps@imk-tss-web.imk-tro.kit.edu:' + path_webserver)
    os.system('scp ' + cfg.latest_run_info_file + ' iconeps@imk-tss-web.imk-tro.kit.edu:' + path_webserver)
    """
    exit()

# out_netcdf_path = data_path + '_streamers.nc'

ob = pipeline.get_feature_desc()

# Print CV averages for specific year to histogram.
"""
avg_cvs = []
for wt_set in ob.sets:
    for track_id, track in enumerate(wt_set.tracks):
        set_nodes = [e.parent for e in track.edges]
        for node in set_nodes:
            # print(node.feature.properties.avg_cv_along_wt)
            avg_cvs.append(node.feature.properties.avg_cv_along_wt)

import numpy as np
np.save(f'cv_avg_{year}.npy', np.array(avg_cvs), allow_pickle=True)
"""

if enable_out:
    ds = i_strat.orig_dataset
    ob = pipeline.get_feature_desc()
    ds = ds.load()

    ### ADD FAKE WTs
    # cant be called parallelized cause of IDs
    ob = interpolate_wts(ob, african_easterly_waves_pb2)
    
    ### add WTs to orig DS. field tracks and field WTs
    ds = add_wts_to_ds(ds, ob)
    print(ds)
    if 'u' in ds.data_vars:
        ds = ds.drop_vars(['u'])
    if 'v' in ds.data_vars:
        ds = ds.drop_vars(['v'])
    if 'cv' in ds.data_vars:
        ds = ds.drop_vars(['cv'])
    
    pipeline.set_data(ds)

    pipeline.save_result(description_type='json', description_path=out_json_path, dataset_path=out_data_path)
