#!/usr/bin/env python
from os.path import expanduser, join
from datetime import timedelta 
import numpy as np
import os

in_data_pv = "/project/meteo/w2w/C3/fischer/data/aew_new/2021/pv2021.nc"
in_data_fft = "/project/meteo/w2w/C3/fischer/data/aew_new/2021/uv2021_fft.nc"
in_data_wts = "/project/meteo/w2w/C3/fischer/data/aew_new/2021/wts2021.nc"
in_wts = "/project/meteo/w2w/C3/fischer/data/aew_new/2021/wts2021.json"

in_data_era5 = "/project/meteo/w2w/C3/fischer/data/aew_new/"


out_json_path = "/project/meteo/w2w/C3/fischer/data/aew_new/2021/desc.json"
out_dataset_path = "/project/meteo/w2w/C3/fischer/data/aew_new/test.nc"

circles_file = "/project/meteo/w2w/C3/fischer/data/circles.nc"

# kw_rain_cm_file = '/project/meteo/w2w/C3/fischer/belanger/colorpalette_dyamond_prec_rate.txt'
# fig_dir = '/project/meteo/w2w/C3/fischer/data/vortices/figs'

# levels = [700, 850]  # 700 hPa
u_dim = 'u'
v_dim = 'v'
use_wt_field = 'tracks'

# base_threshold = {'cv': 3e-5, 'cva': 1e-5, 'rv': 3e-5}

start_date =  None # '2008-08-01T00:00' #  # '2008-08-01T00:00'
end_date = None # '2008-08-15T00:00' # None # '2008-08-03T00:00'

# PVU threshold
pvu_thr_upper = 0.7
pvu_thr_lower = 0.5

# Phase Threshold
phase_range = (-0.9 * np.pi, -0.1 * np.pi) # -pi/2 is WT, +- pi/2 ?
phase_level = 700

# WT radius
wt_radius = 800000 # 800km


# lat filter month->(min,max)
"""
lat_range = {6:  (3, 16),
    7:  (3, 16),
    8:  (3, 16),
    9:  (3, 16),
    10: (3, 16),
}
"""

# where to identify PV
lat_range = (3, 16) # start range
lev_range = (299, 901) # start and filter range
