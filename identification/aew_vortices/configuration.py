#!/usr/bin/env python
from os.path import expanduser, join
from datetime import timedelta
import numpy as np
import os

# data area
# latN = 35
# latS = -35
# lonW = -100
# lonE = 45

in_data = "/project/meteo/w2w/C3/fischer/data/aew_new/2019/uv2019.nc"
in_data_rain = "/project/meteo/w2w/C3/fischer/data/imerg/case2019.nc" # rain just for plotting, optional
in_data_fft = "/project/meteo/w2w/C3/fischer/data/aew_new/2019/uv2019_fft.nc"
in_wts = "/project/meteo/w2w/C3/fischer/data/aew_new/2019/wts2019.json"

# in_data = "/home/ws/he7273/phd_all/data/coll_oper/jjaso2021.nc"
# in_wts = "/home/ws/he7273/phd_all/data/coll_oper/jjaso2021.json"

circles_file = "/project/meteo/w2w/C3/fischer/data/circles.nc"
fig_dir = '/project/meteo/w2w/C3/fischer/data/vortices/figs'


# circles_file = "/home/ws/he7273/phd_all/data/aew/circles.nc"
# fig_dir = '/home/ws/he7273/phd_all/data/vortices/figs'


# kw_rain_cm_file = '/home/he7273/phd_all/data/tracked/colorpalette_dyamond_prec_rate.txt'
kw_rain_cm_file = '/project/meteo/w2w/C3/fischer/belanger/colorpalette_dyamond_prec_rate.txt'

# plot_dir = '/home/ws/he7273/phd_all/data/coll_oper/' # '/project/meteo/w2w/C3/fischer/belanger/plots/' # join('/home/ws/he7273/phd_all/data/aew/plots/') # '/project/meteo/w2w/C3/fischer/belanger/plots/'

method = 'critpts' # 'hodges' or 'critpts'
field = 'rv'
bpf = True
out_json_path = f'/project/meteo/w2w/C3/fischer/data/aew_new/2019/easterly_vortices_{method}_{field}_{bpf}.json'

fig_dir = fig_dir + '_' + method + '_' + ('bpf' if bpf else 'no_bpf') + '_' + field + '/'
try:
    os.makedirs(fig_dir)
except:
    pass

levels = [600, 700, 850] # 850  # 700 hPa
u_dim = 'u'
v_dim = 'v'

base_threshold = {'cv': 0.5e-5, 'cva': 0.1e-5, 'rv': 1e-5, 'rv_bpf': 1e-5} # testing
# for correspondance
max_dist_deg_per_hr_lon = 0.66 # 0.5deg/hr =~ max 15 m/s,use roughly 20
max_dist_deg_per_hr_lat = 0.35 # allowed to jump 2deg N/S per 6hs

duration_threshold = timedelta(hours=36)

# for tracking filter: average speed
filter_easterly = True # AEWs
avg_speed_t = timedelta(hours=36)
avg_speed_min_m_per_s = -3 # min abs, so faster than 3 to west
avg_speed_min_deg_per_h = avg_speed_min_m_per_s * 3.6 * 0.00914

# time of interest, if None all
# june-oct is AEW season

start_date = '2019-06-01T00:00' # '2019-08-20T00:00'  # '2022-08-01T00:00' # None # '2008-08-01T00:00' #  # '2008-08-01T00:00'
end_date = '2019-10-31T23:00' # '2019-08-20T06:00'  # '2022-08-15T00:00'  # None # '2008-08-15T00:00' # None # '2008-08-03T00:00'

radius = 500000  # 500 km
prominence_radius = 400000  # 300 km
