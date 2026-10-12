#!/usr/bin/env python
from os.path import expanduser, join
from datetime import timedelta
import numpy as np


data_lat = (3, 35)
data_lon = (-100, 45)

contour_gap_pts = 4
match_buffer_deg = 0.5
proportion_n = 0.75
zigzag_buffer_deg = 1.0
trough_merge_dist_deg = 3.0 # set to 0 to turn off
zigzag_min_lat_overlap_deg = 1.0

match_one_to_one = False
match_expected_speed_m_per_s = 7.0
match_expected_speed_deg_per_h = -match_expected_speed_m_per_s * 3.6 * 0.00914

aew_clim_dir = '/dev/shm/cemerson/cv_clim_era5.nc' 

generate_output = True

plot_dir = '/unity/f1/cemerson/' 

timedelta_ana = np.timedelta64(7, 'D')

ens_rain_threshold = 1.0 # display prob of exceeding X mm hr-1

cv_data_dir = '/dev/shm/cemerson/' # reference where ALL cv data is (only for clim calc.)
cv_data_ex = join(cv_data_dir, '*.nc')

def get_clim_file():
    fn = aew_clim_dir
    #fn = (aew_clim_dir + "cv_clim_" + str(abs(data_lat[1])) + ('N' if data_lat[1] > 0 else 'S') + "_"
    #      + str(abs(data_lat[0])) + ('N' if data_lat[0] > 0 else 'S') + "_"
    #      + str(abs(data_lon[0])) + ('E' if data_lon[0] > 0 else 'W') + "_"
    #      + str(abs(data_lon[1])) + ('E' if data_lon[1] > 0 else 'W')) + ".nc"
    return fn


# wave area to be extracted: at least one point of trough needs to be in this range
wave_filter_lat = (-5, 35)
wave_filter_lon = (-110, 55)

levels = [700]  # 700 hPa
u_dim = 'u_total'
v_dim = 'v_total'

add_fake_wts = True # add fake wavetroughs in feature description to bridge gaps in trackspb_reference

# Algorithm parameters
# max u wind (m/s) (0 = only keep west-propagating). Belanger: 2.5; Berry: 0.0
max_u_thresh = 2.5  # m/s

# CV anomaly percentile
# NOTE: this is precomputed from the climatology, so delete climatology file and change this after to be re-executed.
cv_percentile = 66  # 66% percentile of PV anomalies as reference on what areas to consider
cv_sigma_k = 0.8 # could be 0.7
cv_sigma_floor_pctile = 20

# need positive 2nd time derivative
second_advection_min_thr = 0.0

### FILTERING

# spatial filtering: if wave to small, discard
# threshold in degrees of wave length (sum of wave segments)
degree_len_thr = 5

### TRACKING
duration_threshold = timedelta(days=2)
# max time allowed between detected waves
max_cmp_delta = timedelta(hours=12)

# duration of min avg speed
avg_speed_t = timedelta(hours=36)

# changed to -2 according to belanger, consider adding a maximum of -25 as they do
avg_speed_min_m_per_s = -2 # min abs, so faster than 3 to west
avg_speed_max_m_per_s = -25
avg_speed_min_deg_per_h = avg_speed_min_m_per_s * 3.6 * 0.00914
avg_speed_max_deg_per_h = avg_speed_max_m_per_s * 3.6 * 0.00914
add_intensity_measure = True

# speed range of AEWs
# at 10°N we have in longitude direction 0.00914 degrees/km (360/(40,000*cos(10deg)))
speed_range_m_per_s = [3.0, 18.0] # [5,10], but be more gentle/conservative with polygons.
speed_deg_per_h = [-m_per_s * 3.6 * 0.00914 for m_per_s in speed_range_m_per_s]  # negative -> westward [-5, -10]

def is_point_in_area(lon, lat):

    if lon < -65:
        if 5 < lat < 28:
            return True
    elif lon < -50:
        if 3 < lat < 27:
            return True
    else:
        if 0 < lat < 26:
            return True
    
    return False
