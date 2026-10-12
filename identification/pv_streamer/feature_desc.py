import numpy as np
import xarray as xr
import math

# get the feature data from the labeled dataset
def get_feature_data(self_, stereo_ds, level_list, dist_expand, area_map, config, index_accessor):
    pvu_stereo3d = stereo_ds['pvu'].values
    labeled_stereo = stereo_ds['streamer']
    reshuffled_labeled_stereo = xr.zeros_like(stereo_ds['streamer'], dtype=int)
    
    lonlat_indices = np.stack([np.squeeze(stereo_ds['lon_field'].values), np.squeeze(stereo_ds['lat_field'].values)],
                              axis=2)

    max_num_in_data = np.amax(labeled_stereo.values)
    features = []

    from .._proto_gen import pv_streamer_pb2

    for label_i in range(1, max_num_in_data + 1):
        # get new feature, set id
        obj = self_.get_new_feature(index_accessor)
        # each feature
        obj_props = obj.properties

        obj_bin_stereo = (labeled_stereo == label_i)  # careful, dont use latlon due to singularities (180WE)
        reshuffled_labeled_stereo.data[obj_bin_stereo] = obj.id
        
        obj_indices_stereo = np.where(obj_bin_stereo)

        obj_areas_s = area_map[obj_indices_stereo[1:]]  # area per pixel,
        obj_pv_s = pvu_stereo3d[obj_indices_stereo]

        if np.amin(obj_pv_s) < 0 and np.amax(obj_pv_s) > 0:
            # both areas of negative PV (<-2pvu) and positive PV (>+2pvu) in one single feature. probably high-frequency noise near ground
            # set both to positive for positive interpolation weights
            obj_pv_s = np.abs(obj_pv_s)

        # ----------------------
        # CENTROID
        # ----------------------
        if config.dims == 2:
            ce = get_pv_centroid_2d(obj_indices_stereo, obj_areas_s, obj_pv_s, lonlat_indices, config)
        else:
            ce = get_pv_centroid_3d(obj_indices_stereo, obj_areas_s, obj_pv_s, lonlat_indices, level_list, config)
            obj_props.centroid.level = ce[2]
        obj_props.centroid.lon = ce[0]
        obj_props.centroid.lat = ce[1]

        # ----------------------
        # VOLUME
        # ----------------------
        volume_km3 = 0
        volume_km2K = 0
        if config.dims == 3:
            lv_idxs = {}  # np.zeros(shape=(obj_indices_stereo[0].max() - obj_indices_stereo[0].min() + 1))
            lv_area = {}  # np.zeros(shape=(obj_indices_stereo[0].max() - obj_indices_stereo[0].min() + 1))
            for k in range(obj_indices_stereo[0].min(), obj_indices_stereo[0].max() + 1):
                lv_idxs[k] = np.where(obj_indices_stereo[0] == k)
                lv_area[k] = np.sum(obj_areas_s[lv_idxs[k]])

                # take height of this layer instead of inbetween two, to handle single layer 3d data.
                # config has height inbetween layers.
                height_km = (config.v_data_z_dist[max(0, k - 1)] + config.v_data_z_dist[min(len(config.v_data_z_dist) - 1, k)]) / 2
                volume_km3 += lv_area[k] * height_km
                volume_km2K += lv_area[k] * config.res_z  # layer * dist between layers.

            obj_props.volume_km3 = volume_km3
            obj_props.volume_km2K = volume_km2K

        elif config.dims == 2:

            lv_area = np.sum(obj_areas_s)
            obj_props.area_km2 = lv_area


        # ----------------------
        # BOUNDING BOX (GEOREF)
        # ----------------------

        lonlat_values_obj = lonlat_indices[obj_indices_stereo[1:]]
        lon_values_obj = lonlat_values_obj[:, 0]
        lat_values_obj = lonlat_values_obj[:, 1]
        min_lat, max_lat = min(lat_values_obj), max(lat_values_obj)
        min_lon, max_lon = min(lon_values_obj), max(lon_values_obj)
        # assume antimeridian PV structure if minmax diff is >350. if antimeridian it should contain ~-180 and ~+180
        if max_lon - min_lon > 350:
            min_lon = min(lon_values_obj[lon_values_obj > 0])
            max_lon = max(lon_values_obj[lon_values_obj < 0])

        obj_props.min_bbox.lat = min_lat
        obj_props.min_bbox.lon = min_lon
        obj_props.max_bbox.lat = max_lat
        obj_props.max_bbox.lon = max_lon

        if config.dims == 3:
            level_values_obj = level_list[obj_indices_stereo[0]]
            min_lev = min(level_values_obj)
            max_lev = max(level_values_obj)
            obj_props.min_bbox.level = min_lev
            obj_props.max_bbox.level = max_lev

        # ----------------------
        # AVERAGE / MAX PV
        # ----------------------

        sc_max_pv = max(obj_pv_s.min(), obj_pv_s.max(), key=abs)
        sc_avg_pv = np.average(obj_pv_s)

        obj_props.max_pv = sc_max_pv
        obj_props.avg_pv = sc_avg_pv

        expa_values = dist_expand[obj_indices_stereo] # is cutoff if inf?
        is_cutoff = (np.inf in expa_values)
        obj_props.is_cutoff = is_cutoff

        # ----------------------
        # LENGTH + AVG DISTANCE
        # ----------------------

        max_length = np.amax(expa_values) - config.alg.w_bar
        average_dist_ratio = np.inf if is_cutoff else np.average(expa_values) / max_length

        obj_props.length = max_length
        obj_props.length_ratio = average_dist_ratio

        # ----------------------
        # ELLIPSOID
        # ----------------------

        if len(obj_indices_stereo[0]) < 2:
            # only one pixel - cant create ellip.
            pass
        else:

            px_list = np.asarray(obj_indices_stereo)
            cov = np.cov(px_list)
            evals, evecs = np.linalg.eigh(cov)  # eigenvectors of obj
            evecs_len = np.ones(shape=(config.dims,), dtype=float) # length of them to be calculated

            for ev_i in range(config.dims):  # 0..1 if 2d, 0..2 if 3d .. small to big
                evec_cur = np.array([evecs[:, ev_i]])
                vec_repeat = np.repeat(evec_cur, px_list.shape[1], axis=0)
                dot_axis = np.multiply(px_list.T, vec_repeat)
                sum_dot_axis = np.sum(dot_axis, axis=1)
                evecs_len[ev_i] = np.ptp(sum_dot_axis)

            obj_props.major_axis.z = evecs[0][0] * evecs_len[0]
            obj_props.major_axis.y = evecs[0][1] * evecs_len[0] # TODO this is in pixel space.
            obj_props.major_axis.x = evecs[0][2] * evecs_len[0]

            if config.dims == 3:
                obj_props.medium_axis.z = evecs[1][0] * evecs_len[1]
                obj_props.medium_axis.y = evecs[1][1] * evecs_len[1]
                obj_props.medium_axis.x = evecs[1][2] * evecs_len[1]

            obj_props.minor_axis.z = evecs[-1][0] * evecs_len[-1]
            obj_props.minor_axis.y = evecs[-1][1] * evecs_len[-1]
            obj_props.minor_axis.x = evecs[-1][2] * evecs_len[-1]

        features.append(obj)

        # add_boundary_rep(levellatlon_mask, obj, config)
        # add_voxel_rep(levellatlon_indices, obj)
        # add_contour_line_rep(obj_bin, obj, longlatitudes_2d)

    # obj_with_indices = list(zip(range(1, max_num_in_data + 1), features))
    return features, reshuffled_labeled_stereo


# centroid (Massenschwerpunkt, 1.Moment) of feature
# weighted by PV and distortion in stereo
# sum(area_xy*PV_xy
# then interpolate the positions lat/lon
def get_pv_centroid_2d(object_px, object_areas, object_pv, lonlat_table, config):
    AP = np.multiply(object_areas, object_pv)

    avg_x = np.average(object_px[1], weights=AP)
    avg_y = np.average(object_px[2], weights=AP)
    return bilinear_interpolate_latlon(lonlat_table, avg_y, avg_x).tolist()


def get_pv_centroid_3d(object_px, object_areas, object_pv, lonlat_table, level_list, config):
    AP = np.multiply(object_areas, object_pv)

    avg_x = np.average(object_px[1], weights=AP)
    avg_y = np.average(object_px[2], weights=AP)
    avg_lonlat_value = bilinear_interpolate_latlon(lonlat_table, avg_y, avg_x).tolist()

    vertical_component_plist = np.array(level_list)[object_px[0]]
    PAP = np.multiply(AP, vertical_component_plist)

    avg_z = np.average(object_px[0], weights=PAP)

    avg_level_value = linear_interpolate(level_list, avg_z).item()
    avg_lonlat_value = bilinear_interpolate_latlon(lonlat_table, avg_y, avg_x).tolist()

    return avg_lonlat_value + [avg_level_value]


# bilinear interpolate indices of a 2d grid containing latlon indices
# special version for -180/180 issues
def bilinear_interpolate_latlon(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1);
    x1 = np.clip(x1, 0, im.shape[1] - 1);
    y0 = np.clip(y0, 0, im.shape[0] - 1);
    y1 = np.clip(y1, 0, im.shape[0] - 1);

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    # if dateline
    # I_[0] contain both ~-180 and ~+180
    if min(Ia[0], Ib[0], Ic[0], Id[0]) < -90 and max(Ia[0], Ib[0], Ic[0], Id[0]) > +90:
        if Ia[0] < 0:
            Ia[0] = Ia[0] + 360.0
        if Ib[0] < 0:
            Ib[0] = Ib[0] + 360.0
        if Ic[0] < 0:
            Ic[0] = Ic[0] + 360.0
        if Id[0] < 0:
            Id[0] = Id[0] + 360.0

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    latlon = wa * Ia + wb * Ib + wc * Ic + wd * Id
    if latlon[0] > 180.0:
        latlon[0] -= 360.0
    return latlon


def linear_interpolate(list, idx):
    # handle scalar and array
    idx = np.asarray(idx)
    scalar_input = False
    if idx.ndim == 0:
        idx = idx[None]  # Makes x 1D
        scalar_input = True

    lo = idx.astype(dtype=int)
    w = idx - lo

    # hi = min(lo + 1, len(list) - 1)
    ret = np.where(w == 0, list[lo], list[lo] + (list[np.minimum(lo + 1, len(list) - 1)] - list[lo]) * w)

    if scalar_input:
        return np.squeeze(ret)
    return ret
