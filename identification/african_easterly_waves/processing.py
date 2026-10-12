# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import math
from feature.util.enstools_utils import get_u_var, get_v_var, get_vertical_dim, get_longitude_dim, \
    get_latitude_dim
from feature.util.data_utils import pb_str_to_datetime64, simplenamespace_to_proto, datetime64_to_pb_str, proto_to_simplenamespace, clip
from google.protobuf.message import Message
from skimage.draw import line
from collections import defaultdict

# calculates the dx's and dy's for each grid cell
# takes list of latitudes and longitudes as input and returns field with dimensions len(lats) x len(lons)
def calculate_dx_dy(lats, lons):
    lat_res = lats[1] - lats[0]
    lon_res = lons[1] - lons[0]
    nlat = len(lats)
    nlon = len(lons)
    lon_percent = (lon_res * nlon) / 360.0  # percentage of global longitudes. e.g. if 110W to 70E -> 0.5
    lat_percent = (lat_res * nlat) / 180.0  # percentage of global latitudes
    Re = 6378100

    # field filled with the latitude value at each grid point
    lat_f = np.tile(lats, (nlon, 1)).transpose()
    # field filled with the longitude value at each grid point
    lon_f = np.tile(lons, (nlat, 1))
    # dx: circumference at current latitude  *  percentage of longitudes in dataset  /  amount of lon grid points
    dx = np.cos(lat_f * math.pi / 180.0) * 2.0 * math.pi * Re * lon_percent / nlon
    # dy: constant everywhere: half circumference  *  latitude percentage /  amount of lat points
    dy = (lat_f * 0) + lat_percent * math.pi * Re / nlat

    return dx, dy


def compute_cv(dataset, u_str, v_str, cv_str):
    xr.set_options(keep_attrs=True)  # assign coords keeps attributes

    lon_str = get_longitude_dim(dataset)
    lat_str = get_latitude_dim(dataset)

    lats = dataset.coords[lat_str].data
    lons = dataset.coords[lon_str].data

    # be careful if your data is non-continuous by the chosen window (e.g. +120E..-120W)
    # reorder axis for efficient numpy broadcasting
    dataset = dataset.sortby(lat_str)
    dataset = dataset.transpose(..., lat_str, lon_str)

    dataset[cv_str] = xr.zeros_like(dataset[u_str], dtype=float)
    dataset[cv_str].attrs = {'long_name': "Curvature Vorticity", 'units': "s**-1"}

    # calculate dx and dy for each cell in grid in meters
    dx, dy = calculate_dx_dy(lats, lons)

    # Relative Vorticity = dv/dx - du/dy, use central differences
    u_arr = dataset[u_str].data
    v_arr = dataset[v_str].data
    ndims = u_arr.ndim

    # roll axes so we have a reference to each cell's neighbours
    # ax[-1] = lon, ax[-2] = lat
    # works as expected if longitude band is full 360 degrees. Rolling does exactly that.
    v_xp = np.roll(v_arr, -1, axis=ndims - 1)  # roll -1 to get +1. v_xp = v(x+1)
    v_xm = np.roll(v_arr, 1, axis=ndims - 1)
    v_yp = np.roll(v_arr, -1, axis=ndims - 2)
    v_ym = np.roll(v_arr, 1, axis=ndims - 2)

    u_xp = np.roll(u_arr, -1, axis=ndims - 1)  # roll -1 to get +1
    u_xm = np.roll(u_arr, 1, axis=ndims - 1)
    u_yp = np.roll(u_arr, -1, axis=ndims - 2)
    u_ym = np.roll(u_arr, 1, axis=ndims - 2)

    # central differences:  (V(x+1)-V(x-1)) / 2x - (U(y+1)-U(y-1)) / 2y
    RV = ((v_xp - v_xm) / (2 * dx)) - ((u_yp - u_ym) / (2 * dy))
    # results verified to ERA5 vo-data

    # split into shear + curvature: RV = -dV/dn + V/R
    # shear vorticity:
    # rate of change of wind speed in the direction of flow
    # -dV/dn

    # wind direction of this cell
    ref_angle = np.arctan2(v_arr, u_arr)

    ### METHOD 1 ### from analytical standpoint

    sin_angle = np.sin(ref_angle)
    cos_angle = np.cos(ref_angle)
    # here is where the magic happens...
    # to cartesian: dV/dn = - dV/dx * sin(phi) + dV/dy * cos(phi) ## n is normal to V, split it into x,y respecting the
    # n-direction, where phi is the rotation angle of the natural coordinate system, this also is the angle of the
    # wind vector
    #
    # we get the magnitude of shear as the projection of dV/dn onto the vector itself (direction e_t)
    # with e_t = cos(phi)*e_x + sin(phi)*e_y , so:
    # dV/dn * e_t = du/dx * (-sin*cos) + du/dy cos^2 + dv/dx *(-sin^2) + dv/dy (cos*sin)
    SV = (u_xp - u_xm) / (2 * dx) * (-sin_angle * cos_angle) + \
         (u_yp - u_ym) / (2 * dy) * (cos_angle ** 2) + \
         (v_xp - v_xm) / (2 * dx) * (-(sin_angle ** 2)) + \
         (v_yp - v_ym) / (2 * dy) * (sin_angle * cos_angle)
    SV = -SV  # -dV/dn

    # remainder is CV
    CV = RV - SV
    dataset[cv_str].values = CV

    xr.set_options(keep_attrs='default')  # assign coords keeps attributes
    return dataset


import math
from feature.util.enstools_utils import get_u_var, get_v_var, get_vertical_dim, get_longitude_dim, \
    get_latitude_dim



# calculates the dx's and dy's for each grid cell
# takes list of latitudes and longitudes as input and returns field with dimensions len(lats) x len(lons)
def calculate_dx_dy(lats, lons):
    lat_res = lats[1] - lats[0]
    lon_res = lons[1] - lons[0]
    nlat = len(lats)
    nlon = len(lons)
    lon_percent = (lon_res * nlon) / 360.0  # percentage of global longitudes. e.g. if 110W to 70E -> 0.5
    lat_percent = (lat_res * nlat) / 180.0  # percentage of global latitudes
    Re = 6378100

    # field filled with the latitude value at each grid point
    lat_f = np.tile(lats, (nlon, 1)).transpose()
    # field filled with the longitude value at each grid point
    lon_f = np.tile(lons, (nlat, 1))
    # dx: circumference at current latitude  *  percentage of longitudes in dataset  /  amount of lon grid points
    dx = np.cos(lat_f * math.pi / 180.0) * 2.0 * math.pi * Re * lon_percent / nlon
    # dy: constant everywhere: half circumference  *  latitude percentage /  amount of lat points
    dy = (lat_f * 0) + lat_percent * math.pi * Re / nlat

    return dx, dy


def compute_cv(dataset, u_str, v_str, cv_str):
    xr.set_options(keep_attrs=True)  # assign coords keeps attributes

    lon_str = get_longitude_dim(dataset)
    lat_str = get_latitude_dim(dataset)

    lats = dataset.coords[lat_str].data
    lons = dataset.coords[lon_str].data

    # be careful if your data is non-continuous by the chosen window (e.g. +120E..-120W)
    # reorder axis for efficient numpy broadcasting
    dataset = dataset.sortby(lat_str)
    dataset = dataset.transpose(..., lat_str, lon_str)

    dataset[cv_str] = xr.zeros_like(dataset[u_str], dtype=float)
    dataset[cv_str].attrs = {'long_name': "Curvature Vorticity", 'units': "s**-1"}

    # calculate dx and dy for each cell in grid in meters
    dx, dy = calculate_dx_dy(lats, lons)

    # Relative Vorticity = dv/dx - du/dy, use central differences
    u_arr = dataset[u_str].data
    v_arr = dataset[v_str].data
    ndims = u_arr.ndim

    # roll axes so we have a reference to each cell's neighbours
    # ax[-1] = lon, ax[-2] = lat
    # works as expected if longitude band is full 360 degrees. Rolling does exactly that.
    v_xp = np.roll(v_arr, -1, axis=ndims - 1)  # roll -1 to get +1. v_xp = v(x+1)
    v_xm = np.roll(v_arr, 1, axis=ndims - 1)
    v_yp = np.roll(v_arr, -1, axis=ndims - 2)
    v_ym = np.roll(v_arr, 1, axis=ndims - 2)

    u_xp = np.roll(u_arr, -1, axis=ndims - 1)  # roll -1 to get +1
    u_xm = np.roll(u_arr, 1, axis=ndims - 1)
    u_yp = np.roll(u_arr, -1, axis=ndims - 2)
    u_ym = np.roll(u_arr, 1, axis=ndims - 2)

    # central differences:  (V(x+1)-V(x-1)) / 2x - (U(y+1)-U(y-1)) / 2y
    RV = ((v_xp - v_xm) / (2 * dx)) - ((u_yp - u_ym) / (2 * dy))
    # results verified to ERA5 vo-data

    # split into shear + curvature: RV = -dV/dn + V/R
    # shear vorticity:
    # rate of change of wind speed in the direction of flow
    # -dV/dn

    # wind direction of this cell
    ref_angle = np.arctan2(v_arr, u_arr)

    ### METHOD 1 ### from analytical standpoint

    sin_angle = np.sin(ref_angle)
    cos_angle = np.cos(ref_angle)
    # here is where the magic happens...
    # to cartesian: dV/dn = - dV/dx * sin(phi) + dV/dy * cos(phi) ## n is normal to V, split it into x,y respecting the
    # n-direction, where phi is the rotation angle of the natural coordinate system, this also is the angle of the
    # wind vector
    #
    # we get the magnitude of shear as the projection of dV/dn onto the vector itself (direction e_t)
    # with e_t = cos(phi)*e_x + sin(phi)*e_y , so:
    # dV/dn * e_t = du/dx * (-sin*cos) + du/dy cos^2 + dv/dx *(-sin^2) + dv/dy (cos*sin)
    SV = (u_xp - u_xm) / (2 * dx) * (-sin_angle * cos_angle) + \
         (u_yp - u_ym) / (2 * dy) * (cos_angle ** 2) + \
         (v_xp - v_xm) / (2 * dx) * (-(sin_angle ** 2)) + \
         (v_yp - v_ym) / (2 * dy) * (sin_angle * cos_angle)
    SV = -SV  # -dV/dn

    # remainder is CV
    CV = RV - SV
    dataset[cv_str].values = CV

    xr.set_options(keep_attrs='default')  # assign coords keeps attributes
    return dataset


def populate_feature(obj_props, path, data_chunk, cfg):
    # obj_props.num_nodes = len(path)
    # fill the properties defined in the .proto file.

    # first, remove vertices out of area
    filtered_vertices = []
    for v_idx, v in enumerate(path):
        lon = v[0]
        lat = v[1]

        # dont use vertex not fulfilling area restriction
        if cfg.is_point_in_area(lon, lat):
            filtered_vertices.append(v)
            
    if not filtered_vertices:
        return

    # vertices of path
    min_lat, max_lat, min_lon, max_lon = 90.0, -90.0, 180.0, -180.0
    dist_deg = 0.0
    for v_idx, v in enumerate(filtered_vertices):
        line_pt = obj_props.line_pts.add()
        line_pt.lon = v[0]
        line_pt.lat = v[1]

        if v[0] < min_lon:
            min_lon = v[0]
        if v[0] > max_lon:
            max_lon = v[0]
        if v[1] < min_lat:
            min_lat = v[1]
        if v[1] > max_lat:
            max_lat = v[1]

        if v_idx > 0:
            dist_deg = dist_deg + (
                        ((filtered_vertices[v_idx - 1][0] - v[0]) ** 2 + (filtered_vertices[v_idx - 1][1] - v[1]) ** 2) ** 0.5)

    # bounding box
    obj_props.bb.min.lat = min_lat
    obj_props.bb.min.lon = min_lon
    obj_props.bb.max.lat = max_lat
    obj_props.bb.max.lon = max_lon

    obj_props.length_deg = dist_deg
    
    if cfg.add_intensity_measure:
        lon_str = get_longitude_dim(data_chunk)
        lat_str = get_latitude_dim(data_chunk)
        
        # FIX: Get actual coordinate values and spacings
        lon_vals = data_chunk[lon_str].values
        lat_vals = data_chunk[lat_str].values
        
        lons = len(lon_vals)
        lats = len(lat_vals)
        
        min_lat_val = lat_vals.min()
        max_lat_val = lat_vals.max()
        min_lon_val = lon_vals.min()
        max_lon_val = lon_vals.max()
        
        temp_da = xr.zeros_like(data_chunk.cv, dtype=bool)
        for v_idx in range(len(obj_props.line_pts) - 1):
            start_lonlat = obj_props.line_pts[v_idx].lon, obj_props.line_pts[v_idx].lat
            end_lonlat = obj_props.line_pts[v_idx + 1].lon, obj_props.line_pts[v_idx + 1].lat

            # new nearest-neighbor lookup
            start_idx_lon = np.argmin(np.abs(lon_vals - start_lonlat[0]))
            start_idx_lat = np.argmin(np.abs(lat_vals - start_lonlat[1]))
            end_idx_lon = np.argmin(np.abs(lon_vals - end_lonlat[0]))
            end_idx_lat = np.argmin(np.abs(lat_vals - end_lonlat[1]))

            
#            start_idx = ((start_lonlat[0] - min_lon_val) / (max_lon_val - min_lon_val) * lons,
#                         (start_lonlat[1] - min_lat_val) / (max_lat_val - min_lat_val) * lats)
            # start_idx = clip(start_idx, (0, 0), (lons, lats))

#            end_idx = ((end_lonlat[0] - min_lon_val) / (max_lon_val - min_lon_val) * lons,
#                       (end_lonlat[1] - min_lat_val) / (max_lat_val - min_lat_val) * lats)
            # end_idx = clip(end_idx, (0, 0), (lons, lats))

            rr, cc = line(int(start_idx_lon), int(start_idx_lat), int(end_idx_lon), int(end_idx_lat))
            rr = clip(rr, 0, lons - 1)
            cc = clip(cc, 0, lats - 1)

            temp_da.values[cc, rr] = True
        
        # for temp_da get CVs
        wt_where = data_chunk.cv.where(temp_da)
        
        if not np.isnan(wt_where).all():
            val = np.nanmean(wt_where.data)
            obj_props.avg_cv_along_wt = val

# identify troughs in data (should contain U,V,cv), based on the cv climatology
# def identify_troughs(data, cv_clim, cfg):

# with id_ to be created
def create_fake_wt_edges(edge, child_idx, id_, pb_ref):
    # make multiple edges out of this one
    
    parent_node = edge.parent
    assert isinstance(parent_node, Message)
    parent_node_pb = parent_node # simplenamespace_to_proto(parent_node, pb_ref.GraphNode())
    child_node = edge.children[child_idx]
    child_node_pb = child_node # simplenamespace_to_proto(child_node, pb_ref.GraphNode())

    parent_node_time = pb_str_to_datetime64(parent_node.time)
    child_node_time = pb_str_to_datetime64(child_node.time)

    delt = child_node_time - parent_node_time
    if delt != np.timedelta64(12, 'h'):
        print("No 12h diff, abort")
        print(delt)
        exit(1)

    fake_node_time = parent_node_time + 0.5 * delt # based on above assertion

    # create fake node as copy of parent
    fake_node = pb_ref.GraphNode() # json_format.Parse(json.dumps(parent_json), pb_ref.GraphNode(), ignore_unknown_fields=False)
    fake_node.time = datetime64_to_pb_str(fake_node_time)

    # id and flag False
    fake_node.feature.id = id_
    fake_node.feature.flag = False

    # set properties: bb as mean of parent and child
    parent_props = parent_node.feature.properties
    child_props = child_node.feature.properties
    fake_props = fake_node.feature.properties

    fake_props.bb.min.lat = (parent_props.bb.min.lat + child_props.bb.min.lat) / 2.0
    fake_props.bb.max.lat = (parent_props.bb.max.lat + child_props.bb.max.lat) / 2.0
    fake_props.bb.min.lon = (parent_props.bb.min.lon + child_props.bb.min.lon) / 2.0
    fake_props.bb.max.lon = (parent_props.bb.max.lon + child_props.bb.max.lon) / 2.0

    avg_lon = (fake_props.bb.min.lon + fake_props.bb.max.lon) / 2.0
    fake_props.line_pts.add()
    fake_props.line_pts[0].lat = fake_props.bb.min.lat
    fake_props.line_pts[0].lon = avg_lon
    fake_props.line_pts.add()
    fake_props.line_pts[1].lat = fake_props.bb.max.lat
    fake_props.line_pts[1].lon = avg_lon

    fake_props.length_deg = math.sqrt((fake_props.bb.max.lat - fake_props.bb.min.lat) ** 2 + (fake_props.bb.max.lon - fake_props.bb.min.lon) ** 2)

    # new edge:
    edge1 = pb_ref.GraphConnection()
    edge1.parent.CopyFrom(parent_node_pb)
    edge1.children.append(fake_node)

    edge2 = pb_ref.GraphConnection()
    edge2.parent.CopyFrom(fake_node)
    edge2.children.append(child_node_pb)

    return [edge1, edge2]


# interpolate wavetroughs, create fake WTs in skipped timesteps.
# cant be called parallelized cause of IDs
def interpolate_wts(data_desc, pb_ref):
    print("Add fake WTs...")
    
    next_id = -1
    added_edges = 0
    for set_ in data_desc.sets:

        for track in set_.tracks:
            # old_edges = []
            new_edges = []

            for edge in track.edges:
                parent_node = edge.parent
                parent_node_time = pb_str_to_datetime64(parent_node.time)
                if not hasattr(edge, 'children'):
                    continue

                for child_idx, child_node in enumerate(edge.children):
                    child_node_time = pb_str_to_datetime64(child_node.time)
                    
                    if child_node_time - parent_node_time > np.timedelta64(6, 'h'):
                        
                        wt_edges = create_fake_wt_edges(edge, child_idx, next_id, pb_ref)
                        next_id = next_id - 1
                        # old_edges.append((edge, child_idx))
                        new_edges.extend(wt_edges)


            # remove old_edges from this track and from graph
            added_edges += len(new_edges)
            
            new_edges_sn = [e for e in new_edges] # proto_to_simplenamespace(e)
            track.edges.extend(new_edges_sn)
            track.edges.sort(key=lambda item: item.parent.time)

            set_.graph.edges.extend(new_edges_sn)
            set_.graph.edges.sort(key=lambda item: item.parent.time)

    print("Added " + str(added_edges) + " nodes.")
    return data_desc
    
            
def add_wts_to_ds(dataset, data_desc):
    print("Generate netCDF lines...")
    
    lon_str = get_longitude_dim(dataset)
    lat_str = get_latitude_dim(dataset)
    lev_str = get_vertical_dim(dataset)
    u_str = get_u_var(dataset)
    v_str = get_v_var(dataset)
    
    if lev_str is not None and len(dataset[lev_str].values) > 1:
        dataset['wavetroughs'] = xr.zeros_like(dataset[u_str].isel(level=0).squeeze(), dtype=int) # all WTs
        dataset['tracks'] = xr.zeros_like(dataset[u_str].isel(level=0).squeeze(), dtype=int) # filtered by track heuristics
        dataset['tracks_id'] = xr.zeros_like(dataset[u_str].isel(level=0).squeeze(), dtype=int) # filtered by track heuristics
    elif lev_str is not None and len(dataset[lev_str].values) == 1:
        dataset['wavetroughs'] = xr.zeros_like(dataset[u_str].squeeze(), dtype=int) # all WTs
        dataset['tracks'] = xr.zeros_like(dataset[u_str].squeeze(), dtype=int) # filtered by track heuristics
        dataset['tracks_id'] = xr.zeros_like(dataset[u_str].squeeze(), dtype=int) # filtered by track heuristics
    else:
        dataset['wavetroughs'] = xr.zeros_like(dataset[u_str], dtype=int) # all WTs
        dataset['tracks'] = xr.zeros_like(dataset[u_str], dtype=int) # filtered by track heuristics
        dataset['tracks_id'] = xr.zeros_like(dataset[u_str], dtype=int) # filtered by track heuristics
    
    min_lat = dataset[lat_str].data.min()
    max_lat = dataset[lat_str].data.max()
    min_lon = dataset[lon_str].data.min()
    max_lon = dataset[lon_str].data.max()
    lons = len(dataset[lon_str].data)
    lats = len(dataset[lat_str].data)

    wt = dataset.wavetroughs
    wt_t = dataset.tracks
    wt_tid = dataset.tracks_id

    for wt_set in data_desc.sets:

        cur_set = wt_set

        # if use_fc:
        #     initTime = wt_set.initTime
        #     set_ds = dataset.sel(time=initTime)  # init time
        set_ds = dataset
        
        # get nodes from all tracks in set
        set_nodes = []
        set_trackids = []
        for track_id, track in enumerate(cur_set.tracks):
            set_nodes.extend([e.parent for e in track.edges])
            set_trackids.extend([track_id + 1 for e in track.edges])
        
        # put them into buckets
        node_buckets = defaultdict(list)
        trackid_buckets = defaultdict(list)
        for nid, x in enumerate(set_nodes):
            node_buckets[x.time].append(x)
            trackid_buckets[x.time].append(set_trackids[nid])
        
        # iterate buckets
        for time, cur_nodes in node_buckets.items():

            try:
                wt_t_da = wt_t.sel(time=time)
                wt_tid_da = wt_tid.sel(time=time)
            except KeyError:
                print("Skipping timestep (not in dataset) " + str(vt))
                continue
                
            for nodeid, node in enumerate(cur_nodes):
                cur_track_id = trackid_buckets[time][nodeid]
                props = node.feature.properties
                if not hasattr(props, 'line_pts'):
                    print("?")

                for v_idx in range(len(props.line_pts) - 1):
                    start_lonlat = props.line_pts[v_idx].lon, props.line_pts[v_idx].lat
                    end_lonlat = props.line_pts[v_idx + 1].lon, props.line_pts[v_idx + 1].lat

                    start_idx = ((start_lonlat[0] - min_lon) / (max_lon - min_lon) * lons,
                                 (start_lonlat[1] - min_lat) / (max_lat - min_lat) * lats)
                    # start_idx = clip(start_idx, (0, 0), (lons, lats))

                    end_idx = ((end_lonlat[0] - min_lon) / (max_lon - min_lon) * lons,
                               (end_lonlat[1] - min_lat) / (max_lat - min_lat) * lats)
                    # end_idx = clip(end_idx, (0, 0), (lons, lats))

                    rr, cc = line(int(start_idx[0]), int(start_idx[1]), int(end_idx[0]), int(end_idx[1]))
                    rr = clip(rr, 0, lons - 1)
                    cc = clip(cc, 0, lats - 1)

                    wt_t_da.values[cc, rr] = node.feature.id
                    wt_tid_da.values[cc, rr] = cur_track_id
                    
                    
                    """
                    # make circle
                    for px_idx in range(len(rr)):

                        circle = circles.isel(longitude_center=rr[px_idx], latitude_center=cc[px_idx])
                        influence_area = circle.where(circle < d, -1)

                        # update influence area dataarray
                        infl_da = xr.where(influence_area >= 0, node.feature.id, infl_da)
                    """
            wt_t.loc[dict(time=time)] = wt_t_da.values
                
                
        ### ALL NODES
        graph = cur_set.graph
        graph_nodes = [e.parent for e in graph.edges]
        
        # put them into buckets
        node_buckets = defaultdict(list)
        for x in graph_nodes:
            node_buckets[x.time].append(x)

        # iterate buckets
        for time, cur_nodes in node_buckets.items():
            try:
                wt_da = wt.sel(time=time)
            except KeyError:
                print("Skipping timestep (not in dataset) " + str(vt))
                continue
                
            for node in cur_nodes:
                props = node.feature.properties
                if not hasattr(props, 'line_pts'):
                    print("?")

                for v_idx in range(len(props.line_pts) - 1):
                    start_lonlat = props.line_pts[v_idx].lon, props.line_pts[v_idx].lat
                    end_lonlat = props.line_pts[v_idx + 1].lon, props.line_pts[v_idx + 1].lat

                    start_idx = ((start_lonlat[0] - min_lon) / (max_lon - min_lon) * lons,
                                 (start_lonlat[1] - min_lat) / (max_lat - min_lat) * lats)
                    # start_idx = clip(start_idx, (0, 0), (lons, lats))

                    end_idx = ((end_lonlat[0] - min_lon) / (max_lon - min_lon) * lons,
                               (end_lonlat[1] - min_lat) / (max_lat - min_lat) * lats)
                    # end_idx = clip(end_idx, (0, 0), (lons, lats))

                    rr, cc = line(int(start_idx[0]), int(start_idx[1]), int(end_idx[0]), int(end_idx[1]))
                    rr = clip(rr, 0, lons - 1)
                    cc = clip(cc, 0, lats - 1)

                    wt_da.values[cc, rr] = node.feature.id
                    
                    """
                    # make circle
                    for px_idx in range(len(rr)):

                        circle = circles.isel(longitude_center=rr[px_idx], latitude_center=cc[px_idx])
                        influence_area = circle.where(circle < d, -1)

                        # update influence area dataarray
                        infl_da = xr.where(influence_area >= 0, node.feature.id, infl_da)
                    """
            wt.loc[dict(time=time)] = wt_da.values
        
    return dataset
    