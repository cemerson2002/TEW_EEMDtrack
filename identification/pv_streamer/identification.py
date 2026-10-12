from feature.identification.pv_streamer.processing import *
from feature.identification import IdentificationStrategy
import xarray as xr
import numpy as np
import copy
from feature.identification.pv_streamer.feature_desc import get_feature_data
from feature.identification.pv_streamer.projection_cdo import project_latlon_to_stereo, project_stereo_to_latlon


class PVIdentification(IdentificationStrategy):

    def __init__(self, unit='pv', mode_2d_layer=None, theta_range=None, extract_containing=None, centroid_lat_thr=None,
                 out_type='stereo', **kwargs):
        """
        Initialize the PV Identification.

        Parameters (experimental)
        ----------
        unit
            pv data unit: 'pv' or 'pvu'
        mode_2d_layer
            2d identification for given layer in dataset
        theta_range
            slice this range out of dataset to process
        extract_containing
            extract features contains the selected layer (K) or overlaps with selected layer-range
        centroid_lat_thr
            only keep if centroid is north of this threshold
        out_type
            stereo or ll for latlon output
        kwargs
        """

        self.config = None  # config
        self.distance_map = None  # precomputed distance map, reduced from 8 directions to one due to conformal projection
        self.area_map = None  # precomputed area map (km^2 per stereographic pixel)

        # put custom args
        self.pv_data_unit = unit
        self.theta_range = theta_range
        self.sel_layer = extract_containing
        self.centroid_lat_thr = centroid_lat_thr  # centroid < thr degree north -> discard
        self.layer_2d_id = mode_2d_layer
        
        self.out_type = out_type
        pass

    def precompute(self, dataset: xr.Dataset, **kwargs):
        print("Precompute for PV identification...")

        # init filter dataset, remove stuff we do not need, makes processing faster
        from .data_util import compute_global_fields, filter_dataset
        dataset = filter_dataset(dataset, self.theta_range, self.layer_2d_id)

        # create configuration
        from .configuration import DetectionConfig
        self.config = DetectionConfig.config_from_dataset(dataset)
        self.config.pv_data_unit = self.pv_data_unit
        self.config.sel_layer = self.sel_layer
        self.config.centroid_lat_thr = self.centroid_lat_thr

        pv = dataset[self.config.pv_dim]
        # make invalid pole points valid
        pv_tmp = xr.where(np.logical_and(pv[self.config.latitude_dim] == 90, pv <= -999), 10, pv)
        dataset[self.config.pv_dim].values = pv_tmp.transpose(*pv.dims).values
        
        # if we go later back to latlon, create grid desc.
        if self.out_type == 'll':
            import tempfile
            with tempfile.NamedTemporaryFile(mode = "w", delete=False) as tmp:
                self.tmp_gf_name = tmp.name
                gf = ('gridtype = lonlat' + "\nxsize=" + str(self.config.lon_indices_len) +
                  "\nysize=" + str(self.config.lat_indices_len) + 
                  "\nxfirst=" + str(self.config.lon_min) + 
                  "\nyfirst=" + str(self.config.lat_min) + 
                  "\nxinc=" + str(self.config.res_lon) + 
                  "\nyinc=" + str(self.config.res_lat)
                 )
                tmp.write(gf)
        
        

        # transform dataset to stereographic
        stereo_ds = project_latlon_to_stereo(dataset)

        # computes pvu, pvu2
        stereo_ds = compute_global_fields(stereo_ds, self.config)

        from .projection_cdo import precompute_distance_map_simple, precompute_area_map_simple
        self.distance_map = precompute_distance_map_simple(stereo_ds)
        self.area_map = precompute_area_map_simple(self.distance_map)

        # if only 2D dataset, create third dimension for uniform handling
        from feature.util.enstools_utils import get_vertical_dim, add_vertical_dim
        if self.config.dims == 2:
            add_vertical_dim(stereo_ds, inplace=True)
            self.config.vertical_dim = get_vertical_dim(stereo_ds)

        # internal fields (resp. output fields)
        stereo_ds['streamer'] = xr.zeros_like(stereo_ds[self.config.pv_dim], dtype=int)
        # stereo_ds['inner_PV'] = xr.zeros_like(stereo_ds[self.config.pv_dim], dtype=float) a debug mode

        return stereo_ds

    # main identify, called on spatial 2d/3d subsets from feature.
    def identify(self, spatial_stereo_ds: xr.Dataset, index_accessor, **kwargs):
        levels_list = spatial_stereo_ds.coords[self.config.vertical_dim].values

        # preprocess binary fields: remove disturbances etc. in 2d, in 3d more flexible
        if spatial_stereo_ds['pvu2'].data.ndim == 3 and spatial_stereo_ds['pvu2'].data.shape[0] > 1:
            preprocessed_pvu2 = spatial_stereo_ds['pvu2'].data
        else:
            preprocessed_pvu2 = preprocess_pvu2(spatial_stereo_ds['pvu2'], self.config) # spatial_stereo_ds['pvu2'].data

        # create distance map from outer contour using precomputed distance functions
        dist_from_outer = create_dist_map_outer(preprocessed_pvu2, self.config, self.distance_map)

        core_reservoir = (dist_from_outer > self.config.alg.w_bar)

        # get data pole: maximum distance from boundary (center of reservoir)
        pv_pole = get_data_pole(dist_from_outer)

        # core_reservoir might have multiple areas: which one is main reservoir? flood biggest one (pole!)!
        # also compute mask containing cutoffs (isolated areas)
        # overwrite core_reservoir with only biggest one
        core_reservoir, cutoff_mask = flood_reservoir(core_reservoir, preprocessed_pvu2, pv_pole)

        if core_reservoir.ndim == 3 and core_reservoir.shape[0] > 1: # 3d keep top layer where growing can start
            core_reservoir[-1] = preprocessed_pvu2[-1]

        # ceate distance map from outer boundary of inner reservoir growing outwards, only in pvu>2 regions.
        dist_expand = create_dist_map_inner(core_reservoir, preprocessed_pvu2, self.config, self.distance_map)

        # set isolated areas as infinite distance from main reservoir
        dist_expand[cutoff_mask] = np.inf
        streamer_areas = (dist_expand > self.config.alg.w_bar * 1.05) # bit delta for subpixel inacc.
        del core_reservoir

        """
        from .plotting import plot_pvu2_binary, plot_erosion, plot_inner_binary, plot_dilation, plot_streamer_areas, \
            plot_pvu2_pp, plot_erosion_animate, plot_dilation_animate
        # plot_pvu2_binary(spatial_stereo_ds, self.config)
        # plot_pvu2_pp(spatial_stereo_ds, preprocessed_pvu2, self.config)
        # plot_erosion(spatial_stereo_ds, dist_from_outer, self.config)
        # plot_erosion_animate(spatial_stereo_ds, dist_from_outer, self.config)
        # plot_inner_binary(spatial_stereo_ds, core_reservoir, self.config)
        # plot_dilation(spatial_stereo_ds, dist_expand, self.config)
        # plot_dilation_animate(spatial_stereo_ds, dist_expand, self.config)
        # plot_streamer_areas(spatial_stereo_ds, dist_expand, streamer_areas_exact, self.config)
        # exit()
        """

        # filter area based. here also 3d filtering strategy
        streamer_areas = filter_area_based(streamer_areas, preprocessed_pvu2, self.config)  # data_filtering

        # label continuous 3d areas
        labeled_areas = label_areas(streamer_areas)
        del preprocessed_pvu2
        del streamer_areas

        # labeled_areas[~spatial_stereo_ds['pvu2']] = 0  # mask areas from holes closed before

        spatial_stereo_ds['streamer'].values = labeled_areas
        # generate feature descriptions

        feature_list, reshuffled_streamer_indices = get_feature_data(self, spatial_stereo_ds, levels_list, dist_expand, self.area_map, self.config, index_accessor)
        
        spatial_stereo_ds['streamer'].values = reshuffled_streamer_indices

        # filter feature descriptions
        spatial_stereo_ds, feature_list = filter_feature_based(spatial_stereo_ds, feature_list, self.config)
        # squish filtered IDs.
        # spatial_stereo_ds, feature_list = squish(spatial_stereo_ds, feature_list)

        # for id_, obj in feature_properties_with_corres_indices:
        #     featureProperties.add_to_features(feature_block, obj, id=id_)

        del dist_from_outer
        del dist_expand
        del cutoff_mask

        del labeled_areas
        return spatial_stereo_ds, feature_list

    def postprocess(self, dataset: xr.Dataset, pb2_desc, **kwargs):

        dropped = dataset.drop_vars(['pvu', 'pvu2', 'lat_field', 'lon_field'])
        
        # if 2d remove 3rd dim
        if self.config.dims == 2:
            pv = dropped[self.config.pv_dim]
            streamers = dropped['streamer']
            pv = pv.squeeze(dim=self.config.vertical_dim, drop=True)
            streamers = streamers.squeeze(dim=self.config.vertical_dim, drop=True)
            
            dropped[self.config.pv_dim] = pv
            dropped['streamer'] = streamers
        
        if self.out_type == 'll':
            dropped = project_stereo_to_latlon(dropped, self.tmp_gf_name)

        return dropped, pb2_desc
