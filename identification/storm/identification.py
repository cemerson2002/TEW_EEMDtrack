from ..identification import IdentificationStrategy

import xarray as xr
from random import randrange
from scipy import ndimage as ndi
import numpy as np
from feature.identification.storm.util import *


class StormIdentification(IdentificationStrategy):

    def __init__(self, start_pressure_thr=965, step_pressure_thr=2, size_thr=10000, amplitude_thr=2, **kwargs):
        # Constructor. Called from example_template.py, parameters can be passed and set here.

        self.start_pressure_thr = start_pressure_thr
        self.step_pressure_thr = step_pressure_thr
        self.size_thr = size_thr
        self.amplitude_thr = amplitude_thr

        pass

    def precompute(self, dataset: xr.Dataset, **kwargs):
        plt.switch_backend('agg')  # this is thread safe matplotlib but cant display.

        dataset['storm_areas'] = xr.zeros_like(dataset['msl'], dtype=int)
        return dataset

    def identify(self, dataset: xr.Dataset, index_accessor, **kwargs):

        msl_data = dataset.msl
        considered_mask = xr.zeros_like(msl_data, dtype=bool)

        pressure_thrs = np.arange(self.start_pressure_thr, 1000, self.step_pressure_thr)

        obj_list = []
        for pressure_thr in pressure_thrs:

            lp_area = get_msl_areas_by_hPa_thr(msl_data, pressure_thr)

            storm_areas = split_areas(lp_area)

            for storm_area in storm_areas:

                if np.any(np.logical_and(considered_mask.data, storm_area.data)):
                    continue

                storm_size = get_storm_size_km2(storm_area)
                if storm_size < self.size_thr:
                    continue

                if not has_local_minimum(storm_area, msl_data):
                    continue

                storm_amplitude = get_storm_amplitude_hPa(storm_area, msl_data)
                if storm_amplitude < self.amplitude_thr:
                    continue

                # get an instance of a new feature, can pass an ID or set in manually afterwards
                obj = self.get_new_feature(index_accessor)
                # set some ID to it
                storm_id = obj.id
                
                # keep storm
                dataset['storm_areas'].data[storm_area] = storm_id
                considered_mask.data[storm_area] = True

                # get properties of feature and populate them (like defined in template.proto)
                properties = obj.properties

                properties.min_pressure, lon, lat = get_min_pressure_hPa(storm_area, msl_data, return_pos=True)
                properties.min_pressure_pos.lat = lat
                properties.min_pressure_pos.lon = lon

                obj_list.append(obj)
                storm_id += 1

        # return the dataset (can be changed here), and the list of features
        return dataset, obj_list

    def postprocess(self, dataset: xr.Dataset, obj_desc, **kwargs):

        return dataset, obj_desc
