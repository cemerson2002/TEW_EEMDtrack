from ..identification import IdentificationStrategy

import xarray as xr
import numpy as np
from random import randrange
import datetime


class OverlapIdentificationExample(IdentificationStrategy):

    def __init__(self, some_parameter='', **kwargs):
        # Constructor. Called from example_template.py, parameters can be passed and set here.
        self.param = some_parameter

        pass

    def precompute(self, dataset: xr.Dataset, **kwargs):
        # Precomputation, done once for the entire dataset.
        # All work which requires or takes the entire input can be performed here, identify() later is called in
        #   parallel for all timesteps, ensemble members, ...

        # Also data fields need to be added here in case they are needed and populated later on.
        dataset['identified_areas'] = xr.zeros_like(dataset['cv'], dtype=int)
        return dataset

    def identify(self, dataset: xr.Dataset, index_accessor, **kwargs):
        # Identification, here is the place for your identification algorithm.
        # This is called in parallel for each spatial subset of the data, e.g., for each member and for each timestep
        # Therefore, dataset contains only spatial dimensions (lat,lon) if self.processing_mode = '2d' is set,
        #   (see __init__), or (lat,lon,level) if '3d' is set.

        # Let's say you detected 5 features:
        obj_list = []

        cur_time_item = dataset.time.data[0].item()
        dt = datetime.datetime.utcfromtimestamp(cur_time_item / 1e9)

        identified_areas = dataset.identified_areas.isel(plev=0)

        for i in range(5):
            # get an instance of a new feature, can pass an ID or set in manually afterwards
            # set some ID to it
            obj = self.get_new_feature(index_accessor)
            obj_id = obj.id

            # In this example, just mark some areas of the field corresponding to this feature.
            # So the field has 1s later at positions corresponding to feature id 1, and so on
            if dt.hour == 0:
                identified_areas[10:15, 10:15] = obj_id
            elif dt.hour == 6:
                identified_areas[13:18, 13:18] = obj_id
            elif dt.hour == 12:
                identified_areas[16:21, 16:21] = obj_id
            elif dt.hour == 18:
                identified_areas[19:24, 19:24] = obj_id

            # get properties of feature and populate them (like defined in template.proto)
            properties = obj.properties
            
            # if we also want to add the voxel data to the object description
            # VERY VERBOSE!
            float_mask = xr.where(identified_areas == obj_id, 42.0, np.nan)
            self.add_voxel_rep(obj, float_mask)

            properties.size = 42
            properties.centroid.x = randrange(10)
            properties.centroid.y = 2 * obj_id
            properties.centroid.z = 3 * obj_id

            properties.list_of_something.append("type x")
            properties.list_of_something.append("type z")

            obj_list.append(obj)

        # return the dataset (can be changed here), and the list of features
        return dataset, obj_list

    def postprocess(self, dataset: xr.Dataset, obj_desc, **kwargs):
        # Here post process: called once after processing the whole dataset.
        # If needed, the dataset or features can be changed here.

        return dataset, obj_desc
