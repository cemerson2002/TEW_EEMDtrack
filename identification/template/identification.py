from ..identification import IdentificationStrategy

import xarray as xr
from random import randrange


class IdentificationTemplate(IdentificationStrategy):

    def __init__(self, some_parameter='', **kwargs):
        # Constructor. Called from example_template.py, parameters can be passed and set here.
        self.param = some_parameter

        pass

    def precompute(self, dataset: xr.Dataset, **kwargs):
        # Precomputation, done once for the entire dataset.
        # All work which requires or takes the entire input can be performed here, identify() later is called in
        #   parallel for all timesteps, ensemble members, ...

        # Also data fields need to be added here in case they are needed and populated later on.
        # e.g. dataset['identified_areas'] = xr.zeros_like(dataset['existing_field'], dtype=bool)
        return dataset

    def identify(self, dataset: xr.Dataset, index_accessor, **kwargs):
        # Identification, here is the place for your identification algorithm.
        # This is called in parallel for each spatial subset of the data, e.g., for each member and for each timestep
        # Therefore, dataset contains only spatial dimensions (lat,lon) if self.processing_mode = '2d' is set,
        #   (see __init__), or (lat,lon,level) if '3d' is set.

        # here your code ... identify features for this subset.

        # Let's say you detected 5 features:
        obj_list = []

        for i in range(5):
            # get an instance of a new feature, can pass an ID or set in manually afterwards
            obj = self.get_new_feature(index_accessor)
            id_ = obj.id

            # get properties of feature and populate them (like defined in template.proto)
            properties = obj.properties

            properties.size = 42
            properties.centroid.x = randrange(4)
            properties.centroid.y = 2 * id_
            properties.centroid.z = 3 * id_

            properties.list_of_something.append("type x")
            properties.list_of_something.append("type z")

            obj_list.append(obj)

        # return the dataset (can be changed here), and the list of features
        return dataset, obj_list

    def postprocess(self, dataset: xr.Dataset, obj_desc, **kwargs):
        # Here post process: called once after processing the whole dataset.
        # If needed, the dataset or features can be changed here.

        # features are already in finished protobuf description, cf. the JSON output. pb2_desc.sets is list of
        # trackable sets, each one contains the timesteps list, where each entry is a valid time and contains a list
        # of features.

        return dataset, obj_desc
