from feature.identification import IdentificationStrategy
from feature.tracking import TrackingStrategy
from feature.util.enstools_utils import get_vertical_dim

from datetime import datetime
import xarray as xr


class FeaturePipeline:
    """
    Feature detection pipeline (identification and tracking). This encapsulates
    the main pipeline, which will be executed by the user.

    Parameters
    ----------
    id_tech : IdentificationStrategy
        The identification technique to use
    tr_tech : TrackingStrategy
        The tracking technique to use
    pb_reference
        Compiled protobuf file for this description
    processing_mode : {'2d', '3d'}
        Specify if identification and tracking is performed on 2D levels or in
        3D, per 3D block.
    workers : int
        Number of parallel workers to use.
    """

    def __init__(self, proto_ref, processing_mode='2d', workers=4):
        """Creates a new pipeline.

        Parameters
        ----------
        proto_ref
            Reference to the compiled protobuf type, see templates for usage examples.
        processing_mode : '2d' or '3d'
            Specify if identification and tracking is performed on 2D levels or in
            3D, per 3D block.
        workers : int, optional
            Number of parallel workers to use

        """
        self.id_tech = None
        self.tr_tech = None

        self.processing_mode = processing_mode
        self.workers = workers

        self.feature_desc = None

        self.dataset = None
        self.dataset_path = None

        self.pb_reference = proto_ref

    def set_identification_strategy(self, strategy: IdentificationStrategy):
        """
        Set the strategy to use for the identification.

        Parameters
        ----------
        strategy : IdentificationStrategy
            The identification strategy to use in the pipeline.
        """
        self.id_tech = strategy
        self.id_tech.pb_reference = self.pb_reference
        self.id_tech.processing_mode = self.processing_mode
        self.id_tech.workers = self.workers

    def set_tracking_strategy(self, strategy: TrackingStrategy, generate_tracks=True):
        """
        Set the strategy to use for the tracking.

        Parameters
        ----------
        strategy : TrackingStrategy | None
            The tracking strategy to use in the pipeline. Set to `None` or
            don't invoke this method at all if no tracking should be carried
            out.
        generate_tracks: bool
            Whether tracks should be generated from the tracking graph after tracking
        """
        self.tr_tech = strategy
        if strategy is not None:
            self.tr_tech.pb_reference = self.pb_reference
            self.tr_tech.processing_mode = self.processing_mode
            self.tr_tech.workers = self.workers
            self.tr_tech.tracks_requested = generate_tracks

    def set_data_path(self, path):
        """
        Set the path to the dataset(s) to process.

        This function calls :py:func:`enstools.io.read` and therefore can read
        directories using wildcards.

        Parameters
        ----------
        path : list of str or tuple of str
                names of individual files or filename pattern

        See Also
        --------
        :py:meth:`.set_data`

        """
        if path is None:
            raise Exception("None path provided.")

        from enstools.io import read
        import xarray as xr
        self.dataset = read(path)
        self.dataset_path = path

    def set_data(self, dataset: xr.Dataset):
        """
        Set the dataset to process.

        Parameters
        ----------
        dataset : xr.Dataset
                the xarray Dataset
        """
        if dataset is None:
            raise Exception("None dataset provided.")

        self.dataset = dataset
        self.dataset_path = ""

    def execute_identification(self):
        """Execute only the identification strategy."""
        return_obj_desc_id, return_ds = self.id_tech.execute(self.dataset)
        self.feature_desc = return_obj_desc_id
        if return_ds is not None:
            self.dataset = return_ds

        self.feature_desc.file = str(self.dataset_path)
        self.feature_desc.run_time = str(datetime.now().isoformat())

    def execute_tracking(self):
        """Execute only the tracking strategy."""
        self.tr_tech.execute(self.feature_desc, self.dataset)
        

    def execute(self):
        """
        Execute the entire feature detection pipeline.

        See Also
        --------
        :py:meth:`.execute_identification`, :py:meth:`.execute_tracking`
        """

        self.execute_identification()
        if self.tr_tech is not None:
            self.execute_tracking()

    def get_feature_desc(self):
        return self.feature_desc

    def get_data(self):
        return self.dataset

    def is_data_3d(self):
        """
        Checks if the provided dataset is spatially 3D (has a vertical dim)

        Returns
        -------
        bool
            `True` if vertical dim in dataset else `False`.
        """
        if self.dataset is None:
            raise Exception("None dataset provided.")

        vd = get_vertical_dim(self.dataset)
        return vd is not None

    def get_features_as_json(self):
        """
        Get the JSON type message of the currently saved result.

        Returns
        -------
        JSON feature of identification/tracking result.
        """
        from google.protobuf.json_format import MessageToJson
        json_dataset = MessageToJson(self.feature_desc)
        return json_dataset

    def save_result(self, description_path=None, description_type='json', dataset_path=None, engine="netcdf4"):
        """
        Save the result of the detection process.

        Parameters
        ----------
        description_path : str
                Path to the file where the feature descriptions will be stored.
        description_type : {'json', 'binary'}
                Type of the descriptions, either in JSON or in a binary format. Default is JSON.
        dataset_path : str
                Path to the file where the (altered) dataset should be stored.
        engine : str
                The engine to use for saving the data set to netCDF. Gets forwarded to xarray's
                to_netcdf.
        """

        if description_path is not None:

            if description_type == 'binary':
                print("writing binary to " + description_path)
                with open(description_path, "wb") as file:
                    file.write(self.feature_desc.SerializeToString())

            elif description_type == 'json':
                from google.protobuf.json_format import MessageToJson
                print("writing json to " + description_path)
                with open(description_path, "w") as file:
                    json_dataset = MessageToJson(self.feature_desc)
                    file.write(json_dataset)

            else:
                print("Unknown type format, supported are 'binary' and 'json'.")

        if dataset_path is not None:
            # write netcdf dataset to path
            print("Writing netcdf to " + dataset_path)
            
            from enstools.io import write
            write(self.dataset, dataset_path, engine=engine)
