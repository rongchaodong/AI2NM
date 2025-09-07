import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional

# Define the unified target resolution
TARGET_RESOLUTION = 0.5

class CH4_Model:
    """
    A generic base class for handling methane data models from various sources.
    It standardizes the data into an xarray.Dataset, unifies coordinates,
    and resamples the spatial resolution to a target value.

    Subclasses must implement the _load_data method.
    """
    def __init__(self, name: str, path: str, resolution: Union[float, Tuple[float, float]] = 0.5):
        """
        Initializes the CH4_Model instance.

        Args:
            name (str): The name of the data model (e.g., "TEM", "Carbon-Tracker").
            path (str): The path to the data file or directory.
            resolution (Union[float, Tuple[float, float]]): The original resolution(s) in degrees.
                                                             Use a float for square pixels (e.g., 0.5) or a
                                                             tuple for non-square pixels (e.g., (4, 5) for lat, lon).
                                                             default would be 0.5 degree.
        """
        self.name = name
        self.path = Path(path)
        
        if isinstance(resolution, (int, float)):
            self.original_resolution = (resolution, resolution)
        else:
            self.original_resolution = resolution # Expects (lat_res, lon_res)
            
        self.dataset = None
        
        # This calls the implementation from the specific subclass
        self._load_data()
        
        # Standard processing steps after data is loaded
        if self.dataset is not None:
            self._standardize_dataset()

    def _load_data(self):
        """
        Abstract method for loading data from the source path.
        This method MUST be overridden by any subclass.
        It should load the data from self.path and assign it to self.dataset as an xarray.Dataset.
        """
        raise NotImplementedError("Subclasses must implement the `_load_data` method.")

    def _standardize_dataset(self):
        """
        Standardizes the dataset by unifying coordinate names and resolution.
        """
        # 1. Unify coordinate names
        rename_map = {
            'latitude': 'lat',
            'longitude': 'lon',
            'Latitude': 'lat',
            'Longitude': 'lon',
            'Lat': 'lat',
            'Lon': 'lon',
        }
        actual_rename_map = {k: v for k, v in rename_map.items() if k in self.dataset.coords}
        if actual_rename_map:
            self.dataset = self.dataset.rename(actual_rename_map)

        # 2. Ensure latitude is in ascending order (-90 to 90)
        if 'lat' in self.dataset.coords and self.dataset.lat.values[0] > self.dataset.lat.values[-1]:
            self.dataset = self.dataset.reindex(lat=self.dataset.lat[::-1])

        # 3. Resample if the resolution does not match the target
        # Check both lat and lon resolution
        if self.original_resolution[0] != TARGET_RESOLUTION or self.original_resolution[1] != TARGET_RESOLUTION:
            print(f"INFO: Model '{self.name}' has resolution {self.original_resolution}, resampling to {TARGET_RESOLUTION}°.")
            self._resample_to_target()

    def _resample_to_target(self):
        """
        Resamples the dataset to the target resolution using the 'nearest' method.
        This will fill blocks of the new grid with the value of the nearest point in the old grid.
        """
        new_lat = np.arange(-90 + TARGET_RESOLUTION, 90, TARGET_RESOLUTION)
        new_lon = np.arange(-180, 179.5, TARGET_RESOLUTION)
        
        # Use reindex with 'nearest' method for block-filling instead of interpolation
        self.dataset = self.dataset.reindex(
            lat=new_lat, 
            lon=new_lon, 
            method='nearest', 
            tolerance=max(self.original_resolution) # Look for nearest point within a radius of the old resolution
        )

    def query(self, 
              lat_range: Tuple[float, float], 
              lon_range: Tuple[float, float], 
              time_range: Optional[Tuple[str, str]] = None) -> Optional[pd.DataFrame]:
        """
        Queries data within a given latitude, longitude, and optional time range.

        Args:
            lat_range (tuple): Latitude range (min_lat, max_lat).
            lon_range (tuple): Longitude range (min_lon, max_lon).
            time_range (tuple, optional): Time range (start_date, end_date) as strings
                                          (e.g., ("2000-05", "2001-06")). Defaults to None.

        Returns:
            pandas.DataFrame: A DataFrame containing the query results, or None if no data is found.
        """
        if self.dataset is None:
            print("Error: cannot query, Dataset ", self.name, "is not loaded!")
            return None

        min_lat, max_lat = lat_range
        if not -90 <= min_lat <= 90 or not -90 <= max_lat <= 90 or min_lat >= max_lat:
            print(f"ERROR: Invalid latitude range. Must be within [-90, 90] and min_lat < max_lat. Got: {lat_range}")
            return None

        min_lon, max_lon = lon_range
        if not -180 <= min_lon <= 180 or not -180 <= max_lon <= 180 or min_lon >= max_lon:
            print(f"ERROR: Invalid longitude range. Must be within [-180, 180] and min_lon < max_lon. Got: {lon_range}")
            return None

        try:
            spatial_selection = self.dataset.sel(
                lat=slice(lat_range[0], lat_range[1]),
                lon=slice(lon_range[0], lon_range[1])
            )

            # print("spatial:", spatial_selection)
            # exit()
            
            if not time_range:
                print("ERROR: Must specify a time range to query, for example: (\"2000-01\", \"2000-06\").")
                return None

            start_date = pd.to_datetime(time_range[0])
            end_date = pd.to_datetime(time_range[1])
            if start_date >= end_date:
                print(f"ERROR: Invalid time range. Start date must be before end date. Got: {time_range}")
                return None

            ds = spatial_selection
            # if the start and end is same year
            if start_date.year == end_date.year:
                time_mask = (ds.year == start_date.year) & \
                            (ds.month >= start_date.month) & \
                            (ds.month <= end_date.month)
            else: # The time range spans across multiple years
                start_year_mask = (ds.year == start_date.year) & (ds.month >= start_date.month)
                end_year_mask = (ds.year == end_date.year) & (ds.month <= end_date.month)
                intermediate_years_mask = (ds.year > start_date.year) & (ds.year < end_date.year)
                time_mask = start_year_mask | end_year_mask | intermediate_years_mask

            # drop=True removes all data points that don't meet the condition
            time_selection = ds.where(time_mask, drop=True)
            # print("try:", time_selection)
            # df = time_selection.to_dataframe().dropna(how='all').reset_index()
            df = time_selection.to_dataframe().reset_index()

            if df.empty:
                print(df)
                print(f"Locations from lan {lat_range} to lon {lon_range} within time range {time_range} no data exist!")
            else:
                # print(df)
                return df

        except Exception as e:
            print(f"ERROR: An error occurred while querying model '{self.name}': {e}")
            return None


    def __repr__(self):
        """Returns a string representation of the object."""
        if self.dataset is not None:
            dims = ", ".join([f"{k}: {len(v)}" for k, v in self.dataset.dims.items()])
            return f"<CH4_Model name='{self.name}', resolution={self.original_resolution}°, data_dims=({dims})>"
        return f"<CH4_Model name='{self.name}', resolution={self.original_resolution}° (Data not loaded)>"
