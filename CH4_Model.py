import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict

# Define the unified target resolution
TARGET_RESOLUTION = 0.5

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None) 

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
        new_lat = np.arange(-90 + TARGET_RESOLUTION, 90 + TARGET_RESOLUTION, TARGET_RESOLUTION)
        new_lon = np.arange(-180, 180, TARGET_RESOLUTION)
        
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
              time_range: Optional[Tuple[str, str]] = None, 
              target: Optional[list] = None) -> Optional[pd.DataFrame]:
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
            df = time_selection.to_dataframe()
            if target and all(item in df.columns for item in target):
                # Drop NaN rows
                df = df.dropna(subset=target).reset_index()
            # default: no droping, remain all data. here, the dropna is used to drop the contents outside the time range
            else: # replace NaN
                df = df.dropna(how='all').reset_index()

            if df.empty:
                print(df)
                print(f"Locations from lan {lat_range} to lon {lon_range} within time range {time_range} no data exist!")
            else:
                # print(df)
                return df

        except Exception as e:
            print(f"ERROR: An error occurred while querying model '{self.name}': {e}")
            return None
        
    
    def _to_nc_file(self, source: pd.DataFrame, output_path: str):
        """Transform dataframe file into nc file.
        Args:
            source (pd.DataFrame): dataframe source.
            output_path (str): The path to the nc file.

        Returns:
            None."""
        ds = source.to_xarray()
        if isinstance(output_path, Path):
            path = self.path
        else:
            path = Path(output_path)

        ds.to_netcdf(output_path)
        ds_read = xr.open_dataset(output_path)
        if ds_read:
            print(f'Source file is exported to {output_path} successfully!')
            ds_read.close()
        else:
            print("Export to nc file error!")
            exit(-1)

    @staticmethod
    def merge_models_to_nc(results: Dict[str, pd.DataFrame], output_path: str):
        """
        Merge multiple model query results into a single NetCDF file.
        
        Args:
            results (Dict[str, pd.DataFrame]): Dictionary mapping model names to their query results
            output_path (str): Path for the output NetCDF file
            
        Returns:
            None
        """
        if not results or all(df is None for df in results.values()):
            print("No valid data to export")
            return
        
        if len(results) == 1:
            CH4_Model._to_nc_file(CH4_Model, list(results.values())[0], output_path)
            return
        # Filter out None results
        valid_results = {name: df for name, df in results.items() if df is not None and not df.empty}
        
        if not valid_results:
            print("No valid data to export")
            return
            
        print(f"\n=== Converting {len(valid_results)} models to {output_path} ===")
        
        merged_data = {}
        model_info = {}
        
        for model_name, df in valid_results.items():
            print(f"Processing {model_name} with {len(df)} records")
            
            # Add model name as a column
            df_with_model = df.copy()
            df_with_model['model_name'] = model_name
            
            coord_cols = ['lat', 'lon', 'year', 'month', 'day', 'model_name']
            data_cols = [col for col in df_with_model.columns if col not in coord_cols]
            
            # Rename data columns with model prefix
            rename_dict = {col: f"{model_name}_{col}" for col in data_cols}
            df_with_model = df_with_model.rename(columns=rename_dict)
            
            # Store model information
            model_info[model_name] = {
                'record_count': len(df_with_model),
                'variables': list(rename_dict.values()),
                'original_variables': data_cols,
                'time_range': (df['year'].min(), df['year'].max()) if 'year' in df.columns else None,
                'spatial_range': {
                    'lat': (df['lat'].min(), df['lat'].max()) if 'lat' in df.columns else None,
                    'lon': (df['lon'].min(), df['lon'].max()) if 'lon' in df.columns else None
                }
            }
            
            merged_data[model_name] = df_with_model
        
        # Combine all dataframes
        all_dfs = list(merged_data.values())
        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        
        # Fill NaN values with -9999 for consistency
        combined_df = combined_df.fillna(-9999)
        
        try:
            # Handle mixed data types by converting to string where necessary
            for col in combined_df.columns:
                if combined_df[col].dtype == 'object':
                    # Check if it's a mixed type column
                    if combined_df[col].apply(lambda x: isinstance(x, (int, float))).any() and \
                       combined_df[col].apply(lambda x: isinstance(x, str)).any():
                        # Convert all to string to avoid mixed type issues
                        combined_df[col] = combined_df[col].astype(str)
            
            # Set index for proper xarray conversion
            index_cols = ['lat', 'lon', 'year', 'month', 'day', 'model_name']
            existing_index_cols = [col for col in index_cols if col in combined_df.columns]
            
            if existing_index_cols:
                combined_df_indexed = combined_df.set_index(existing_index_cols)
                ds = combined_df_indexed.to_xarray()
            else:
                # If no index columns or other errors, create a simple unique index
                combined_df_indexed = combined_df.reset_index(drop=True)
                combined_df_indexed['unique_id'] = range(len(combined_df_indexed))
                combined_df_indexed = combined_df_indexed.set_index('unique_id')
                ds = combined_df_indexed.to_xarray()
            
            # Add global attributes
            ds.attrs['title'] = 'CH4 Model Data Query Results'
            ds.attrs['description'] = f'Combined results from {len(valid_results)} models'
            ds.attrs['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ds.attrs['models_included'] = ', '.join(valid_results.keys())
            
            # Add model-specific attributes
            for model_name, info in model_info.items():
                ds.attrs[f'model_{model_name}_records'] = info['record_count']
                ds.attrs[f'model_{model_name}_variables'] = ', '.join(info['original_variables'])
                if info['time_range']:
                    ds.attrs[f'model_{model_name}_time_range'] = f"{info['time_range'][0]}-{info['time_range'][1]}"
                if info['spatial_range']['lat']:
                    ds.attrs[f'model_{model_name}_lat_range'] = f"{info['spatial_range']['lat'][0]:.2f} to {info['spatial_range']['lat'][1]:.2f}"
                if info['spatial_range']['lon']:
                    ds.attrs[f'model_{model_name}_lon_range'] = f"{info['spatial_range']['lon'][0]:.2f} to {info['spatial_range']['lon'][1]:.2f}"
            
            # Add variable attributes
            for var_name in ds.data_vars:
                if var_name.startswith(tuple(valid_results.keys())):
                    model_name = var_name.split('_')[0]
                    original_var = '_'.join(var_name.split('_')[1:])
                    ds[var_name].attrs['model'] = model_name
                    ds[var_name].attrs['original_variable'] = original_var
                    ds[var_name].attrs['description'] = f'{original_var} from {model_name} model'
            
            ds.to_netcdf(output_path)
            
            # Verification
            try:
                test_ds = xr.open_dataset(output_path)
                print(f"Successfully exported data to {output_path}")
                print(f"  - Total records: {len(combined_df)}")
                print(f"  - Models included: {', '.join(valid_results.keys())}")
                test_ds.close()
            except Exception as e:
                print(f"Error verifying output file: {e}")
                
        except Exception as e:
            print(f" Error creating NetCDF file: {e}")
            return


    def __repr__(self):
        """Returns a string representation of the object."""
        if self.dataset is not None:
            # Handle both old and new xarray versions
            try:
                dims = ", ".join([f"{k}: {v}" for k, v in self.dataset.dims.items()])
            except (TypeError, AttributeError):
                # For newer xarray versions, use sizes
                dims = ", ".join([f"{k}: {v}" for k, v in self.dataset.sizes.items()])
            return f"<CH4_Model name='{self.name}', resolution={self.original_resolution}°, data_dims=({dims})>"
        return f"<CH4_Model name='{self.name}', resolution={self.original_resolution}° (Data not loaded)>"
