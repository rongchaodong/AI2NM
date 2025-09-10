import xarray as xr
from CH4_Model import CH4_Model
import pandas as pd
import numpy as np
from CH4_Model import TARGET_RESOLUTION
from pathlib import Path

class Chamber_ABCFlux_Model(CH4_Model):
    
    def __init__(self, name: str, path: str, resolution: float = TARGET_RESOLUTION):
        self.metadata = None
        self.flux_data = None
        self.processed_data = None
        super().__init__(name, path, resolution)

    def _load_data(self):
        try:
            metadata_path = self.path / 'ABCFlux2_metadata.xlsx'
            flux_data_path = self.path / 'ABCFluxv2.ter.aq.csv'

            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            if not flux_data_path.exists():
                raise FileNotFoundError(f"Flux data file not found: {flux_data_path}")
            
            # Load metadata information
            self.metadata = pd.read_excel(metadata_path)
            # Load flux data
            self.flux_data = pd.read_csv(flux_data_path)

            self._process_data()
            
            # harmonize with simulated model data format, convert to gridded format
            self._convert_to_grid()
            
        except FileNotFoundError as e:
            print(f"WARNING: A required file was not found: {e}")
            self.dataset = None
        except Exception as e:
            print(f"ERROR: Failed to load Chamber_ABCFlux model from {self.path}: {e}")
            self.dataset = None

    def _process_data(self):
        """Process the ABCFlux data and prepare it for gridding."""
        if self.flux_data is None:
            print("No flux data to process")
            return
        
        # Create a copy of the original data
        processed_df = self.flux_data.copy()
        
        # Remove unwanted columns
        columns_to_remove = [
            'data_contributor_or_author', 'site_reference', 'email', 
            'extraction_source', 'citation', 'country'
        ]
        
        # Remove columns if they exist
        for col in columns_to_remove:
            if col in processed_df.columns:
                processed_df = processed_df.drop(columns=[col])
        
        # Remove the last column if it's a unique ID (assuming it's the last column)
        if len(processed_df.columns) > 0:
            last_col = processed_df.columns[-1]
            # Check if the last column looks like a unique ID (numeric, starting from 1)
            if processed_df[last_col].dtype in ['int64', 'int32', 'float64', 'float32']:
                if processed_df[last_col].min() == 1 and processed_df[last_col].max() == len(processed_df):
                    processed_df = processed_df.drop(columns=[last_col])
        
        # Rename columns to standard format
        column_mapping = {
            'latitude': 'lat',
            'longitude': 'lon', 
            'ch4_flux_total': 'emission'
        }
        
        # Apply column mapping for existing columns
        for old_name, new_name in column_mapping.items():
            if old_name in processed_df.columns:
                processed_df = processed_df.rename(columns={old_name: new_name})
        
        # Ensure we have the required columns
        required_columns = ['lat', 'lon', 'year', 'month']
        missing_columns = [col for col in required_columns if col not in processed_df.columns]
        
        if missing_columns:
            print(f"WARNING: Missing required columns: {missing_columns}")
            # Try to infer year and month from other columns if available
            if 'year' not in processed_df.columns and 'date' in processed_df.columns:
                processed_df['year'] = pd.to_datetime(processed_df['date']).dt.year
            if 'month' not in processed_df.columns and 'date' in processed_df.columns:
                processed_df['month'] = pd.to_datetime(processed_df['date']).dt.month
        
        # Remove rows with missing essential data
        processed_df = processed_df.dropna(subset=['lat', 'lon'])
        
        # Fill NaN values with -9999 for consistency with other models
        processed_df = processed_df.fillna(-9999)
        
        self.processed_data = processed_df
        print(f"Processed {len(self.processed_data)} records")

    def _convert_to_grid(self):
        if self.processed_data is None or len(self.processed_data) == 0:
            print("No data to convert to grid")
            return
        
        target_lats = np.arange(-90 + TARGET_RESOLUTION, 90 + TARGET_RESOLUTION, TARGET_RESOLUTION)
        target_lons = np.arange(-180, 180, TARGET_RESOLUTION)
        
        gridded_data = []
        
        for idx, row in self.processed_data.iterrows():
            # Find the nearest grid point
            lat_idx = np.argmin(np.abs(target_lats - row['lat']))
            lon_idx = np.argmin(np.abs(target_lons - row['lon']))
            
            nearest_lat = target_lats[lat_idx]
            nearest_lon = target_lons[lon_idx]
            
            # Create a record for this grid point with all available variables
            grid_record = {
                'lat': nearest_lat,
                'lon': nearest_lon,
                'year': row.get('year', -9999),
                'month': row.get('month', -9999)
            }
            
            # Add all other variables from the original data
            for col in self.processed_data.columns:
                if col not in ['lat', 'lon', 'year', 'month']:
                    grid_record[col] = row[col]
            
            gridded_data.append(grid_record)
        
        if gridded_data:
            gridded_df = pd.DataFrame(gridded_data)
            gridded_df = gridded_df.set_index(['lat', 'lon', 'year', 'month'])
            
            # Remove duplicated data (similar to FLUXNET implementation)
            # TODO: but it has problems, some of year and month are -9999 for same loation
            gridded_df = gridded_df[~gridded_df.index.duplicated(keep='first')]
            
            # Convert to xarray Dataset
            self.dataset = gridded_df.to_xarray()
            print(f"Converted to gridded format with {len(gridded_df)} unique grid points")
        else:
            print("No data found for grid conversion")
            self.dataset = None

    def get_metadata(self):
        """Get metadata information."""
        if self.metadata is None:
            print("Metadata not loaded")
            return None
        return self.metadata

    def get_flux_data(self):
        """Get original flux data."""
        if self.flux_data is None:
            print("Flux data not loaded")
            return None
        return self.flux_data

    def query(self, 
              lat_range: tuple, 
              lon_range: tuple, 
              time_range: tuple = None, 
              target: list = None) -> pd.DataFrame:
        """
        Query data with monthly resolution support.
        
        Args:
            lat_range (tuple): Latitude range (min_lat, max_lat)
            lon_range (tuple): Longitude range (min_lon, max_lon)
            time_range (tuple): Time range (start_date, end_date) as strings
            target (list): Target variables to query
            
        Returns:
            pandas.DataFrame: Query results
        """
        if self.dataset is None:
            print("Error: cannot query, Dataset", self.name, "is not loaded!")
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
            spatial_mask = (
                (self.dataset.lat >= lat_range[0]) & 
                (self.dataset.lat <= lat_range[1]) &
                (self.dataset.lon >= lon_range[0]) & 
                (self.dataset.lon <= lon_range[1])
            )
            
            spatial_selection = self.dataset.where(spatial_mask, drop=True)
            
            if not time_range:
                print("ERROR: Must specify a time range to query, for example: (\"2000-01\", \"2000-06\").")
                return None

            start_date = pd.to_datetime(time_range[0])
            end_date = pd.to_datetime(time_range[1])
            if start_date >= end_date:
                print(f"ERROR: Invalid time range. Start date must be before end date. Got: {time_range}")
                return None

            ds = spatial_selection
            
            # Create time mask for monthly data
            if start_date.year == end_date.year:
                # Same year
                time_mask = (ds.year == start_date.year) & \
                           (ds.month >= start_date.month) & \
                           (ds.month <= end_date.month)
            else:
                # Different years
                start_year_mask = (ds.year == start_date.year) & (ds.month >= start_date.month)
                end_year_mask = (ds.year == end_date.year) & (ds.month <= end_date.month)
                intermediate_years_mask = (ds.year > start_date.year) & (ds.year < end_date.year)
                time_mask = start_year_mask | end_year_mask | intermediate_years_mask

            time_selection = ds.where(time_mask, drop=True)
            df = time_selection.to_dataframe()
            
            if target and all(item in df.columns for item in target):
                # Drop NaN rows for target variables
                df = df.dropna(subset=target).reset_index()
            else:
                # Drop all NaN rows
                df = df.dropna(how='all').reset_index()

            if df.empty:
                print(f"Locations from lat {lat_range} to lon {lon_range} within time range {time_range} no data exist!")
                return None
            else:
                return df

        except Exception as e:
            print(f"ERROR: An error occurred while querying model '{self.name}': {e}")
            return None

if __name__ == '__main__':
    # Test Chamber_ABCFlux model
    model = Chamber_ABCFlux_Model("Chamber_ABCFlux", "../chamber/ABCFluxv2/")
    
    if model.dataset is not None:
        # print(model.dataset)
        # exit()
        result_monthly = model.query(
            lat_range=(70.0, 72.0), 
            lon_range=(-157, -155), 
            time_range=("2013/7", "2013/12")
        )
        
        if result_monthly is not None and not result_monthly.empty:
            print("Monthly query results:")
            print(result_monthly.head(10))
        else:
            print("No monthly data found for the query")
        
        print("\n=== Metadata Information ===")
        metadata = model.get_metadata()
        if metadata is not None:
            print(f"Metadata shape: {metadata.shape}")
            print("Metadata columns:")
            print(metadata.columns.tolist())
        
        print("\n=== Flux Data Information ===")
        flux_data = model.get_flux_data()
        if flux_data is not None:
            print(f"Flux data shape: {flux_data.shape}")
            print("Flux data columns:")
            print(flux_data.columns.tolist())
    else:
        print("Failed to load Chamber_ABCFlux model")