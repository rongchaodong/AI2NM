
import xarray as xr
from CH4_Model import CH4_Model
import pandas as pd
import numpy as np
from CH4_Model import TARGET_RESOLUTION
from pathlib import Path
from typing import Optional, Tuple, List

class FLUXNET_Model(CH4_Model):
    
    def __init__(self, name: str, path: str, resolution: float = TARGET_RESOLUTION):
        self.site_info = None
        self.flux_data = None
        self.processed_data = None
        super().__init__(name, path, resolution)

    def _load_data(self):
        try:
            site_info_path = self.path / 'FLUXNET_CH4_2024.csv'
            flux_data_path = self.path / 'FLUXNET_T1_DD.csv'
            
            if not site_info_path.exists():
                raise FileNotFoundError(f"Site info file not found: {site_info_path}")
            if not flux_data_path.exists():
                raise FileNotFoundError(f"Flux data file not found: {flux_data_path}")
            
            self.site_info = pd.read_csv(site_info_path)
            self.flux_data = pd.read_csv(flux_data_path)

            self._process_and_merge_data()
            
            # harmonize with simulated model data format, convert to gridded format
            # self._convert_to_grid()
            
        except FileNotFoundError as e:
            print(f"WARNING: A required file was not found: {e}")
            self.dataset = None
        except Exception as e:
            print(f"ERROR: Failed to load FLUXNET model from {self.path}: {e}")
            self.dataset = None

    def _process_and_merge_data(self):
        """Process and merge site info with flux data."""
        site_info_processed = self.site_info.copy()
        site_info_processed['SITE_ID'] = site_info_processed['SITE_ID'].str.replace('-', '.', regex=False)
        site_info_processed.rename(columns={'SITE_ID': 'Site'}, inplace=True)
        
        merged_df = pd.merge(self.flux_data, site_info_processed, on='Site', how='left')
        
        # Process timestamps
        merged_df['TIMESTAMP'] = pd.to_datetime(merged_df['TIMESTAMP'])
        merged_df['year'] = merged_df['TIMESTAMP'].dt.year
        merged_df['month'] = merged_df['TIMESTAMP'].dt.month
        merged_df['day'] = merged_df['TIMESTAMP'].dt.day
        
        # Remove leap year day (Feb 29) for consistency
        merged_df = merged_df[~((merged_df['month'] == 2) & (merged_df['day'] == 29))]
        
        # Create emission variable (use FCH4_F if available, otherwise FCH4_F_ANNOPTLM)
        merged_df['emission'] = merged_df['FCH4_F'].fillna(merged_df['FCH4_F_ANNOPTLM'])
        

        final_df = merged_df[['LOCATION_LAT', 'LOCATION_LONG', 'year', 'month', 'day', 
                                       'emission', 'Site', 'LOCATION_ELEV', 'MAT', 'MAP', 'NEE_F',
                                        'TA_F', 'VPD_F', 'P_F']].copy()
        final_df.rename(columns={'LOCATION_LAT': 'lat', 'LOCATION_LONG': 'lon', 'LOCATION_ELEV': 'ELEV'}, inplace=True)
        
        # reserve vaild data
        final_df = final_df.dropna(subset=['lat', 'lon', 'emission'])
        # same with other models, fill other NaN column with -9999
        final_df = final_df.fillna(-9999)
        self.processed_data = final_df
        # print(f"Final processed data: {len(self.processed_data)} records")
        # print(self.processed_data.head(10))
        # exit()

    def _convert_to_grid(self, df, lat_range, lon_range):
        if df is None or len(df) == 0:
            print("No data to convert to grid")
            return
        min_lat, max_lat = lat_range
        min_lon, max_lon = lon_range
        target_lats = np.arange(min_lat, max_lat + TARGET_RESOLUTION, TARGET_RESOLUTION)
        target_lons = np.arange(min_lon, max_lon + TARGET_RESOLUTION, TARGET_RESOLUTION)
        
        gridded_data = []
        
        for idx, row in df.iterrows():
            
            # Find the nearest grid point
            lat_idx = np.argmin(np.abs(target_lats - row['lat']))
            lon_idx = np.argmin(np.abs(target_lons - row['lon']))
            
            nearest_lat = target_lats[lat_idx]
            nearest_lon = target_lons[lon_idx]
            
            # Create a record for this grid point
            grid_record = {
                'lat': nearest_lat,
                'lon': nearest_lon,
                'year': row['year'],
                'month': row['month'],
                'day': row['day'],
                'emission': row['emission'],
                'Site': row['Site'],
                'ELEV': row['ELEV'],
                'MAT': row['MAT'],
                'MAP': row['MAP'],
                'NEE_F': row['NEE_F'],
                'TA_F': row['TA_F'],
                'VPD_F': row['VPD_F'],
                'P_F': row['P_F']
            }
            gridded_data.append(grid_record)
        
        if gridded_data:
            gridded_df = pd.DataFrame(gridded_data)
            gridded_df = gridded_df.set_index(['lat', 'lon', 'year', 'month', 'day'])
            
            # remove duplicated data, I checked, they are same, but where are they coming from?
            gridded_df = gridded_df[~gridded_df.index.duplicated(keep='first')]

            # Convert to xarray Dataset
            self.dataset = gridded_df.to_xarray()
            # print(self.dataset)
            # exit()
            self.dataset = self.dataset.reindex(
                lat=target_lats, 
                lon=target_lons, 
                fill_value=-9999
            )
        else:
            print("No data found for grid conversion")
            self.dataset = None

    
    def query_ori(self,
              lat_range: Tuple[float, float],
              lon_range: Tuple[float, float],
              time_range: Optional[Tuple[str, str]] = None,
              target: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        if self.processed_data is None or self.processed_data.empty:
            print("Error: cannot query, fluxnet data is not loaded!")
            return None

        min_lat, max_lat = lat_range
        if not -90 <= min_lat <= 90 or not -90 <= max_lat <= 90 or min_lat >= max_lat:
            print(f"ERROR: Invalid latitude range. Must be within [-90, 90] and min_lat < max_lat. Got: {lat_range}")
            return None

        min_lon, max_lon = lon_range
        if not -180 <= min_lon <= 180 or not -180 <= max_lon <= 180 or min_lon >= max_lon:
            print(f"ERROR: Invalid longitude range. Must be within [-180, 180] and min_lon < max_lon. Got: {lon_range}")
            return None

        if not time_range:
            print('ERROR: Must specify a time range to query, for example: ("2000-01", "2000-06") or ("2000-01-01", "2000-06-30").')
            return None

        try:
            df = self.processed_data.copy()

            # Spatial filter
            df = df[(df['lat'] >= lat_range[0]) & (df['lat'] <= lat_range[1]) &
                    (df['lon'] >= lon_range[0]) & (df['lon'] <= lon_range[1])]

            if df.empty:
                print(f"No fluxnet data within spatial range lat {lat_range}, lon {lon_range}")
                return None

            # Determine if day is available
            has_day = 'day' in df.columns and df['day'].notna().any()
            has_month = 'month' in df.columns and df['month'].notna().any()

            # Build a datetime for filtering; if day missing, use day=15 for mid-month
            if 'year' not in df.columns:
                print("ERROR: year column not found in chamber data after renaming.")
                return None

            day_series = df['day'] if has_day else 15
            month_series = df['month'] if has_month else 6
            dt = pd.to_datetime({
                'year': df['year'].astype('Int64'),
                'month': pd.Series(month_series, index=df.index).astype('Int64'),
                'day': pd.Series(day_series, index=df.index).astype('Int64')
            }, errors='coerce')

            start_date = pd.to_datetime(time_range[0])
            end_date = pd.to_datetime(time_range[1])
            if start_date > end_date:
                print(f"ERROR: Invalid time range. Start date must be before or equal to end date. Got: {time_range}")
                return None

            df = df[dt.between(start_date, end_date, inclusive='both')]

            if df.empty:
                print(f"No chamber data within time range {time_range}")
                return None

            # Sort output by lat lon year month day (if columns exist)
            sort_cols = [c for c in ['lat', 'lon', 'year', 'month', 'day'] if c in df.columns]
            df = df.sort_values(by=sort_cols).reset_index(drop=True)

            # If target variables specified, keep them plus the sort keys; else keep all
            if target:
                keep_cols = list(dict.fromkeys(sort_cols + target))
                existing_keep_cols = [c for c in keep_cols if c in df.columns]
                if existing_keep_cols:
                    df = df[existing_keep_cols]

            return df
        except Exception as e:
            print(f"ERROR: An error occurred while querying fluxnet data: {e}")
            return None

    def get_site_info(self, site_id: str = None):
        if self.site_info is None:
            print("Site information not loaded")
            return None
        
        if site_id is None:
            return self.site_info
        else:
            return self.site_info[self.site_info['SITE_ID'] == site_id]

    def get_flux_data(self, site_id: str = None):
        if self.flux_data is None:
            print("Flux data not loaded")
            return None
        
        if site_id is None:
            return self.flux_data
        else:
            return self.flux_data[self.flux_data['Site'] == site_id]

    def query(self, 
              lat_range: tuple, 
              lon_range: tuple, 
              time_range: tuple = None, 
              target: list = None) -> pd.DataFrame:
        """
        Query data with daily resolution support.
        
        Args:
            lat_range (tuple): Latitude range (min_lat, max_lat)
            lon_range (tuple): Longitude range (min_lon, max_lon)
            time_range (tuple): Time range (start_date, end_date) as strings
            target (list): Target variables to query
            
        Returns:
            pandas.DataFrame: Query results
        """
        df = self.query_ori(lat_range, lon_range, time_range, target)
        # if self.dataset is None:
        #     print("Error: cannot query, Dataset", self.name, "is not loaded!")
        #     return None

        min_lat, max_lat = lat_range
        if not -90 <= min_lat <= 90 or not -90 <= max_lat <= 90 or min_lat >= max_lat:
            print(f"ERROR: Invalid latitude range. Must be within [-90, 90] and min_lat < max_lat. Got: {lat_range}")
            return None

        min_lon, max_lon = lon_range
        if not -180 <= min_lon <= 180 or not -180 <= max_lon <= 180 or min_lon >= max_lon:
            print(f"ERROR: Invalid longitude range. Must be within [-180, 180] and min_lon < max_lon. Got: {lon_range}")
            return None
        
        target_lats = np.arange(-90 + TARGET_RESOLUTION, 90 + TARGET_RESOLUTION, TARGET_RESOLUTION)
        target_lons = np.arange(-180, 180, TARGET_RESOLUTION)
        min_lat_idx = np.argmin(np.abs(target_lats - min_lat))
        max_lat_idx = np.argmin(np.abs(target_lats - max_lat))
        min_lon_idx = np.argmin(np.abs(target_lons - min_lon))
        max_lon_idx = np.argmin(np.abs(target_lons - max_lon))
        nearest_lat_range = target_lats[min_lat_idx], target_lats[max_lat_idx]
        nearest_lon_range = target_lons[min_lon_idx], target_lons[max_lon_idx]
        dataset = self._convert_to_grid(df, nearest_lat_range, nearest_lon_range)

        try:
            spatial_mask = (
                (dataset.lat >= lat_range[0]) & 
                (dataset.lat <= lat_range[1]) &
                (dataset.lon >= lon_range[0]) & 
                (dataset.lon <= lon_range[1])
            )
            
            spatial_selection = dataset.where(spatial_mask, drop=True)
            
            if not time_range:
                print("ERROR: Must specify a time range to query, for example: (\"2000-01\", \"2000-06\").")
                return None

            start_date = pd.to_datetime(time_range[0])
            end_date = pd.to_datetime(time_range[1])
            if start_date >= end_date:
                print(f"ERROR: Invalid time range. Start date must be before end date. Got: {time_range}")
                return None

            ds = spatial_selection
            
            # Create time mask for daily data
            if start_date.year == end_date.year:
                if start_date.month == end_date.month:
                    # Same year, same month
                    time_mask = (ds.year == start_date.year) & \
                               (ds.month == start_date.month) & \
                               (ds.day >= start_date.day) & \
                               (ds.day <= end_date.day)
                else:
                    # Same year, different months
                    start_month_mask = (ds.year == start_date.year) & \
                                      (ds.month == start_date.month) & \
                                      (ds.day >= start_date.day)
                    end_month_mask = (ds.year == end_date.year) & \
                                    (ds.month == end_date.month) & \
                                    (ds.day <= end_date.day)
                    intermediate_months_mask = (ds.year == start_date.year) & \
                                             (ds.month > start_date.month) & \
                                             (ds.month < end_date.month)
                    time_mask = start_month_mask | end_month_mask | intermediate_months_mask
            else:
                # Different years
                start_year_mask = (ds.year == start_date.year) & \
                                 (ds.month >= start_date.month) & \
                                 ((ds.month > start_date.month) | (ds.day >= start_date.day))
                end_year_mask = (ds.year == end_date.year) & \
                               (ds.month <= end_date.month) & \
                               ((ds.month < end_date.month) | (ds.day <= end_date.day))
                intermediate_years_mask = (ds.year > start_date.year) & (ds.year < end_date.year)
                time_mask = start_year_mask | end_year_mask | intermediate_years_mask

            # Apply time mask
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
    # Test FLUXNET model
    fluxnet_model = FLUXNET_Model("FLUXNET", "../FLUXNET-CH4/")
    
    if fluxnet_model.dataset is not None:
        result_daily = fluxnet_model.query(
            lat_range=(40.0, 50.0), 
            lon_range=(-10.0, 10.0), 
            time_range=("2010/06/01", "2010/06/30")
        )
        
        if result_daily is not None and not result_daily.empty:
            print(result_daily.head(10))
        else:
            print("No daily data found for the query")
        
        result_monthly = fluxnet_model.query_ori(
            lat_range=(40.0, 50.0), 
            lon_range=(-10.0, 10.0), 
            time_range=("2009/12", "2010/02")
        )
        
        if result_monthly is not None and not result_monthly.empty:
            print(result_monthly)
        else:
            print("No monthly data found for the query")
        
        print("\n=== Site Information ===")
        # site_info = fluxnet_model.get_site_info()
        # if site_info is not None:
        #     print(f"Total sites: {len(site_info)}")
        #     print("Sample sites:")
        #     print(site_info[['SITE_ID', 'SITE_NAME', 'LOCATION_LAT', 'LOCATION_LONG']])
    else:
        print("Failed to load FLUXNET model")