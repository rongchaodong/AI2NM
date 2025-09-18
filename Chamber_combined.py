import pandas as pd
import numpy as np
from CH4_Model import TARGET_RESOLUTION
from pathlib import Path
from typing import Optional, Tuple, List
from CH4_Model import CH4_Model


class Chamber_Combined_Model(CH4_Model):
    def __init__(self, name: str, path: str, resolution: float = TARGET_RESOLUTION):
        self.df = None
        super().__init__(name, path, resolution)

        # self._convert_to_grid()

    def _load_data(self):
        try:
            csv_path = self.path
            if isinstance(csv_path, Path):
                csv_path = self.path
            else:
                csv_path = Path(self.path)

            if not csv_path.exists():
                raise FileNotFoundError(f"Combined chamber CSV not found: {csv_path}")

            df = pd.read_csv(csv_path)

            # Rename to year/month/day if present
            rename_map = {}
            if 'FCH4_sample_year' in df.columns:
                rename_map['FCH4_sample_year'] = 'year'
            if 'FCH4_sample_month' in df.columns:
                rename_map['FCH4_sample_month'] = 'month'
            if 'FCH4_sample_day' in df.columns:
                rename_map['FCH4_sample_day'] = 'day'

            if rename_map:
                df = df.rename(columns=rename_map)

            # # Ensure numeric types where applicable
            # for col in ['lat', 'lon', 'year', 'month', 'day']:
            #     if col in df.columns:
            #         df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop invalid lat/lon/year/month rows (-9999)
            df = df[~((df['lat'] == -9999) | (df['lon'] == -9999))]
            df = df[~((df['year'] == -9999) | (df['month'] == -9999))]

            final_df = df[['lat', 'lon', 'year', 'month', 'day', 'dataset_no',
                                       'site_unique_id', 'wetland', 'FCH4_daily_mean', 'FCH4_daily_median', 'FCH4_daily_sd', 'FCH4_daily_n_spa', 'FCH4_daily_n_tem', 'soil_temp', 'soil_temp_depth',
                                       'air_temp', 'wtd_depth']].copy()
            self.df = final_df
            self.dataset = final_df.to_xarray()
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            self.df = None
            self.dataset = None
        except Exception as e:
            print(f"ERROR: Failed to load Chamber_Combined model from {self.path}: {e}")
            self.df = None
            self.dataset = None

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
                'dataset_no': row['dataset_no'],
                'site_unique_id': row['site_unique_id'],
                'wetland': row['wetland'],
                'FCH4_daily_mean': row['FCH4_daily_mean'],
                'FCH4_daily_median': row['FCH4_daily_median'],
                'FCH4_daily_sd': row['FCH4_daily_sd'],
                'FCH4_daily_n_spa': row['FCH4_daily_n_spa'],
                'FCH4_daily_n_tem': row['FCH4_daily_n_tem'],
                'soil_temp': row['soil_temp'],
                'soil_temp_depth': row['soil_temp_depth'],
                'air_temp': row['air_temp'],
                'wtd_depth': row['wtd_depth'],
            }
            gridded_data.append(grid_record)
        
        if gridded_data:
            gridded_df = pd.DataFrame(gridded_data)
            gridded_df = gridded_df.set_index(['lat', 'lon', 'year', 'month', 'day'])
            
            # remove duplicated data, I checked, they are same, but where are they coming from?
            gridded_df = gridded_df[~gridded_df.index.duplicated(keep='first')]

            # Convert to xarray Dataset
            dataset = gridded_df.to_xarray()
            dataset = dataset.reindex(
                lat=target_lats, 
                lon=target_lons, 
                fill_value=-9999
            )
            # print(dataset)
            # exit()
        else:
            print("No data found for grid conversion")
            dataset = None
        return dataset
    
    def query_ori(self,
              lat_range: Tuple[float, float],
              lon_range: Tuple[float, float],
              time_range: Optional[Tuple[str, str]] = None,
              target: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        if self.df is None or self.df.empty:
            print("Error: cannot query, Chamber combined data is not loaded!")
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
            df = self.df.copy()

            # Spatial filter
            df = df[(df['lat'] >= lat_range[0]) & (df['lat'] <= lat_range[1]) &
                    (df['lon'] >= lon_range[0]) & (df['lon'] <= lon_range[1])]

            if df.empty:
                print(f"No chamber data within spatial range lat {lat_range}, lon {lon_range}")
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
            print(f"ERROR: An error occurred while querying chamber combined data: {e}")
            return None
    
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
        # print(df)
        # exit()
        # if df is None:
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
            
            # TODO: I cannot convert all into gridded data due to memory issue, maybe we can do round lan and lon before output
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
    model = Chamber_Combined_Model(
        "Chamber_Combined",
        "../chamber/combined_chamber_data_Sep_12_2025_Youmi_Oh.csv"
    )

    result = model.query(
        lat_range=(30.0, 31.0),
        lon_range=(-90.0, -88.0),
        time_range=("2012-01-01", "2020-12-31")
    )

    if result is not None and not result.empty:
        print(result)
    else:
        print("No data returned for sample chamber combined query")