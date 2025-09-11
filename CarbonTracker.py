
import xarray as xr
from CH4_Model import CH4_Model
from TEM_Model import TEM_Model
from typing import Union, Tuple, Optional
import pandas as pd

class CarbonTracker_Model(CH4_Model):

    def __init__(self, name, path, resolution = 0.5):
        self.consumption_data = None

        super().__init__(name, path, resolution)
        self._get_consumption()


    def _load_data(self):
        try:
            ds = xr.open_dataset(self.path, engine="netcdf4")

            # Now rename the variables to our standard names
            rename_vars = {
                'post_wetland': 'emission'
            }
            ds = ds.rename_vars(rename_vars)
            ds = ds.transpose('latitude', 'longitude', 'year', 'month')
            # remain all data, convert NaN into -9999
            self.dataset = ds.fillna(-9999)
        except FileNotFoundError:
            print(f"WARNING: File not found at {self.path}")
        except Exception as e:
            print(f"ERROR: Failed to load TEM model from {self.path}: {e}")


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

            spatial_consumption = self.consumption_data.sel(
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
            time_consumption = spatial_consumption.where(time_mask, drop=True)
            time_selection = time_selection.merge(time_consumption)
            # print("try:", time_selection)
            df = time_selection.to_dataframe()
            if target and all(item in df.columns for item in target):
                # Drop NaN rows
                df = df.dropna(subset=target).reset_index()
            # default: no droping, remain all data. here, the dropna is used to drop the contents outside the time range
            else: # replace NaN
                df = df.dropna(how='all').reset_index()
            
            # print(df)
            # exit()
            if df.empty:
                print(df)
                print(f"Locations from lan {lat_range} to lon {lon_range} within time range {time_range} no data exist!")
            else:
                # print(df)
                return df

        except Exception as e:
            print(f"ERROR: An error occurred while querying model '{self.name}': {e}")
            return None

    def _get_consumption(self):
        # add TEM model's consumption into here, since TEM consumption equals CarbonTracker's consumption
        tem_model = TEM_Model("CarbonTracker", "../bottom-up/TEM/TEM_ch4_wetland_soilsink_1950_2020.nc4")
        consumption_ds = tem_model.dataset
        consumption_ds = consumption_ds[['lat', 'lon', 'year', 'month', 'consumption']]
        self.consumption_data = consumption_ds

if __name__ == '__main__':
    model = CarbonTracker_Model("CarbonTracker", "../top-down/CT-CH4_v2025_posterior_emission_1x1_category.nc", (1.0, 1.0))
    # print(model.dataset)
    # exit()
    # print("---" * 10)
    print(model.query((-45.0, -43.0), (45.0, 47.0), ("2000-11", "2001-5")))