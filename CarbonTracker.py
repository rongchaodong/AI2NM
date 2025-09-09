
import xarray as xr
from CH4_Model import CH4_Model
from TEM_Model import TEM_Model

class CarbonTracker_Model(CH4_Model):
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
        
        # todo: somehow add TEM model's consumption into here, since TEM consumption equals CarbonTracker's consumption
        # try:
        #     tem_model = TEM_Model("CarbonTracker", "../bottom-up/TEM/TEM_ch4_wetland_soilsink_1950_2020.nc4")
        # except FileNotFoundError:
        #     print(f"WARNING: File not found at {self.path}")
        # except Exception as e:
        #     print(f"ERROR: Failed to load TEM model from {self.path}: {e}")

model = CarbonTracker_Model("CarbonTracker", "../top-down/CT-CH4_v2025_posterior_emission_1x1_category.nc", (1.0, 1.0))
# print(model.dataset)
# exit()
# print("---" * 10)
print(model.query((-45.0, -43.0), (45.0, 47.0), ("2000-11", "2001-5")))