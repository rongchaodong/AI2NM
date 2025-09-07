
import xarray as xr
from CH4_Model import CH4_Model

class MeMo_Model(CH4_Model):
    def _load_data(self):
        try:
            ds = xr.open_dataset(self.path, engine="netcdf4")
            # print("orginal ds", ds["Time"])
            
            time_idx = ds['Time'].values
            zero_based_idx = time_idx - 1
            start_year = 1990
            years = start_year + (zero_based_idx // 12)
            months = (zero_based_idx % 12) + 1
            ds = ds.assign_coords(year=('Time', years), month=('Time', months))
            # ds = ds.reset_coords('Time', drop=True)

            # Now rename the variables to our standard names
            rename_vars = {
                'CH4uptake': 'consumption'
            }
            self.dataset = ds.rename_vars(rename_vars)
        except FileNotFoundError:
            print(f"WARNING: File not found at {self.path}")
        except Exception as e:
            print(f"ERROR: Failed to load TEM model from {self.path}: {e}")


model = MeMo_Model("MeMo", "../bottom-up/MeMo_1990-2009_monthly_corrected.nc", (1.0, 1.0))
# print(model.dataset)
# exit()
print(model.query((-45.0, -43.0), (45.0, 47.0), ("2000-11", "2001-5"), ['consumption']))