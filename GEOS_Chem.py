
import xarray as xr
from CH4_Model import CH4_Model
from MeMo import MeMo_Model
# TODO: waiting the source files
class GEOS_Chem_Model(CH4_Model):
    def _load_data(self):
        try:
            ds = xr.open_dataset(self.path, engine="netcdf4")
            # print("orginal ds", ds['latitude'])
            # exit()

            # Now rename the variables to our standard names
            rename_vars = {
                'post_wetland': 'emission'
            }
            self.dataset = ds.rename_vars(rename_vars)
        except FileNotFoundError:
            print(f"WARNING: File not found at {self.path}")
        except Exception as e:
            print(f"ERROR: Failed to load TEM model from {self.path}: {e}")
        
        # todo: somehow add Memo model's consumption into here, since same consumption
        try:
            tem_model = MeMo_Model("GEOS_Chem", "../bottom-up/MeMo_1990-2009_monthly_corrected.nc")
        except FileNotFoundError:
            print(f"WARNING: File not found at {self.path}")
        except Exception as e:
            print(f"ERROR: Failed to load TEM model from {self.path}: {e}")


model = GEOS_Chem_Model("GEOS_Chem", "../top-down/CT-CH4_v2025_posterior_emission_1x1_category.nc", (1.0, 1.0))
print(model.dataset)
print(model.query((-45.0, -43.0), (45.0, 47.0), ("2000-11", "2001-5")))