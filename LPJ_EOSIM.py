
import xarray as xr
import numpy as np
from CH4_Model import CH4_Model
from CH4_Model import TARGET_RESOLUTION

class PLJ_EOSIM_Model(CH4_Model):
    def _load_data(self):
        try:
            ds = xr.open_dataset(self.path, engine="netcdf4")
            # print("Original dataset:", ds)
            
            years = ds.time.dt.year.data
            months = ds.time.dt.month.data
            
            ds = ds.assign_coords(
                year=('time', years),
                month=('time', months)
            )

            # Now rename the variables to our standard names
            rename_vars = {
                'dch4e': 'emission',
            }
            ds = ds.rename_vars(rename_vars)
            
            ds = ds.set_index(time=['year', 'month'])
            ds = ds.unstack('time')
            
            if 'bnds' in ds.dims:
                ds = ds.drop_dims('bnds')
            if 'time_bnds' in ds.coords:
                ds = ds.drop_vars('time_bnds')
            ds = ds.transpose('latitude', 'longitude', 'year', 'month')
            
            new_lat = np.arange(-90 + TARGET_RESOLUTION, 90 + TARGET_RESOLUTION, TARGET_RESOLUTION)
            new_lon = np.arange(-180, 180, TARGET_RESOLUTION)
            
            # Use reindex with 'nearest' method for block-filling instead of interpolation
            ds = ds.reindex(
                latitude=new_lat, 
                longitude=new_lon, 
                method='nearest', 
                tolerance=max(self.original_resolution) # Look for nearest point within a radius of the old resolution
            )
            ds = ds.fillna(-9999)
            self.dataset = ds

        except FileNotFoundError:
            print(f"WARNING: File not found at {self.path}")
        except Exception as e:
            print(f"ERROR: Failed to load LPJ-EOSIM model from {self.path}: {e}")

if __name__ == '__main__':
    tem_model = PLJ_EOSIM_Model("PLJ_EOSIM", "../bottom-up/LPJ-EOSIM_ensemble_mch4e_1990-2024.nc")
    # print(tem_model.dataset)
    # exit()
    print(tem_model.query((-45.0, -43.0), (45.0, 47.0), ("2000-11", "2001-5")))