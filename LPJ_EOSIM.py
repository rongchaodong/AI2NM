
import xarray as xr
import numpy as np
from CH4_Model import CH4_Model
from CH4_Model import TARGET_RESOLUTION

class PLJ_EOSIM_Model(CH4_Model):
    def _load_data(self):
        try:
            emission_data_path = self.path / 'LPJ-EOSIM_ensemble_mch4e_1990-2024.nc'
            fraction_data_path = self.path / 'ensemble_mwet_frac_1990-2024.nc'

            if not emission_data_path.exists():
                raise FileNotFoundError(f"Emission data file not found: {emission_data_path}")
            if not fraction_data_path.exists():
                raise FileNotFoundError(f"Fraction data file not found: {fraction_data_path}")

            ds = xr.open_dataset(emission_data_path, engine="netcdf4", chunks='auto')
            frac_ds = xr.open_dataset(fraction_data_path, engine="netcdf4", chunks='auto')
            
            # Now rename the variables to our standard names
            rename_vars = {
                'dch4e': 'emission',
            }
            ds = self._process_data(ds, rename_vars)
            frac_rename_vars = {
                'mwet_frac': 'wetland_fraction',
            }
            frac_ds = self._process_data(frac_ds, frac_rename_vars)

            ds = ds.merge(frac_ds)

            # Intensity calculation: adjust emissions by wetland fraction
            # ds['emission'] = xr.where(ds['emission'] != 0, ds['emission'] / frac_ds['wetland_fraction'], ds['emission'])


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

    def _process_data(self, ds: xr.Dataset, rename: dict):
        years = ds.time.dt.year.data
        months = ds.time.dt.month.data
        
        ds = ds.assign_coords(
            year=('time', years),
            month=('time', months)
        )
        ds = ds.rename_vars(rename)
            
        ds = ds.set_index(time=['year', 'month'])
        ds = ds.unstack('time')
        
        if 'bnds' in ds.dims:
            ds = ds.drop_dims('bnds')
        if 'time_bnds' in ds.coords:
            ds = ds.drop_vars('time_bnds')
        ds = ds.transpose('latitude', 'longitude', 'year', 'month')
        return ds

if __name__ == '__main__':
    tem_model = PLJ_EOSIM_Model("PLJ_EOSIM", "../bottom-up/PLJ-EOSIM/")
    # print(tem_model.dataset)
    # exit()
    print(tem_model.query((65.0, 66.0), (-176.0, -175.0), ("2000-11", "2001-5")).head(10))