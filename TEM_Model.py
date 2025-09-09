
import xarray as xr
from CH4_Model import CH4_Model

class TEM_Model(CH4_Model):
    def _load_data(self):
        try:
            ds = xr.open_dataset(self.path, engine="netcdf4")
            ds['year'] = ds['year'].astype(int)
            ds['month'] = ds['month'].astype(int)
            # Now rename the variables to our standard names
            rename_vars = {
                'ch4_emission': 'emission',
                'ch4_oxidation': 'consumption'
            }
            ds = ds.rename_vars(rename_vars)
            ds = ds.transpose('latitude', 'longitude', 'year', 'month')
            # remain all data, convert NaN into -9999
            self.dataset = ds.fillna(-9999)
        except FileNotFoundError:
            print(f"WARNING: File not found at {self.path}")
        except Exception as e:
            print(f"ERROR: Failed to load TEM model from {self.path}: {e}")

if __name__ == '__main__':
    tem_model = TEM_Model("TEM", "../bottom-up/TEM/TEM_ch4_wetland_soilsink_1950_2020.nc4")
    # print(tem_model.dataset)
    # exit()
    print(tem_model.query((-45.0, -43.0), (45.0, 47.0), ("2000-11", "2001-5")))