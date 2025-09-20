
import xarray as xr
from CH4_Model import CH4_Model
import numpy as np

class TEM_Driver(CH4_Model):
    def _load_data(self):
        try:

            stat_dict = {}
            stat_features = ['clelev', 'clfaotxt', 'cltveg', 'phh2o', 'topsoil_bulk_density', 'vegetation_type_11', 'wetlandtype', 'climatetype']
            stat_files = ['clelev.nc', 'clfaotxt.nc', 'cltveg.nc', 'phh2o.nc', 'topsoil_bulk_density.nc', 'vegetation_type_11.nc', 'wetlandtype.nc', 'climatetype.nc']
            for i, feat in enumerate(stat_features):
                path = self.path / stat_files[i]
                feature = xr.open_dataset(path)
                # print(feature)
                # exit()
                feature = feature[feat]
                stat_dict[feat] = feature

            npp_list = []
            for year in range(1979, 2019):
                npp_path = self.path / f'monthly_NPP_{year}.nc'
                feature = xr.open_dataset(npp_path)
                feature = feature['NPP']
                npp = feature.expand_dims({'year': [year]})
                npp_list.append(npp)
            
            npp_data = xr.concat(npp_list, dim='year')
            npp_data = npp_data.transpose('latitude', 'longitude', 'year', 'month')
            npp_data = npp_data.rename({'latitude': 'lat', 'longitude': 'lon'})
            npp_data = npp_data.fillna(-9999)

            broadcast = {}

            for feat, val in stat_dict.items():
                val_with_time = val.expand_dims({'year': npp_data['year'], 'month': npp_data['month']}).transpose('lat', 'lon', 'year', 'month')
                broadcast[feat] = val_with_time

            ds = {**broadcast, 'NPP': npp_data}
            ds = xr.Dataset(ds)
            self.dataset = ds.fillna(-9999)
        except FileNotFoundError:
            print(f"WARNING: File not found at {self.path}")
        except Exception as e:
            print(f"ERROR: Failed to load TEM model from {self.path}: {e}")

if __name__ == '__main__':
    model = TEM_Driver("TEM_Driver", "../TEM_monthly_driver/")
    # print(model.dataset)
    # exit()
    print(model.query((-45.0, -43.0), (45.0, 47.0), ("2000-11", "2001-5")))