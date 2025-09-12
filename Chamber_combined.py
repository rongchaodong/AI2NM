import xarray as xr
from CH4_Model import CH4_Model
import pandas as pd
import numpy as np
from CH4_Model import TARGET_RESOLUTION
from pathlib import Path


class Chamber_Combined_Model(CH4_Model):
    
    def __init__(self, name: str, path: str, resolution: float = TARGET_RESOLUTION):
        self.site_info = None
        self.flux_data = None
        self.processed_data = None
        super().__init__(name, path, resolution)

    def _load_data(self):
        try:
            site_info_path = self.path / 'FLUXNET_CH4_2024.csv'
            flux_data_path = self.path / 'FLUXNET_T1_DD.csv'
        
        except:
            print()