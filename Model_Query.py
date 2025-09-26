import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import argparse

from TEM_Model import TEM_Model
from MeMo import MeMo_Model
from LPJ_EOSIM import PLJ_EOSIM_Model
from CarbonTracker import CarbonTracker_Model
from FLUXNET import FLUXNET_Model
from Chamber_combined import Chamber_Combined_Model
from CH4_Model import CH4_Model
from TEM_driver import TEM_Driver

class ModelQueryManager:
    
    def __init__(self):
        self.models = {}
        self.model_configs = self._get_model_configs()
        self._initialize_models()
    
    def _get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            'TEM': {
                'class': TEM_Model,
                'path': '../bottom-up/TEM/TEM_ch4_wetland_soilsink_1950_2020.nc4',
                'resolution': 0.5,
            },
            'MeMo': {
                'class': MeMo_Model,
                'path': '../bottom-up/MeMo_1990-2009_monthly_corrected.nc',
                'resolution': (1.0, 1.0),
            },
            'LPJ_EOSIM': {
                'class': PLJ_EOSIM_Model,
                'path': '../bottom-up/LPJ-EOSIM_ensemble_mch4e_1990-2024.nc',
                'resolution': 0.5,
            },
            'CarbonTracker': {
                'class': CarbonTracker_Model,
                'path': '../top-down/CT-CH4_v2025_posterior_emission_1x1_category.nc',
                'resolution': 0.5,
            },
            'Chamber_Combined': {
                'class': Chamber_Combined_Model,
                'path': '../chamber/combined_chamber_data_Sep_12_2025_Youmi_Oh.csv',
                'resolution': 0.5,
            },
            'FLUXNET': {
                'class': FLUXNET_Model,
                'path': '../FLUXNET-CH4/',
                'resolution': 0.5,
            },
            'TEM_Driver': {
                'class': TEM_Driver,
                'path': '../TEM_monthly_driver/',
                'resolution': 0.5,
            }
        }
    
    def _initialize_models(self):
        print("Available models:")
        for name, config in self.model_configs.items():
            print(f"  - {name}")
    
    def _load_model(self, model_name: str):
        if model_name not in self.models:
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.model_configs.keys())}")
            
            config = self.model_configs[model_name]
            try:
                model = config['class'](
                    name=model_name,
                    path=config['path'],
                    resolution=config['resolution']
                )
                self.models[model_name] = model
                print(f"{model_name} loaded successfully")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                self.models[model_name] = None
    
    def get_available_models(self) -> List[str]:
        return list(self.model_configs.keys())
    
    
    def query_models(self,
                    lat_range: Tuple[float, float],
                    lon_range: Tuple[float, float],
                    time_range: Tuple[str, str],
                    model_names: Optional[List[str]] = None,
                    target_variables: Optional[List[str]] = None,
                    all_grid: bool = False,
                    output_path: str = "") -> Dict[str, pd.DataFrame]:
        """
        Query multiple models with the same spatial and temporal constraints.
        
        Args:
            lat_range: Latitude range (min_lat, max_lat)
            lon_range: Longitude range (min_lon, max_lon)
            time_range: Time range (start_date, end_date) as strings
            model_names: List of model names to query. If None, queries all available models.
            target_variables: List of target variables to include in results
            
        Returns:
            Dictionary mapping model names to their query results (DataFrames)
        """
        if model_names is None:
            model_names = self.get_available_models()
        
        # Validate model names
        invalid_models = [name for name in model_names if name not in self.model_configs]
        if invalid_models:
            raise ValueError(f"Unknown models: {invalid_models}. Available: {list(self.model_configs.keys())}")
        
        results = {}
        
        print(f"\n=== Querying {len(model_names)} models ===")
        print(f"Spatial range: lat {lat_range}, lon {lon_range}")
        print(f"Time range: {time_range}")
        if target_variables:
            print(f"Target variables: {target_variables}")
        print("-" * 50)
        
        for i, model_name in enumerate(model_names):
            print(f"\n[{i+1}/{len(model_names)}] Querying {model_name}...")
            
            # Load model if not already loaded
            self._load_model(model_name)
            
            if self.models[model_name] is None:
                print(f"  Skipping {model_name} (failed to load)")
                results[model_name] = None
                continue
            
            try:
                # Query the model
                if not all_grid and (model_name == 'FLUXNET' or model_name == 'Chamber_Combined'):
                    result = self.models[model_name].query_ori(
                    lat_range=lat_range,
                    lon_range=lon_range,
                    time_range=time_range,
                    target=target_variables
                    )
                else:
                    result = self.models[model_name].query(
                        lat_range=lat_range,
                        lon_range=lon_range,
                        time_range=time_range,
                        target=target_variables
                    )
                
                if result is not None and not result.empty:
                    results[model_name] = result
                    print(result)
                else:
                    results[model_name] = None
                    
            except Exception as e:
                print(f"  Query failed: {e}")
                results[model_name] = None
        
        # Export to NetCDF if output_path is specified
        if output_path:
            CH4_Model.merge_models_to_nc(results, output_path)
        
        return results
    

def main():
    parser = argparse.ArgumentParser(description="Query models of natural methane dataset")
    parser.add_argument(
        "--lat_range",
        nargs=2,
        type=float,
        required=True,
        metavar=("START_LAT", "END_LAT"),
        help="Start and end latitude for the query (e.g., -45.0 -43.0)"
    )
    parser.add_argument(
        "--lon_range",
        nargs=2,
        type=float,
        required=True,
        metavar=("START_LON", "END_LON"),
        help="Start and end longitude for the query (e.g., 45.0 47.0)"
    )
    parser.add_argument(
        "--time_range",
        nargs=2,
        type=str,
        required=True,
        metavar=("START_TIME", "END_TIME"),
        help="Start and end time in YYYY-MM format (e.g., 2000-11 2001-5)"
    )
    parser.add_argument(
        "--models",
        nargs='*',
        type=str,
        default=None,
        help="Optional: a list of specific models to query (e.g., TEM MeMo LPJ_EOSIM CarbonTracker Chamber_Combined FLUXNET TEM_Driver). If not provided, all models are queried."
    )
    parser.add_argument("--all_grid", action="store_true", help="Optional: output gridded data for FLUXNET and Chamber data.")
    parser.add_argument("--output_path", type=str, default="", help="Optional: Output the queried results into a netcdf file with specified output_path (e.g. /content/drive/MyDrive/output.nc).")
    args = parser.parse_args()

    manager = ModelQueryManager()
    
    # # Example 1: Query all models
    # print("\n" + "="*60)
    # print("EXAMPLE 1: Query all available models")
    # print("="*60)
    
    # results_all = manager.query_models(
    #     lat_range=(-45.0, -43.0),
    #     lon_range=(45.0, 47.0),
    #     time_range=("2000-11", "2001-5")
    # )
    
    # # Example 2: Query specific models only
    # print("\n" + "="*60)
    # print("EXAMPLE 2: Query specific models only")
    # print("="*60)
    
    resultss = manager.query_models(
        lat_range=tuple(args.lat_range),
        lon_range=tuple(args.lon_range),
        time_range=tuple(args.time_range),
        model_names=args.models,
        all_grid = args.all_grid,
        output_path = args.output_path
    )
    
if __name__ == '__main__':
    main()
