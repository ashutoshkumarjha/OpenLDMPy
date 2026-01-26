import rasterio as rio
import pandas as pd
import numpy as np
import logging
from .config import NA_VALUE, logger

class DataManager:
    @staticmethod
    def raster_to_dataframe(file_path: str, layer_name_prefix: str = "band") -> pd.DataFrame:
        """Reads a raster into a Pandas DataFrame with an 'id' column."""
        logger.info(f"Reading raster: {file_path}")
        try:
            with rio.open(file_path) as src:
                data = src.read()
                n_bands, height, width = data.shape
                
                flat_data = data.reshape(n_bands, -1).T
                
                if n_bands == 1:
                    col_names = [layer_name_prefix]
                else:
                    col_names = [f"{layer_name_prefix}_{i+1}" for i in range(n_bands)]
                    
                df = pd.DataFrame(flat_data, columns=col_names)
                df['id'] = np.arange(len(df))
                
                if src.nodata is not None:
                    df[col_names] = df[col_names].replace(src.nodata, np.nan)
                
                return df
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise

    @staticmethod
    def dataframe_to_raster(df: pd.DataFrame, reference_raster: str, output_path: str, value_col: str):
        """Writes a DataFrame column back to a Raster."""
        logger.info(f"Writing output to: {output_path}")
        with rio.open(reference_raster) as src:
            profile = src.profile.copy()
            height, width = src.shape
            
            # FORCE INTEGER DTYPE
            # LULC maps are categorical, so uint8 (0-255) or uint16 is appropriate.
            # We use the profile's dtype if it's integer, otherwise default to uint8.
            if 'int' not in profile['dtype']:
                profile['dtype'] = 'uint8'
                
            dtype = profile['dtype']
            
            # Initialize with NA_VALUE cast to the correct integer type
            out_data = np.full((height * width), int(NA_VALUE), dtype=dtype)
            
            if 'id' in df.columns and value_col in df.columns:
                valid_indices = df['id'].values.astype(int)
                
                # FORCE INTEGER CASTING ON VALUES
                values = df[value_col].fillna(NA_VALUE).values
                # Round to nearest int just in case, then cast
                values = np.rint(values).astype(dtype)
                
                out_data[valid_indices] = values
            
            out_data = out_data.reshape(1, height, width)
            profile.update(count=1, nodata=NA_VALUE)
            
            with rio.open(output_path, 'w', **profile) as dst:
                dst.write(out_data)

    @staticmethod
    def prepare_driver_stack(driver_dict: dict) -> pd.DataFrame:
        """Reads multiple driver files and merges them into a single DataFrame."""
        master_df = None
        for name, path in driver_dict.items():
            df = DataManager.raster_to_dataframe(path, layer_name_prefix=name)
            if master_df is None:
                master_df = df
            else:
                # Merge efficiently by index (assuming aligned grids)
                master_df = pd.concat([master_df, df.drop(columns=['id'])], axis=1)
        return master_df
    
    # In DataManager class, add:
    @staticmethod
    def open(file_path):
        """Context manager for rasterio.open."""
        return rio.open(file_path)

