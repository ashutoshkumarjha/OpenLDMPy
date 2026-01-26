import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from .data_io import DataManager
from .config import logger, NA_VALUE  # ← ADD THIS LINE
import os

# ... rest of functions unchanged ...


def get_yearly_transition_matrix(tm_values, steps=1):
    """
    Decomposes a multi-year transition matrix using Matrix Logarithm/Exponential.
    """
    row_sums = tm_values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = tm_values / row_sums
    
    # Estimate Yearly Generator Matrix Q
    # P_step = P ^ (1/steps) => Q = logm(P) / steps
    try:
        Q = logm(P) / steps
        P_yearly = expm(Q)
        # Handle small complex errors from logm
        P_yearly = np.real(P_yearly)
    except Exception as e:
        logger.warning(f"Matrix decomposition failed: {e}. Using linear interpolation.")
        P_yearly = P # Fallback
        
    final_counts = P_yearly * row_sums
    return np.round(final_counts).astype(int)

# src/LULC/spatial.py (Add to existing file)

import os
from scipy.ndimage import convolve
from .data_io import DataManager
from .config import logger

def compute_neighbor_weights(df_grid, neighbour_params, na_value=NA_VALUE, reference_raster=None):
    """
    FULL R: ParallelComputeNearByWeight() + ComputeNearByWeight()
    Focal convolution for neighborhood weights.
    """
    logger.info("Computing Neighborhood Weights...")
    
    window_size = neighbour_params[0] if neighbour_params else 3
    kernel_size = window_size
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / kernel.sum()  # Normalize
    
    # Get grid shape
    with DataManager.open(reference_raster) as src:
        height, width = src.shape
        grid_shape = (height, width)
    
    # Process each class column
    class_cols = [col for col in df_grid.columns if col != 'id']
    neighbor_weights = {}
    
    for class_col in class_cols:
        class_grid_2d = df_grid[class_col].values.reshape(grid_shape)
        class_grid_2d = np.nan_to_num(class_grid_2d, nan=0.0)
        
        # Focal convolution (R: focal() equivalent)
        neighbor_grid = convolve(class_grid_2d, kernel, mode='constant', cval=0.0)
        neighbor_flat = neighbor_grid.flatten()
        
        # Restore NA mask
        na_mask = pd.isna(df_grid[class_col])
        neighbor_flat[na_mask] = np.nan
        
        neighbor_df = pd.DataFrame({
            'id': df_grid['id'],
            'weight': neighbor_flat
        }).sort_values('weight', ascending=False)
        
        neighbor_weights[class_col] = neighbor_df
    
    return neighbor_weights

def create_neighbor_maps(neighbor_weights, reference_raster, output_directory, class_names):
    """R: createNeighbourMap()"""
    logger.info(f"Saving neighbor maps: {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    
    for class_name, nw_df in neighbor_weights.items():
        nw_file = os.path.join(output_directory, f"{class_name}_NW.tif")
        DataManager.dataframe_to_raster(
            nw_df.rename(columns={'weight': 'NeighborWeight'}),
            reference_raster,
            nw_file,
            'NeighborWeight'
        )
        logger.info(f"  ✓ {nw_file}")

def create_suitability_maps(suitability_maps, reference_raster, output_directory, class_names):
    """R: createSuitabilityMap()"""
    logger.info(f"Saving suitability maps: {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    
    for class_name, sm_df in suitability_maps.items():
        sm_file = os.path.join(output_directory, f"{class_name}_SM.tif")
        DataManager.dataframe_to_raster(
            sm_df.rename(columns={'weight': 'Suitability'}),
            reference_raster,
            sm_file,
            'Suitability'
        )
        logger.info(f"  ✓ {sm_file}")
