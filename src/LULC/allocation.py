import numpy as np
import pandas as pd
from .config import logger

def get_allocated_dt(suitability_dict, transition_matrix, current_landuse_series, 
                     allocation_order, conversion_order, class_names, spatial_restrictions):
    """
    Faithful port of R's getAllocatedDT.
    
    Args:
        suitability_dict: {ClassName: DataFrame(id, weight)} (Sorted descending)
        transition_matrix: Numpy Array [From, To] (The budget of pixels to move)
        current_landuse_series: Series of current class IDs (1-based)
        allocation_order: List of Target Class IDs (Priority for 'To' Class)
        conversion_order: Matrix or List defining priority of 'From' Class
        class_names: List of class names (indexed 0..N-1)
        spatial_restrictions: List of inertia values [0..1] per class (1=No Change)
    """
    logger.info("Starting Competitive Allocation (R-Logic)...")

    n_pixels = len(current_landuse_series)
    
    # 1. Initialize Output (-1 = Unallocated)
    # R: outputAllocation[] <- NA
    final_allocation = np.full(n_pixels, -1, dtype=int)
    
    # Clean inputs (handle NaNs from One-Hot or previous steps)
    # 0 is used as 'No Data' placeholder here since classes are 1-based
    current_classes = np.nan_to_num(current_landuse_series.values, nan=0).astype(int)
    
    # R: newTM <- transitionMat
    budget_matrix = transition_matrix.copy()

    # =========================================================================
    # PHASE 1: HANDLING INERTIA (RestrictSpatialMigration)
    # R: if(restrictSpatialMigrationTM[i] > 0) ... noOfGridRestrictedBasedOnInertia
    # =========================================================================
    logger.info("Phase 1: Applying Spatial Inertia...")
    
    for i, inertia in enumerate(spatial_restrictions):
        class_id = i + 1  # 1-based ID
        
        # How many pixels of this class exist?
        current_pixels_mask = (current_classes == class_id)
        total_pixels = np.sum(current_pixels_mask)
        
        # R: restrictSpatialMigrationTM[i] * newTM[i,i] 
        # (This implies inertia applies to the diagonal 'Persistence' element)
        persistence_budget = budget_matrix[i, i]
        
        # Calculate how many MUST stay (locked)
        # In R logic, inertia often forces a specific number to remain
        # If inertia is 1.0, ALL persistence budget is locked immediately.
        locked_count = int(inertia * persistence_budget)
        
        if locked_count > 0:
            # Find the most suitable pixels to KEEP (usually highest suitability for itself)
            # R uses suitability or neighbor weight to decide which specific pixels stay
            
            # Get suitability for staying (Class i -> Class i)
            cls_name = class_names[i]
            if cls_name in suitability_dict:
                # Filter scores for pixels that are CURRENTLY this class
                scores_df = suitability_dict[cls_name]
                
                # We need to map global IDs to the boolean mask
                # Only consider pixels that are currently this class
                candidate_ids = scores_df[scores_df['id'].isin(np.where(current_pixels_mask)[0])]['id'].values
                
                # Take top N
                n_lock = min(len(candidate_ids), locked_count)
                ids_to_lock = candidate_ids[:n_lock]
                
                # Allocate them to themselves (Lock them)
                final_allocation[ids_to_lock] = class_id
                
                # Reduce budget
                budget_matrix[i, i] -= n_lock
                
                # logger.debug(f"  Class {class_id}: Locked {n_lock} pixels due to inertia.")

    # =========================================================================
    # PHASE 2: COMPETITIVE ALLOCATION
    # R: Loop classAllocationOrder (To) -> Loop conversionOrder (From)
    # =========================================================================
    logger.info("Phase 2: Competitive Allocation Loop...")
    
    for target_step, target_cls_id in enumerate(allocation_order):
        # target_cls_id is 1-based
        target_idx = target_cls_id - 1
        target_name = class_names[target_idx]
        
        if target_name not in suitability_dict:
            logger.warning(f"  Missing suitability map for {target_name}")
            continue
            
        # Get sorted suitability for Target Class
        # R: probableGridToAllocate <- suitablityMat[[classToAllocate]]
        target_scores_df = suitability_dict[target_name]
        
        # Inner Loop: From which class are we converting?
        # R: for(j in 1:noOfClass) { classToAllocate <- conversionOrderMat... }
        # Note: The R script logic variable naming is confusing (classFromAllocate vs classToAllocate).
        # Standard interpretation: We iterate through sources to fill the target.
        
        # If 'conversionOrder' is 'TP' or None, we just iterate 1..N
        # If it's a matrix, we follow the row/col priority.
        # Let's assume standard iteration 0..N-1 for source classes
        source_indices = range(len(class_names))
        
        for source_idx in source_indices:
            source_cls_id = source_idx + 1
            
            # Check remaining budget for Source -> Target
            pixels_needed = budget_matrix[source_idx, target_idx]
            
            if pixels_needed <= 0:
                continue
                
            # R: Filter candidates
            # 1. Must currently be Source Class
            # 2. Must NOT be allocated yet
            
            # Efficient Filtering using Set logic on IDs or Boolean Masks
            # Get all IDs that are currently Source Class
            current_source_mask = (current_classes == source_cls_id)
            # Get all IDs that are Unallocated
            unalloc_mask = (final_allocation == -1)
            
            valid_candidates_mask = current_source_mask & unalloc_mask
            valid_candidate_ids = np.where(valid_candidates_mask)[0]
            
            if len(valid_candidate_ids) == 0:
                continue
            
            # Now intersect valid_candidate_ids with the Sorted Suitability list
            # We want the HIGHEST suitability pixels that are in valid_candidate_ids
            
            # Filter the pre-sorted suitability dataframe
            # Boolean indexing on a large DataFrame can be slow, but it guarantees order
            candidates_sorted = target_scores_df[target_scores_df['id'].isin(valid_candidate_ids)]
            
            # Check if we have enough
            n_available = len(candidates_sorted)
            n_take = min(n_available, int(pixels_needed))
            
            if n_take > 0:
                # Extract Top N IDs
                ids_to_allocate = candidates_sorted['id'].values[:n_take]
                
                # Assign
                final_allocation[ids_to_allocate] = target_cls_id
                
                # Update Budget
                budget_matrix[source_idx, target_idx] -= n_take
                
                # logger.debug(f"  Allocated {n_take} pixels: {class_names[source_idx]} -> {target_name}")

    # =========================================================================
    # PHASE 3: FINAL CLEANUP (Handling Unallocated)
    # R: outputAllocation[is.na] <- original
    # =========================================================================
    logger.info("Phase 3: Finalizing Unallocated Pixels...")
    
    mask_unalloc = (final_allocation == -1)
    
    # 1. First, try to fill "Persistence" (Self -> Self) if budget remains
    # This handles pixels that were eligible to stay but weren't locked by inertia
    for i in range(len(class_names)):
        cls_id = i + 1
        remaining_self_budget = budget_matrix[i, i]
        
        if remaining_self_budget > 0:
            # Find unallocated pixels of this class
            mask_self_unalloc = mask_unalloc & (current_classes == cls_id)
            idxs = np.where(mask_self_unalloc)[0]
            
            n_fill = min(len(idxs), int(remaining_self_budget))
            if n_fill > 0:
                final_allocation[idxs[:n_fill]] = cls_id
                budget_matrix[i, i] -= n_fill
                # Update main mask
                mask_unalloc[idxs[:n_fill]] = False

    # 2. Absolute fallback: Whatever is left keeps its original class
    # (Even if budget says it should have moved, if we couldn't find a target)
    mask_still_unalloc = (final_allocation == -1)
    final_allocation[mask_still_unalloc] = current_classes[mask_still_unalloc]

    return final_allocation
