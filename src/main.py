import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add src to python path to find the LULC package
sys.path.append(str(Path(__file__).parent / "src"))

from LULC import LULCAlgorithms as Algorithm
from LULC.config import logger

def main():
    print("Starting OpenLDM Execution - Python Port")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    # =========================================================================
    # 1. FILE PATHS (Matching runSteps1a.R exactly)
    # =========================================================================
    
    # Base directory relative to this script
    base_dir = Path("../data/example")
    
    T1File = str(base_dir / "LULC/1985.tif")
    T2File = str(base_dir / "LULC/1995.tif")
    T3File = str(base_dir / "LULC/2005.tif")
    
    # Output file
    output_dir = base_dir / "outputdata"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    PredictedFile = str(output_dir / "senario-5-neighwith3-allocorder-134596782-demand-as2005-mixregression-rrrrrrlrl-2005.tif")
    
    suitabilityDirectory = str(output_dir)

    # =========================================================================
    # 2. DRIVERS (Exact paths from R)
    # =========================================================================
    
    # T1 Drivers (1985)
    drvs85 = {
        "DistanceToDrainage": str(base_dir / "Drivers/drivers_85/dist_stream.img"),
        "DistanceToBuiltup": str(base_dir / "Drivers/drivers_85/Dist_urban.img"),
        "DistanceToRoad": str(base_dir / "Drivers/drivers_85/road_final.img"),
        "Elevation": str(base_dir / "Drivers/commonDrivers/elevation.img")
    }

    # T2 Drivers (1995)
    drvs95 = {
        "DistanceToDrainage": str(base_dir / "Drivers/drivers_95/dist_stream.img"),
        "DistanceToBuiltup": str(base_dir / "Drivers/drivers_95/Dist_urban.img"),
        "DistanceToRoad": str(base_dir / "Drivers/drivers_95/road_final.img"),
        "Elevation": str(base_dir / "Drivers/commonDrivers/elevation.img")
    }

    # =========================================================================
    # 3. MODEL PARAMETERS (Exact from R)
    # =========================================================================
    
    clsName = [
        "BuildUp", "Agriculture", "DenseForest", "FallowLand", 
        "GrassLand", "MixedForest", "Plantation", "ScrubLand", "WaterBody"
    ]
    
    # Spatial Restrictions (Inertia) - Exact from R
    restrictSpatial = [1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76]
    
    # Demand (Actual) - Exact from R
    mydemand = [1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992]
    
    # Neighborhood Rules: [WindowSize, Steps, SaveSteps] - Exact from R
    neighbourl = [3, 2, 1]
    
    # Allocation Order - Exact from R
    myallocationorder = [1, 3, 4, 5, 9, 6, 7, 8, 2]
    
    # Conversion Matrix - Set to 'TP' (Transition Probability) as in R
    myconversion = 'TP'
    
    # NA Value - Exact from R
    na_value = 128
    
    # Model Types - Exact from R (all logistic)
    model_type = ['logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic', 'logistic']
    #model_type = ['randomForest', 'randomForest', 'randomForest', 'randomForest', 'randomForest', 'randomForest', 'randomForest', 'randomForest', 'randomForest']
    #model_type = ['nnet', 'nnet', 'nnet', 'nnet', 'nnet', 'nnet', 'nnet', 'nnet', 'nnet' ]
    #model_type = ['svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm', 'svm']
    #model_type = ['regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression', 'regression']
    #model_type = ['randomForest','randomForest','randomForest','randomForest','randomForest','randomForest','logistic','randomForest','logistic']

    
    # =========================================================================
    # 4. MODEL FORMULAS (Complete from R script)
    # =========================================================================
    
    model_formula = [
        "T1.BuildUp ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation",
        "T1.Agriculture ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation",
        "T1.DenseForest ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation",
        "T1.FallowLand ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation",
        "T1.GrassLand ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation",
        "T1.MixedForest ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation",
        "T1.Plantation ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation",
        "T1.ScrubLand ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation",
        "T1.WaterBody ~ TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation"
    ]

    # =========================================================================
    # 5. EXECUTION
    # =========================================================================

    try:
        print("\n--- Running generatePredictedMap ---")
        
        Algorithm.generate_predicted_map(
            modelType=model_type,
            T1File=T1File,
            T2File=T2File,
            withClassName=clsName,
            T1drivers=drvs85,
            T2drivers=drvs95,
            na_value=na_value,
            demand=mydemand,
            restrictSpatialMigration=restrictSpatial,
            neighbour=neighbourl,
            outputfile=PredictedFile,
            conversionOrder=myconversion,
            classAllocationOrder=myallocationorder,
            maskFile=None,  # NA in R
            aoiFile=None,   # NA in R
            modelformula=model_formula,
            suitabilityFileDirectory=suitabilityDirectory
        )
        
        print("\n--- Prediction Complete ---")
        
        # =========================================================================
        # 6. VALIDATION
        # =========================================================================
        if os.path.exists(T3File) and os.path.exists(PredictedFile):
            print("\n--- Calculating Accuracy (Kappa) ---")
            Algorithm.get_kappa_summary(
                actualFile=T3File,
                predictedFile=PredictedFile,
                na_value=na_value,
                classNames=clsName
            )
            
    except Exception as e:
        logger.error(f"Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    end_time = datetime.now()
    print(f"\nEnd Time: {end_time}")
    print(f"Total Duration: {end_time - start_time}")
    print(f"Output saved to: {PredictedFile}")

if __name__ == "__main__":
    # Ensure working directory is set correctly
    os.chdir(Path(__file__).parent)
    main()
