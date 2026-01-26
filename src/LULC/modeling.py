import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from joblib import Parallel, delayed
from .config import PARALLEL_JOBS, logger

class ModelEngine:
    @staticmethod
    def train_and_predict(train_df: pd.DataFrame, 
                          pred_df: pd.DataFrame,
                          target_col: str, 
                          driver_cols: list, 
                          model_types: list, 
                          class_names: list) -> dict:
        """
        Fits models on T1 data and predicts suitability on T2 data.
        Returns: {class_name: DataFrame(id, weight)} sorted by weight.
        """
        logger.info("Starting Parallel Model Training & Prediction...")
        
        X_train = train_df[driver_cols].fillna(0)
        X_pred = pred_df[driver_cols].fillna(0)
        pred_ids = pred_df['id'].values

        def process_single_class(idx, cls_name):
            m_type = model_types[idx] if isinstance(model_types, list) else model_types
            
            # Target: 1 if pixel is this class, 0 otherwise
            target_val = idx + 1 # Assuming 1-based class IDs in Raster
            y_train = (train_df[target_col] == target_val).astype(int)
            
            if y_train.sum() < 10:
                logger.warning(f"Class {cls_name} has insufficient data. Skipping.")
                return cls_name, pd.DataFrame({'id': pred_ids, 'weight': 0.0})

            # Instantiate
            if m_type == 'logistic':
                model = LogisticRegression(max_iter=1000, solver='liblinear')
            elif m_type == 'randomForest':
                model = RandomForestClassifier(n_estimators=100, n_jobs=1, max_depth=15)
            elif m_type == 'nnet':
                model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
            elif m_type == 'svm':
                model = SVC(probability=True, kernel='rbf')
            else:
                model = LogisticRegression()

            # Train & Predict
            try:
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_pred)[:, 1] # Probability of class 1
            except Exception as e:
                logger.error(f"Error training {cls_name}: {e}")
                probs = np.zeros(len(X_pred))

            # Return Result sorted by weight (descending)
            result_df = pd.DataFrame({'id': pred_ids, 'weight': probs})
            return cls_name, result_df.sort_values('weight', ascending=False)

        # Parallel Execution
        results = Parallel(n_jobs=PARALLEL_JOBS)(
            delayed(process_single_class)(i, name) for i, name in enumerate(class_names)
        )
        
        return {cls: df for cls, df in results}
