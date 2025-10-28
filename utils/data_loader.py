"""
Enhanced data loader with regional statistics and caching
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import streamlit as st


class DataLoader:
    """Optimized data loader with caching and regional statistics"""

    def __init__(self, base_path: str = "./SI_leakage_results_test"):
        self.base_path = Path(base_path)
        self.cache = {}

        # Detect if this is SARMAD_MODEL or results_full structure
        self.is_sarmad = "SARMAD" in str(base_path).upper()

    @st.cache_data(ttl=3600)
    def load_model_data(_self, gender: str = 'male', time_horizon: str = '5_year') -> Dict[str, Any]:
        """Load model predictions and related data"""

        # Adjust paths based on directory structure
        if _self.is_sarmad:
            # SARMAD_MODEL structure: gender folders at root level
            shap_path = _self.base_path / f"{gender}/shap_ages.pkl"
            splits_path = _self.base_path / f"prepared_data/{gender}/data_splits.pkl"
            results_path = _self.base_path / f"{gender}/mortality_prediction_results.json"
            model_path = _self.base_path / f"{gender}/model.pkl"
            shap_explainer_path = _self.base_path / f"{gender}/shap_age_object.pkl"
            mortality_model_path = _self.base_path / f"{gender}/mortality_model_{time_horizon.replace('_year', 'y')}_shap.pkl"
        else:
            # Original SI_leakage_results_test structure
            shap_path = _self.base_path / f"{gender}/shap_ages.pkl"
            splits_path = _self.base_path / f"prepared_data/{gender}/data_splits.pkl"
            results_path = _self.base_path / f"{gender}/mortality_prediction_results.json"
            model_path = _self.base_path / f"{gender}/model.pkl"
            shap_explainer_path = _self.base_path / f"{gender}/shap_age_object.pkl"
            mortality_model_path = _self.base_path / f"{gender}/mortality_model_{time_horizon.replace('_year', 'y')}_shap.pkl"

        # Load SHAP ages and predictions
        with open(shap_path, 'rb') as f:
            shap_data = pickle.load(f)

        # Load data splits
        with open(splits_path, 'rb') as f:
            data_splits = pickle.load(f)

        # Load mortality prediction results
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Load actual models for predictions
        with open(model_path, 'rb') as f:
            xgb_model = pickle.load(f)

        # Load SHAP explainer
        with open(shap_explainer_path, 'rb') as f:
            shap_explainer = pickle.load(f)

        # Load mortality models
        if mortality_model_path.exists():
            with open(mortality_model_path, 'rb') as f:
                mortality_model = pickle.load(f)
        else:
            mortality_model = None

        return {
            'shap_data': shap_data,
            'data_splits': data_splits,
            'results': results,
            'xgb_model': xgb_model,
            'shap_explainer': shap_explainer,
            'mortality_model': mortality_model,
            'gender': gender,
            'time_horizon': time_horizon
        }

    @st.cache_data(ttl=3600)
    def load_age_scaler_stats(_self) -> Dict[str, float]:
        """Load age normalization statistics"""
        stats_path = Path('./age_scaler_stats.pkl')

        if stats_path.exists():
            with open(stats_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Default values from the data exploration
            return {
                'median': 33.0,
                'iqr': 21.0,
                'q1': 22.0,
                'q3': 43.0,
                'min': 1,
                'max': 149,
                'mean': 33.14,
                'std': 17.70
            }

    def denormalize_age(self, scaled_age: np.ndarray, scaler_stats: Dict[str, float]) -> np.ndarray:
        """Convert scaled age back to original age"""
        return scaled_age * scaler_stats['iqr'] + scaler_stats['median']

    @st.cache_data(ttl=3600)
    def compute_regional_statistics(_self, data_splits: Dict, shap_data: Dict) -> Dict[str, Any]:
        """Compute statistics for each region"""

        X_test = data_splits['X_test']

        # Extract region columns (they're one-hot encoded)
        # Handle both 'cat__region_en_x_' and 'region_en_x_' prefixes
        region_columns = [col for col in X_test.columns if 'region_en' in col.lower()]

        # Get the region for each patient
        patient_regions = []
        for idx in range(len(X_test)):
            for col in region_columns:
                if X_test.iloc[idx][col] == 1:
                    # Handle different prefixes
                    if 'cat__region_en_x_' in col:
                        region_name = col.replace('cat__region_en_x_', '')
                    elif 'region_en_x_' in col:
                        region_name = col.replace('region_en_x_', '')
                    else:
                        region_name = col.split('_')[-1]
                    patient_regions.append(region_name)
                    break
            else:
                patient_regions.append('Unknown')

        # Load age scaler stats for denormalization
        scaler_stats = _self.load_age_scaler_stats()

        # Denormalize ages
        chrono_ages = _self.denormalize_age(shap_data['age_test'], scaler_stats)
        bio_ages = _self.denormalize_age(shap_data['shap_age_test'], scaler_stats)
        age_acc = bio_ages - chrono_ages

        # Create DataFrame for easier computation
        stats_df = pd.DataFrame({
            'region': patient_regions,
            'chrono_age': chrono_ages,
            'bio_age': bio_ages,
            'age_acceleration': age_acc,
            'patient_idx': range(len(X_test))
        })

        # Add clinical conditions - be flexible with column naming
        condition_mappings = {
            'hypertension': None,
            'diabetes': None,
            'dyslipidemia': None,
            'obesity': None
        }

        # Look for condition columns with flexible matching
        for col in X_test.columns:
            col_lower = col.lower()
            for condition in condition_mappings:
                if condition in col_lower and ('has_' in col_lower or '_x_true' in col_lower):
                    condition_mappings[condition] = col

        # Add found condition columns to stats_df
        for condition, col_name in condition_mappings.items():
            if col_name is not None:
                stats_df[f'has_{condition}_x_True'] = X_test[col_name].values

        # Compute regional statistics
        regional_stats = {}
        for region in stats_df['region'].unique():
            if region != 'Unknown':
                region_data = stats_df[stats_df['region'] == region]

                regional_stats[region] = {
                    'sample_size': len(region_data),
                    'avg_chrono_age': region_data['chrono_age'].mean(),
                    'avg_bio_age': region_data['bio_age'].mean(),
                    'avg_age_acceleration': region_data['age_acceleration'].mean(),
                    'std_age_acceleration': region_data['age_acceleration'].std(),
                    'patient_indices': region_data['patient_idx'].tolist(),
                    'age_acc_percentiles': {
                        'p25': region_data['age_acceleration'].quantile(0.25),
                        'p50': region_data['age_acceleration'].quantile(0.50),
                        'p75': region_data['age_acceleration'].quantile(0.75)
                    }
                }

                # Add condition prevalence
                for condition in ['has_hypertension_x_True', 'has_diabetes_x_True',
                                 'has_dyslipidemia_x_True', 'has_obesity_x_True']:
                    if condition in region_data.columns:
                        condition_name = condition.replace('has_', '').replace('_x_True', '')
                        regional_stats[region][f'{condition_name}_prevalence'] = region_data[condition].mean()

        return {
            'regional_stats': regional_stats,
            'patient_regions': patient_regions,
            'stats_df': stats_df
        }

    @st.cache_data(ttl=3600)
    def load_metadata(_self) -> Dict:
        """Load metadata including region information"""
        metadata_path = Path('./SI_data/metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def get_mortality_risk_scores(self, model_data: Dict, patient_indices: np.ndarray = None) -> np.ndarray:
        """Calculate mortality risk scores for patients"""

        mortality_model = model_data.get('mortality_model')
        shap_data = model_data['shap_data']

        # Load and denormalize ages
        scaler_stats = self.load_age_scaler_stats()
        chrono_ages = self.denormalize_age(shap_data['age_test'], scaler_stats)
        bio_ages = self.denormalize_age(shap_data['shap_age_test'], scaler_stats)

        if mortality_model is None:
            # Use a simple risk score based on SHAP age acceleration
            age_acc = bio_ages - chrono_ages

            # Normalize age acceleration to 0-1 risk score
            # Using sigmoid transformation
            risk_scores = 1 / (1 + np.exp(-age_acc / 10))

            if patient_indices is not None:
                return risk_scores[patient_indices]
            return risk_scores
        else:
            # Mortality model expects 2 features: chronological age and SHAP age
            # Create feature matrix with these two columns

            if patient_indices is not None:
                # Select specific patients
                X_mortality = np.column_stack([
                    shap_data['age_test'][patient_indices],  # Keep scaled for model
                    shap_data['shap_age_test'][patient_indices]  # Keep scaled for model
                ])
            else:
                # Use all patients
                X_mortality = np.column_stack([
                    shap_data['age_test'],  # Chronological age (scaled)
                    shap_data['shap_age_test']  # SHAP age (scaled)
                ])

            # Get prediction probabilities
            if hasattr(mortality_model, 'predict_proba'):
                risk_scores = mortality_model.predict_proba(X_mortality)[:, 1]
            else:
                # For survival models, use different method
                risk_scores = 1 - mortality_model.predict_survival_function(X_mortality, return_array=True)[:, -1]

            return risk_scores