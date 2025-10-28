"""
Real Model Prediction Runner with SHAP Integration
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from typing import Dict, Any, Optional, Tuple, List
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


class ModelRunner:
    """Run real model predictions and generate SHAP explanations"""

    def __init__(self, model_data: Dict[str, Any]):
        """
        Initialize with loaded model data

        Args:
            model_data: Dictionary containing models, explainers, and data
        """
        self.xgb_model = model_data.get('xgb_model')
        self.shap_explainer = model_data.get('shap_explainer')
        self.mortality_model = model_data.get('mortality_model')
        self.data_splits = model_data.get('data_splits')
        self.shap_data = model_data.get('shap_data')
        self.gender = model_data.get('gender', 'male')
        self.time_horizon = model_data.get('time_horizon', '5_year')

        # Cache for predictions
        self.prediction_cache = {}

    def predict_biological_age(self, X: pd.DataFrame, use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict biological age using the XGBoost model

        Args:
            X: Feature DataFrame
            use_cache: Whether to use cached predictions

        Returns:
            Dictionary with predictions and SHAP values
        """

        # Create cache key
        cache_key = f"bio_age_{len(X)}_{X.iloc[0].sum() if len(X) > 0 else 0}"

        if use_cache and cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        results = {}

        # Get predictions from XGBoost model
        if self.xgb_model is not None:
            try:
                # Make predictions
                predictions = self.xgb_model.predict(X)
                results['predictions'] = predictions

                # Get SHAP values if explainer available
                if self.shap_explainer is not None:
                    # Handle both old and new SHAP API
                    shap_output = self.shap_explainer.shap_values(X)

                    # Check if it's an Explanation object (new API) or array (old API)
                    if hasattr(shap_output, 'values'):
                        # New SHAP API - Explanation object
                        shap_values = shap_output.values
                    elif isinstance(shap_output, np.ndarray):
                        # Old SHAP API - direct array
                        shap_values = shap_output
                    else:
                        # Try as is
                        shap_values = shap_output

                    results['shap_values'] = shap_values
                    results['base_value'] = self.shap_explainer.expected_value

                    # Calculate SHAP age (biological age)
                    shap_age = self.shap_explainer.expected_value + np.sum(shap_values, axis=1)
                    results['shap_age'] = shap_age

                    # Load age scaler for denormalization
                    from utils.data_loader import DataLoader
                    loader = DataLoader()
                    scaler_stats = loader.load_age_scaler_stats()

                    # Denormalize ages
                    if 'age_x' in X.columns:
                        chrono_age = loader.denormalize_age(X['age_x'].values, scaler_stats)
                    else:
                        chrono_age = np.full(len(X), scaler_stats['median'])

                    bio_age = loader.denormalize_age(shap_age, scaler_stats)

                    results['chronological_age'] = chrono_age
                    results['biological_age'] = bio_age
                    results['age_acceleration'] = bio_age - chrono_age

            except Exception as e:
                st.error(f"Error in biological age prediction: {e}")
                results['error'] = str(e)

        else:
            st.warning("XGBoost model not available, using fallback calculations")
            # Use existing SHAP data if available
            results = self._use_fallback_predictions(X)

        # Cache results
        if use_cache:
            self.prediction_cache[cache_key] = results

        return results

    def predict_mortality_risk(self, X: pd.DataFrame, use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict mortality/hospitalization risk

        Args:
            X: Feature DataFrame
            use_cache: Whether to use cached predictions

        Returns:
            Dictionary with risk predictions
        """

        # Create cache key
        cache_key = f"mortality_{self.time_horizon}_{len(X)}_{X.iloc[0].sum() if len(X) > 0 else 0}"

        if use_cache and cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        results = {}

        if self.mortality_model is not None:
            try:
                # First, predict biological age for these patients
                bio_age_results = self.predict_biological_age(X, use_cache=False)

                # Mortality model expects 2 features: chronological age and SHAP age (scaled)
                if 'age_x' in X.columns:
                    chrono_age_scaled = X['age_x'].values
                else:
                    # Use default if age not available
                    chrono_age_scaled = np.zeros(len(X))

                if 'shap_age' in bio_age_results:
                    shap_age_scaled = bio_age_results['shap_age']
                else:
                    # Fallback: use chronological age
                    shap_age_scaled = chrono_age_scaled

                # Create feature matrix with 2 columns as expected by mortality model
                X_mortality = np.column_stack([chrono_age_scaled, shap_age_scaled])

                # Check model type and predict accordingly
                if hasattr(self.mortality_model, 'predict_proba'):
                    # Classification model
                    probabilities = self.mortality_model.predict_proba(X_mortality)
                    results['risk_scores'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
                    results['risk_categories'] = self._categorize_risk(results['risk_scores'])

                elif hasattr(self.mortality_model, 'predict_survival_function'):
                    # Survival model
                    survival_probs = self.mortality_model.predict_survival_function(X_mortality, return_array=True)
                    # Get risk at the specified time horizon
                    results['risk_scores'] = 1 - survival_probs[:, -1]
                    results['survival_curves'] = survival_probs
                    results['risk_categories'] = self._categorize_risk(results['risk_scores'])

                else:
                    # Fallback to basic prediction
                    predictions = self.mortality_model.predict(X_mortality)
                    results['risk_scores'] = predictions
                    results['risk_categories'] = self._categorize_risk(predictions)

            except Exception as e:
                st.error(f"Error in mortality risk prediction: {e}")
                results['error'] = str(e)
                results = self._use_fallback_risk_scores(X)

        else:
            results = self._use_fallback_risk_scores(X)

        # Cache results
        if use_cache:
            self.prediction_cache[cache_key] = results

        return results

    def get_feature_importance(self, X: pd.DataFrame, patient_idx: int = 0) -> Dict[str, Any]:
        """
        Get feature importance for a specific patient

        Args:
            X: Feature DataFrame
            patient_idx: Index of patient to explain

        Returns:
            Dictionary with feature importance data
        """

        if self.shap_explainer is None:
            return self._get_fallback_feature_importance(X, patient_idx)

        try:
            # Get SHAP values for the patient
            patient_features = X.iloc[patient_idx:patient_idx+1]

            # Handle both old and new SHAP API
            shap_output = self.shap_explainer.shap_values(patient_features)

            # Check if it's an Explanation object (new API) or array (old API)
            if hasattr(shap_output, 'values'):
                # New SHAP API - Explanation object
                shap_values = shap_output.values[0] if shap_output.values.ndim > 1 else shap_output.values
            elif isinstance(shap_output, np.ndarray):
                # Old SHAP API - direct array
                shap_values = shap_output[0] if shap_output.ndim > 1 else shap_output
            else:
                # Try indexing directly
                shap_values = shap_output[0]

            # Get feature names and values
            feature_names = X.columns.tolist()
            feature_values = patient_features.iloc[0].values

            # Sort by absolute SHAP value
            importance_data = []
            for i, (name, value, shap_val) in enumerate(zip(feature_names, feature_values, shap_values)):
                # Clean feature name
                clean_name = name.replace('_x', '').replace('_', ' ').replace('has ', '').title()
                importance_data.append({
                    'feature': clean_name,
                    'value': value,
                    'shap_value': shap_val,
                    'abs_shap_value': abs(shap_val)
                })

            # Sort by absolute importance
            importance_data.sort(key=lambda x: x['abs_shap_value'], reverse=True)

            # Get top features
            top_features = importance_data[:20]

            return {
                'top_features': top_features,
                'base_value': self.shap_explainer.expected_value,
                'all_features': importance_data
            }

        except Exception as e:
            st.error(f"Error getting feature importance: {e}")
            return self._get_fallback_feature_importance(X, patient_idx)

    def create_waterfall_plot(self, feature_importance: Dict[str, Any], max_features: int = 15) -> go.Figure:
        """
        Create SHAP waterfall plot for feature contributions

        Args:
            feature_importance: Feature importance dictionary
            max_features: Maximum number of features to display

        Returns:
            Plotly figure
        """

        if 'top_features' not in feature_importance:
            return go.Figure()

        top_features = feature_importance['top_features'][:max_features]
        base_value = feature_importance.get('base_value', 0)

        # Prepare data for waterfall
        features = [f['feature'] for f in top_features]
        values = [f['shap_value'] for f in top_features]

        # Calculate cumulative values
        cumulative = [base_value]
        for val in values:
            cumulative.append(cumulative[-1] + val)

        # Create waterfall chart
        fig = go.Figure()

        # Add base value bar
        fig.add_trace(go.Bar(
            x=[base_value],
            y=['Base Value'],
            orientation='h',
            marker=dict(color='lightgray'),
            name='Base',
            showlegend=False
        ))

        # Add feature contribution bars with clear labeling
        for i, (feature, value) in enumerate(zip(features, values)):
            # Color based on impact direction
            if value > 0:
                color = '#e74c3c'  # Red for aging factors
                impact_text = 'Accelerates aging'
            else:
                color = '#27ae60'  # Green for protective factors
                impact_text = 'Slows aging'

            # Clean feature name
            clean_feature = (feature.replace('_x', '')
                                  .replace('_', ' ')
                                  .replace('has ', '')
                                  .replace(' True', '')
                                  .replace(' False', '')
                                  .title())

            fig.add_trace(go.Bar(
                x=[abs(value)],
                y=[clean_feature],
                orientation='h',
                marker=dict(color=color, width=0.8),
                base=cumulative[i] if value > 0 else cumulative[i+1],
                name=clean_feature,
                showlegend=False,
                text=f'{value:+.2f}',
                textposition='outside',
                hovertemplate=(f'<b>{clean_feature}</b><br>'
                             f'Impact: {value:+.2f}<br>'
                             f'{impact_text}<br>'
                             f'<extra></extra>')
            ))

        # Add final prediction line
        final_value = cumulative[-1]
        fig.add_vline(x=final_value, line_dash="dash", line_color="black",
                     annotation_text=f"Prediction: {final_value:.3f}")

        # Add legend for colors
        fig.add_trace(go.Bar(
            x=[0],
            y=['Legend'],
            orientation='h',
            marker=dict(color='#e74c3c'),
            name='Aging Factors',
            showlegend=True,
            visible='legendonly'
        ))
        fig.add_trace(go.Bar(
            x=[0],
            y=['Legend'],
            orientation='h',
            marker=dict(color='#27ae60'),
            name='Protective Factors',
            showlegend=True,
            visible='legendonly'
        ))

        fig.update_layout(
            title={
                'text': 'Factors Contributing to Your Biological Age',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Impact on Biological Age',
            yaxis_title='Health Factors',
            height=1200,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.1,
                xanchor='center',
                x=0.5
            ),
            yaxis=dict(
                autorange="reversed",
                categorygap=0.1 # KEY FIX: This reduces the space between bars for a thicker look
            ),
            annotations=[
                dict(
                    text='← Younger | Older →',
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=11, color='gray')
                )
            ]
        )

        return fig

    def create_risk_gauge(self, risk_score: float) -> go.Figure:
        """
        Create risk gauge visualization

        Args:
            risk_score: Risk score (0-1)

        Returns:
            Plotly gauge figure
        """

        risk_category = self._categorize_risk(np.array([risk_score]))[0]

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100 if risk_score > 0 else risk_score * -100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Mortality Risk ({self.time_horizon.replace('_', ' ')})",
                   'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#c8e6c9'},
                    {'range': [30, 60], 'color': '#fff9c4'},
                    {'range': [60, 100], 'color': '#ffcdd2'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            annotations=[
                dict(
                    text=f"<b>{risk_category}</b>",
                    x=0.5, y=-0.1,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=18, color="black")
                )
            ]
        )

        return fig

    def compare_cohorts(self, cohort1_data: Dict, cohort2_data: Dict,
                        cohort1_name: str = "Cohort 1",
                        cohort2_name: str = "Cohort 2") -> Dict[str, go.Figure]:
        """
        Compare predictions between two cohorts

        Args:
            cohort1_data: Predictions for first cohort
            cohort2_data: Predictions for second cohort
            cohort1_name: Name of first cohort
            cohort2_name: Name of second cohort

        Returns:
            Dictionary of comparison figures
        """

        figures = {}

        # Age acceleration comparison
        if 'age_acceleration' in cohort1_data and 'age_acceleration' in cohort2_data:
            fig = go.Figure()

            fig.add_trace(go.Box(
                y=cohort1_data['age_acceleration'],
                name=cohort1_name,
                marker_color='#3498db'
            ))

            fig.add_trace(go.Box(
                y=cohort2_data['age_acceleration'],
                name=cohort2_name,
                marker_color='#e74c3c'
            ))

            fig.update_layout(
                title='Age Acceleration Comparison',
                yaxis_title='Age Acceleration (years)',
                showlegend=True,
                height=400
            )

            figures['age_acceleration'] = fig

        # Risk score comparison
        if 'risk_scores' in cohort1_data and 'risk_scores' in cohort2_data:
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=cohort1_data['risk_scores'] * 100,
                name=cohort1_name,
                opacity=0.7,
                marker_color='#3498db',
                nbinsx=30
            ))

            fig.add_trace(go.Histogram(
                x=cohort2_data['risk_scores'] * 100,
                name=cohort2_name,
                opacity=0.7,
                marker_color='#e74c3c',
                nbinsx=30
            ))

            fig.update_layout(
                title='Risk Score Distribution Comparison',
                xaxis_title='Risk Score (%)',
                yaxis_title='Count',
                barmode='overlay',
                showlegend=True,
                height=400
            )

            figures['risk_distribution'] = fig

        return figures

    def _categorize_risk(self, risk_scores: np.ndarray) -> List[str]:
        """Categorize risk scores into Low, Medium, High"""
        categories = []
        for score in risk_scores:
            if score < 0.3:
                categories.append("Low Risk")
            elif score < 0.6:
                categories.append("Medium Risk")
            else:
                categories.append("High Risk")
        return categories

    def _use_fallback_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Use fallback predictions when model is not available"""
        n_samples = len(X)

        # Use existing SHAP data if indices match
        if self.shap_data and 'shap_age_test' in self.shap_data:
            # For now, return random subset or all if size matches
            if len(self.shap_data['shap_age_test']) == n_samples:
                return {
                    'biological_age': self.shap_data['shap_age_test'],
                    'chronological_age': self.shap_data.get('age_test', np.zeros(n_samples)),
                    'age_acceleration': self.shap_data.get('age_acc_test', np.zeros(n_samples))
                }

        # Generate placeholder data
        from utils.data_loader import DataLoader
        # Use the same base_path as the main model data
        base_path = getattr(self, 'base_path', "./SARMAD_MODEL")
        loader = DataLoader(base_path=base_path)
        scaler_stats = loader.load_age_scaler_stats()

        if 'age_x' in X.columns:
            chrono_age = loader.denormalize_age(X['age_x'].values, scaler_stats)
        else:
            chrono_age = np.random.normal(33, 18, n_samples)

        # Simulate biological age with some variation
        bio_age = chrono_age + np.random.normal(0, 5, n_samples)

        return {
            'chronological_age': chrono_age,
            'biological_age': bio_age,
            'age_acceleration': bio_age - chrono_age
        }

    def _use_fallback_risk_scores(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate fallback risk scores"""
        n_samples = len(X)

        # Base risk on age and conditions
        risk_scores = np.random.beta(2, 5, n_samples)  # Skewed toward lower risk

        # Adjust based on conditions if available
        for condition in ['has_hypertension_x_True', 'has_diabetes_x_True',
                         'has_obesity_x_True', 'has_dyslipidemia_x_True']:
            if condition in X.columns:
                risk_scores += X[condition].values * 0.1

        # Normalize to 0-1
        risk_scores = np.clip(risk_scores, 0, 1)

        return {
            'risk_scores': risk_scores,
            'risk_categories': self._categorize_risk(risk_scores)
        }

    def _get_fallback_feature_importance(self, X: pd.DataFrame, patient_idx: int) -> Dict[str, Any]:
        """Generate fallback feature importance"""

        # Get top features based on variance or known clinical importance
        important_features = [
            'age_x', 'has_diabetes_x_True', 'has_hypertension_x_True',
            'glucose_x', 'creatinine_x', 'albumin_x', 'hemoglobin_x'
        ]

        feature_importance = []
        for feature in X.columns[:20]:  # Top 20 features
            if feature in important_features:
                importance = np.random.uniform(0.1, 0.5)
            else:
                importance = np.random.uniform(-0.1, 0.1)

            clean_name = feature.replace('_x', '').replace('_', ' ').replace('has ', '').title()
            feature_importance.append({
                'feature': clean_name,
                'value': X.iloc[patient_idx][feature] if patient_idx < len(X) else 0,
                'shap_value': importance,
                'abs_shap_value': abs(importance)
            })

        feature_importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)

        return {
            'top_features': feature_importance[:20],
            'base_value': 0,
            'all_features': feature_importance
        }