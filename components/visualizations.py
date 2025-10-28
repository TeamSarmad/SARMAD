"""
Enhanced Data Visualization Components
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from plotly.subplots import make_subplots


class EnhancedVisualizations:
    """Advanced visualization components for the demo app"""

    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }

    def create_3d_scatter(self, data: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                         color_col: Optional[str] = None, size_col: Optional[str] = None,
                         title: str = "3D Visualization") -> go.Figure:
        """
        Create interactive 3D scatter plot

        Args:
            data: DataFrame with data
            x_col, y_col, z_col: Column names for axes
            color_col: Column for color mapping
            size_col: Column for size mapping
            title: Plot title

        Returns:
            Plotly 3D scatter figure
        """

        fig = go.Figure()

        # Prepare marker properties
        marker_dict = {
            'size': 5,
            'opacity': 0.8,
        }

        if size_col and size_col in data.columns:
            sizes = data[size_col].values
            # Normalize sizes
            marker_dict['size'] = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 15 + 5

        if color_col and color_col in data.columns:
            marker_dict['color'] = data[color_col].values
            marker_dict['colorscale'] = 'Viridis'
            marker_dict['showscale'] = True
            marker_dict['colorbar'] = dict(title=color_col, thickness=15)

        fig.add_trace(go.Scatter3d(
            x=data[x_col],
            y=data[y_col],
            z=data[z_col],
            mode='markers',
            marker=marker_dict,
            text=data.index if isinstance(data.index, pd.Index) else None,
            hovertemplate='<b>Patient %{text}</b><br>' +
                         f'{x_col}: %{{x:.2f}}<br>' +
                         f'{y_col}: %{{y:.2f}}<br>' +
                         f'{z_col}: %{{z:.2f}}<br>' +
                         '<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig

    def create_sankey_diagram(self, source_categories: List[str], target_categories: List[str],
                             flows: List[int], title: str = "Patient Flow Diagram") -> go.Figure:
        """
        Create Sankey diagram for patient flow visualization

        Args:
            source_categories: List of source category names
            target_categories: List of target category names
            flows: List of flow values between categories
            title: Diagram title

        Returns:
            Plotly Sankey figure
        """

        # Create label list (source + target categories)
        labels = list(set(source_categories + target_categories))
        label_indices = {label: i for i, label in enumerate(labels)}

        # Map categories to indices
        source_indices = [label_indices[cat] for cat in source_categories]
        target_indices = [label_indices[cat] for cat in target_categories]

        # Create color map
        node_colors = []
        for label in labels:
            if 'Low' in label or 'Healthy' in label:
                node_colors.append(self.color_scheme['success'])
            elif 'Medium' in label or 'Moderate' in label:
                node_colors.append(self.color_scheme['warning'])
            elif 'High' in label or 'Severe' in label:
                node_colors.append(self.color_scheme['danger'])
            else:
                node_colors.append(self.color_scheme['info'])

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
                hovertemplate='%{label}<br>Count: %{value}<extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=flows,
                color='rgba(0,0,0,0.2)',
                hovertemplate='%{source.label} → %{target.label}<br>Count: %{value}<extra></extra>'
            )
        )])

        fig.update_layout(
            title=title,
            font_size=12,
            height=500
        )

        return fig

    def create_radar_chart(self, patient_values: Dict[str, float],
                          population_means: Dict[str, float],
                          title: str = "Patient Profile") -> go.Figure:
        """
        Create radar chart comparing patient to population

        Args:
            patient_values: Dictionary of patient biomarker values
            population_means: Dictionary of population mean values
            title: Chart title

        Returns:
            Plotly radar chart figure
        """

        categories = list(patient_values.keys())
        patient_vals = list(patient_values.values())
        population_vals = [population_means.get(cat, 0) for cat in categories]

        fig = go.Figure()

        # Patient profile
        fig.add_trace(go.Scatterpolar(
            r=patient_vals,
            theta=categories,
            fill='toself',
            name='Patient',
            line=dict(color=self.color_scheme['primary'], width=2),
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))

        # Population average
        fig.add_trace(go.Scatterpolar(
            r=population_vals,
            theta=categories,
            fill='toself',
            name='Population Average',
            line=dict(color=self.color_scheme['secondary'], width=2, dash='dash'),
            fillcolor='rgba(255, 127, 14, 0.1)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(patient_vals), max(population_vals)) * 1.1]
                )),
            showlegend=True,
            title=title,
            height=400
        )

        return fig

    def create_time_series_forecast(self, historical_data: pd.DataFrame,
                                   forecast_data: Optional[pd.DataFrame] = None,
                                   title: str = "Risk Trajectory") -> go.Figure:
        """
        Create time series plot with optional forecast

        Args:
            historical_data: DataFrame with time series data
            forecast_data: Optional forecast DataFrame
            title: Plot title

        Returns:
            Plotly time series figure
        """

        fig = go.Figure()

        # Historical data
        if 'date' in historical_data.columns and 'value' in historical_data.columns:
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['value'],
                mode='lines+markers',
                name='Historical',
                line=dict(color=self.color_scheme['primary'], width=2),
                marker=dict(size=6)
            ))

            # Add confidence interval if available
            if 'lower_bound' in historical_data.columns and 'upper_bound' in historical_data.columns:
                fig.add_trace(go.Scatter(
                    x=historical_data['date'],
                    y=historical_data['upper_bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=historical_data['date'],
                    y=historical_data['lower_bound'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(31, 119, 180, 0.2)',
                    fill='tonexty',
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # Forecast data
        if forecast_data is not None and 'date' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['value'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=self.color_scheme['warning'], width=2, dash='dash'),
                marker=dict(size=6)
            ))

            # Add forecast confidence interval
            if 'lower_bound' in forecast_data.columns and 'upper_bound' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['upper_bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=forecast_data['date'],
                    y=forecast_data['lower_bound'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(255, 152, 0, 0.2)',
                    fill='tonexty',
                    showlegend=False,
                    hoverinfo='skip'
                ))

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Risk Score',
            hovermode='x unified',
            height=400,
            showlegend=True
        )

        return fig

    def create_correlation_heatmap(self, data: pd.DataFrame, features: List[str],
                                  title: str = "Feature Correlations") -> go.Figure:
        """
        Create correlation heatmap for selected features

        Args:
            data: DataFrame with feature data
            features: List of feature names to include
            title: Heatmap title

        Returns:
            Plotly heatmap figure
        """

        # Calculate correlation matrix
        corr_matrix = data[features].corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title='Correlation', thickness=15),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            height=500,
            xaxis_tickangle=-45,
            margin=dict(b=100)
        )

        return fig

    def create_violin_plots(self, data: pd.DataFrame, groupby_col: str,
                          value_cols: List[str], title: str = "Distribution Comparison") -> go.Figure:
        """
        Create violin plots for distribution comparison

        Args:
            data: DataFrame with data
            groupby_col: Column to group by
            value_cols: Columns to plot distributions for
            title: Plot title

        Returns:
            Plotly violin plot figure
        """

        fig = make_subplots(
            rows=1, cols=len(value_cols),
            subplot_titles=value_cols,
            shared_yaxes=False
        )

        colors = [self.color_scheme['primary'], self.color_scheme['secondary'],
                 self.color_scheme['success'], self.color_scheme['danger']]

        for i, col in enumerate(value_cols):
            groups = data[groupby_col].unique()

            for j, group in enumerate(groups):
                group_data = data[data[groupby_col] == group][col].dropna()

                fig.add_trace(
                    go.Violin(
                        y=group_data,
                        name=str(group),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=colors[j % len(colors)],
                        opacity=0.7,
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=1, col=i+1
                )

        fig.update_layout(
            title=title,
            height=400,
            showlegend=True
        )

        return fig

    def create_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                        model_name: str = "Model") -> go.Figure:
        """
        Create ROC curve visualization

        Args:
            y_true: True labels
            y_scores: Prediction scores
            model_name: Name of the model

        Returns:
            Plotly ROC curve figure
        """

        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(color=self.color_scheme['primary'], width=2),
            hovertemplate='False Positive Rate: %{x:.3f}<br>True Positive Rate: %{y:.3f}<extra></extra>'
        ))

        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=True
        ))

        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            xaxis=dict(constrain='domain'),
            yaxis=dict(scaleanchor='x', scaleratio=1, constrain='domain'),
            showlegend=True
        )

        return fig

    def create_biomarker_timeline(self, patient_data: pd.DataFrame,
                                 biomarkers: List[str], reference_ranges: Dict[str, Tuple[float, float]],
                                 title: str = "Biomarker Timeline") -> go.Figure:
        """
        Create timeline visualization of biomarker values

        Args:
            patient_data: DataFrame with patient biomarker history
            biomarkers: List of biomarker names to plot
            reference_ranges: Dictionary of normal ranges for each biomarker
            title: Plot title

        Returns:
            Plotly timeline figure
        """

        fig = make_subplots(
            rows=len(biomarkers),
            cols=1,
            subplot_titles=biomarkers,
            shared_xaxes=True,
            vertical_spacing=0.05
        )

        for i, biomarker in enumerate(biomarkers):
            if biomarker in patient_data.columns:
                # Plot biomarker values
                fig.add_trace(
                    go.Scatter(
                        x=patient_data.index,
                        y=patient_data[biomarker],
                        mode='lines+markers',
                        name=biomarker,
                        line=dict(color=self.color_scheme['primary'], width=2),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )

                # Add reference range if available
                if biomarker in reference_ranges:
                    low, high = reference_ranges[biomarker]

                    # Lower bound
                    fig.add_hline(
                        y=low,
                        line_dash="dash",
                        line_color="green",
                        line_width=1,
                        row=i+1, col=1
                    )

                    # Upper bound
                    fig.add_hline(
                        y=high,
                        line_dash="dash",
                        line_color="red",
                        line_width=1,
                        row=i+1, col=1
                    )

                    # Shade normal range
                    fig.add_hrect(
                        y0=low, y1=high,
                        fillcolor="green",
                        opacity=0.1,
                        line_width=0,
                        row=i+1, col=1
                    )

        fig.update_layout(
            title=title,
            height=200 * len(biomarkers),
            showlegend=False,
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="Time", row=len(biomarkers), col=1)

        return fig

    def create_intervention_impact_chart(self, baseline: Dict[str, float],
                                        interventions: Dict[str, Dict[str, float]],
                                        title: str = "Intervention Impact Analysis") -> go.Figure:
        """
        Create chart showing impact of different interventions

        Args:
            baseline: Baseline metric values
            interventions: Dictionary of intervention names and their metric impacts
            title: Chart title

        Returns:
            Plotly grouped bar chart
        """

        metrics = list(baseline.keys())
        x = metrics

        fig = go.Figure()

        # Baseline bars
        fig.add_trace(go.Bar(
            x=x,
            y=list(baseline.values()),
            name='Baseline',
            marker_color=self.color_scheme['info'],
            text=[f'{v:.1f}' for v in baseline.values()],
            textposition='auto'
        ))

        # Intervention bars
        colors = [self.color_scheme['success'], self.color_scheme['warning'],
                 self.color_scheme['danger'], self.color_scheme['primary']]

        for i, (intervention_name, intervention_values) in enumerate(interventions.items()):
            values = [intervention_values.get(metric, baseline[metric]) for metric in metrics]
            fig.add_trace(go.Bar(
                x=x,
                y=values,
                name=intervention_name,
                marker_color=colors[i % len(colors)],
                text=[f'{v:.1f}' for v in values],
                textposition='auto'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Metrics',
            yaxis_title='Value',
            barmode='group',
            height=400,
            showlegend=True
        )

        return fig