"""
Saudi Arabia Regional Map Visualization Component
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional


class SaudiMapVisualizer:
    """Interactive map visualization for Saudi Arabia regions"""

    # Saudi Arabia region coordinates (approximate centers for visualization)
    REGION_COORDINATES = {
        'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'ar_name': 'الرياض'},
        'Makkah': {'lat': 21.4225, 'lon': 39.8262, 'ar_name': 'مكة المكرمة'},
        'Madinah': {'lat': 24.4672, 'lon': 39.6024, 'ar_name': 'المدينة المنورة'},
        'Eastern Region': {'lat': 26.4207, 'lon': 50.0888, 'ar_name': 'المنطقة الشرقية'},
        'Asir': {'lat': 18.2164, 'lon': 42.5053, 'ar_name': 'عسير'},
        'Tabuk': {'lat': 28.3838, 'lon': 36.5550, 'ar_name': 'تبوك'},
        'Qassim': {'lat': 26.3267, 'lon': 43.9753, 'ar_name': 'القصيم'},
        'Hail': {'lat': 27.5219, 'lon': 41.6907, 'ar_name': 'حائل'},
        'Northern Region': {'lat': 30.9843, 'lon': 41.1179, 'ar_name': 'الحدود الشمالية'},
        'Jizan': {'lat': 16.8892, 'lon': 42.5511, 'ar_name': 'جازان'},
        'Najran': {'lat': 17.4933, 'lon': 44.1277, 'ar_name': 'نجران'},
        'Bahah': {'lat': 20.0129, 'lon': 41.4677, 'ar_name': 'الباحة'},
        'Jawf': {'lat': 29.8117, 'lon': 39.8621, 'ar_name': 'الجوف'}
    }

    def __init__(self):
        self.selected_region = None

    def create_regional_map(self, regional_stats: Dict[str, Any], metric: str = 'avg_age_acceleration') -> go.Figure:
        """Create an interactive map of Saudi Arabia with regional statistics"""

        # Prepare data for visualization
        map_data = []
        for region, coords in self.REGION_COORDINATES.items():
            if region in regional_stats:
                stats = regional_stats[region]
                map_data.append({
                    'region': region,
                    'region_ar': coords['ar_name'],
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'value': stats.get(metric, 0),
                    'sample_size': stats['sample_size'],
                    'avg_chrono_age': stats['avg_chrono_age'],
                    'avg_bio_age': stats['avg_bio_age'],
                    'avg_age_acceleration': stats['avg_age_acceleration'],
                    'hypertension_prevalence': stats.get('hypertension_prevalence', 0) * 100,
                    'diabetes_prevalence': stats.get('diabetes_prevalence', 0) * 100,
                    'obesity_prevalence': stats.get('obesity_prevalence', 0) * 100,
                })

        df = pd.DataFrame(map_data)

        # Create the map
        fig = go.Figure()

        # Add Saudi Arabia base map outline
        fig.add_trace(go.Scattergeo(
            lon=[35, 55, 55, 35, 35],
            lat=[16, 16, 32, 32, 16],
            mode='lines',
            line=dict(width=2, color='lightgray'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Create color scale based on selected metric with clear labeling
        if metric == 'avg_age_acceleration':
            colorscale = 'RdYlGn_r'  # Red for high acceleration, green for low
            colorbar_title = 'Biological Age<br>← Younger | Older →'
        elif 'prevalence' in metric:
            colorscale = 'YlOrRd'
            colorbar_title = metric.replace('_', ' ').title() + ' (%)'
        else:
            colorscale = 'Viridis'
            colorbar_title = metric.replace('_', ' ').title()

        # Add regional bubbles
        fig.add_trace(go.Scattergeo(
            lon=df['lon'],
            lat=df['lat'],
            text=df['region'],
            mode='markers+text',
            marker=dict(
                size=np.log10(df['sample_size']) * 10,  # Size based on sample size
                color=df['value'],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=colorbar_title,
                    thickness=15,
                    len=0.7,
                    x=0.95
                ),
                line=dict(width=1, color='white'),
                sizemode='diameter',
                sizeref=1,
                sizemin=10
            ),
            textposition='top center',
            textfont=dict(size=10, color='black', family='Arial Black'),
            customdata=np.column_stack((
                df['region_ar'],
                df['sample_size'],
                df['avg_chrono_age'],
                df['avg_bio_age'],
                df['avg_age_acceleration'],
                df['hypertension_prevalence'],
                df['diabetes_prevalence'],
                df['obesity_prevalence']
            )),
            hovertemplate='<b>%{text} (%{customdata[0]})</b><br>' +
                         'Sample Size: %{customdata[1]:,.0f}<br>' +
                         'Avg Chronological Age: %{customdata[2]:.1f} years<br>' +
                         'Avg Biological Age: %{customdata[3]:.1f} years<br>' +
                         '<b>Biological Age Status: ' +
                         '<span style="color: %{marker.color}">%{customdata[4]:+.2f} years</span></b><br>' +
                         '(Positive = aging faster, Negative = aging slower)<br>' +
                         '─────────────────<br>' +
                         'Hypertension: %{customdata[5]:.1f}%<br>' +
                         'Diabetes: %{customdata[6]:.1f}%<br>' +
                         'Obesity: %{customdata[7]:.1f}%' +
                         '<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text='Saudi Arabia Health Metrics by Region',
                font=dict(size=24, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            geo=dict(
                scope='asia',
                projection_type='mercator',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
                showocean=True,
                oceancolor='rgb(230, 245, 255)',
                showlakes=False,
                lakecolor='rgb(230, 245, 255)',
                center=dict(lat=24, lon=45),
                projection_scale=6,
                bgcolor='white'
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        return fig

    def create_regional_comparison(self, regional_stats: Dict[str, Any]) -> go.Figure:
        """Create bar chart comparing all regions"""

        # Prepare data
        regions = []
        age_accelerations = []
        sample_sizes = []

        for region, stats in regional_stats.items():
            regions.append(region)
            age_accelerations.append(stats['avg_age_acceleration'])
            sample_sizes.append(stats['sample_size'])

        # Sort by age acceleration
        sorted_indices = np.argsort(age_accelerations)[::-1]
        regions = [regions[i] for i in sorted_indices]
        age_accelerations = [age_accelerations[i] for i in sorted_indices]
        sample_sizes = [sample_sizes[i] for i in sorted_indices]

        # Create bar chart
        fig = go.Figure()

        # Color bars based on value (red for positive, green for negative)
        colors = ['#e74c3c' if acc > 0 else '#27ae60' for acc in age_accelerations]

        # Create labels that explain the meaning
        text_labels = []
        for acc in age_accelerations:
            if abs(acc) < 0.5:
                text_labels.append(f'{acc:+.2f}')
            elif acc > 0:
                text_labels.append(f'{acc:+.2f} (older)')
            else:
                text_labels.append(f'{acc:+.2f} (younger)')

        fig.add_trace(go.Bar(
            x=regions,
            y=age_accelerations,
            marker=dict(color=colors),
            text=text_labels,
            textposition='outside',
            customdata=sample_sizes,
            hovertemplate='<b>%{x}</b><br>' +
                         'Biological Age Status: %{y:+.2f} years<br>' +
                         '%{y} > 0: Population aging faster than expected<br>' +
                         '%{y} < 0: Population aging slower than expected<br>' +
                         'Sample Size: %{customdata:,.0f}' +
                         '<extra></extra>',
            name='Biological Age Status'
        ))

        # Add zero line with annotation
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1,
                     annotation_text="Age-appropriate (baseline)")

        fig.update_layout(
            title='Regional Biological Age Comparison',
            xaxis_title='Region',
            yaxis_title='Biological Age Difference from Expected (years)',
            height=500,
            showlegend=False,
            xaxis_tickangle=-45,
            margin=dict(b=100),
            yaxis=dict(
                gridcolor='lightgray',
                zerolinecolor='gray',
                zerolinewidth=2
            )
        )

        return fig

    def create_health_metrics_heatmap(self, regional_stats: Dict[str, Any]) -> go.Figure:
        """Create heatmap of health metrics by region"""

        # Prepare data
        regions = list(regional_stats.keys())
        metrics = ['avg_age_acceleration', 'hypertension_prevalence',
                  'diabetes_prevalence', 'obesity_prevalence', 'dyslipidemia_prevalence']
        metric_labels = ['Age Acceleration', 'Hypertension', 'Diabetes', 'Obesity', 'Dyslipidemia']

        # Create matrix
        z_values = []
        for metric in metrics:
            row = []
            for region in regions:
                value = regional_stats[region].get(metric, 0)
                if 'prevalence' in metric:
                    value *= 100  # Convert to percentage
                row.append(value)
            z_values.append(row)

        # Normalize each metric to 0-1 scale for better visualization
        z_normalized = []
        for row in z_values:
            row_array = np.array(row)
            if row_array.std() > 0:
                normalized = (row_array - row_array.min()) / (row_array.max() - row_array.min())
            else:
                normalized = row_array
            z_normalized.append(normalized.tolist())

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_normalized,
            x=regions,
            y=metric_labels,
            colorscale='RdYlGn_r',
            text=[[f'{z_values[i][j]:.1f}' for j in range(len(regions))]
                  for i in range(len(metrics))],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(
                title='Normalized Score',
                thickness=15,
                len=0.7
            ),
            hovertemplate='<b>%{y}</b><br>%{x}<br>Value: %{text}<extra></extra>'
        ))

        fig.update_layout(
            title='Regional Health Metrics Heatmap',
            xaxis_title='Region',
            yaxis_title='Health Metric',
            height=400,
            xaxis_tickangle=-45,
            margin=dict(b=100)
        )

        return fig

    def create_population_pyramid(self, data_splits: Dict, region_name: str,
                                patient_indices: list) -> go.Figure:
        """Create population pyramid for a specific region"""

        if not patient_indices:
            # Return empty figure if no patients
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {region_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        X_test = data_splits['X_test']
        X_region = X_test.iloc[patient_indices]

        # Get gender distribution
        male_count = X_region.get('gender_x_male', pd.Series([0])).sum() if 'gender_x_male' in X_region.columns else 0
        female_count = len(X_region) - male_count

        # Create age bins (we'll need to denormalize ages first)
        from utils.data_loader import DataLoader
        loader = DataLoader()
        scaler_stats = loader.load_age_scaler_stats()

        # Get age column (it might be normalized)
        if 'age_x' in X_region.columns:
            ages = loader.denormalize_age(X_region['age_x'].values, scaler_stats)
        else:
            # Generate sample ages for demo
            ages = np.random.normal(33, 18, len(X_region))

        age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
        age_labels = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

        # Count by age groups
        age_groups = pd.cut(ages, bins=age_bins, labels=age_labels)
        age_dist = pd.DataFrame({'age_group': age_groups, 'count': 1})
        age_counts = age_dist.groupby('age_group', observed=True).count()['count']

        # Create pyramid
        fig = go.Figure()

        # Male bars (left side - negative values)
        male_values = [-age_counts.iloc[i] * (male_count / len(X_region)) if i < len(age_counts) else 0
                      for i in range(len(age_labels))]

        # Female bars (right side - positive values)
        female_values = [age_counts.iloc[i] * (female_count / len(X_region)) if i < len(age_counts) else 0
                        for i in range(len(age_labels))]

        fig.add_trace(go.Bar(
            y=age_labels,
            x=male_values,
            orientation='h',
            name='Male',
            marker=dict(color='#3498db'),
            hovertemplate='Male<br>Age: %{y}<br>Count: %{x:.0f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            y=age_labels,
            x=female_values,
            orientation='h',
            name='Female',
            marker=dict(color='#e74c3c'),
            hovertemplate='Female<br>Age: %{y}<br>Count: %{x:.0f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Age Distribution - {region_name}',
            xaxis_title='Population',
            yaxis_title='Age Group',
            barmode='overlay',
            bargap=0.1,
            height=400,
            xaxis=dict(
                tickvals=[-max(abs(min(male_values)), max(female_values)),
                         -max(abs(min(male_values)), max(female_values))/2,
                         0,
                         max(abs(min(male_values)), max(female_values))/2,
                         max(abs(min(male_values)), max(female_values))],
                ticktext=[f'{abs(int(v))}' for v in [max(abs(min(male_values)), max(female_values)),
                                                      max(abs(min(male_values)), max(female_values))/2,
                                                      0,
                                                      max(abs(min(male_values)), max(female_values))/2,
                                                      max(abs(min(male_values)), max(female_values))]]
            )
        )

        return fig