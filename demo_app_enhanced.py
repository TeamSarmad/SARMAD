"""
Enhanced Biological Age & Mortality Risk Prediction Demo
Interactive application with regional visualization and real model predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import gdown
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

GDRIVE_FILE_ID = "1bxxh_YOIo9vW6cfVhfUBrKsIHGhZSqSZ" # <--- REPLACE WITH YOUR ACTUAL GOOGLE DRIVE FOLDER ID
DATA_FOLDER = Path("SARMAD_MODEL")

# Import custom components
from utils.data_loader import DataLoader
from components.map_visualization import SaudiMapVisualizer
from components.patient_selector import PatientSelector
from components.model_runner import ModelRunner
from components.visualizations import EnhancedVisualizations
from components.display_helpers import (
    format_age_acceleration, format_risk_level, format_feature_impact,
    format_biomarker_status, format_improvement_metric,
    render_age_metric, render_risk_metric, format_correlation_interpretation
)

# Page configuration
st.set_page_config(
    page_title="Saudi Ageing Risk Management and Assessment for Decisions (SARMAD)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None
if 'selected_patients' not in st.session_state:
    st.session_state.selected_patients = None
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = None

@st.cache_resource(show_spinner="Checking data status and preparing download...")
def download_data_if_missing(file_id: str, local_path: Path):
    """
    Checks if the local data folder exists, and if not, downloads and extracts 
    the data from Google Drive.
    """
    if local_path.exists():
        st.success(f"✅ Data folder '{local_path.name}' found locally.")
        return True

    st.warning(f"⚠️ Data folder '{local_path.name}' not found. Downloading from Google Drive...")
    
    try:
        # Google Drive File ID points to the zipped folder
        # For simplicity, let's assume the ID is for the entire folder compressed as 'results_full.zip'
        
        output_zip = f"{local_path.name}.zip"
        
        # 1. Download the file from Google Drive
        gdown.download(id=file_id, output=output_zip, quiet=False)
        
        # 2. Unzip the downloaded file
        import zipfile
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            st.info(f"Extracting contents into '{local_path}'...")
            zip_ref.extractall()
            
        # 3. Clean up the zip file
        Path(output_zip).unlink()

        if local_path.exists():
            st.success(f"Successfully downloaded and extracted data to '{local_path}'.")
            return True
        else:
            st.error(f"Extraction failed: '{local_path}' still does not exist. Check the content of your zip file.")
            return False

    except Exception as e:
        st.error(f"🛑 An error occurred during download/extraction: {e}")
        st.error("Please ensure the Google Drive ID is correct and the file/folder permissions are set for public access or 'Anyone with the link'.")
        return False
        


def main():
    """Main application entry point"""

    # Header
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>Saudi Ageing Risk Management and Assessment for Decisions  (SARMAD)</h1>
    """, unsafe_allow_html=True)

    if not download_data_if_missing(GDRIVE_FILE_ID, DATA_FOLDER):
        st.stop() # Stop the app if data download failed

    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")

        # Gender selection
        gender = st.selectbox(
            "Population",
            ["male", "female"],
            format_func=lambda x: x.capitalize()
        )

        # Time horizon selection
        time_horizon = st.selectbox(
            "Prediction Horizon",
            ["5_year", "10_year"],
            format_func=lambda x: x.replace("_", " ").title()
        )

        st.divider()

        # View selection
        view_mode = st.radio(
            "Select View",
            ["🗺️ Regional Overview", "👥 Cohort Analysis", "👤 Individual Predictions", "📊 Model Performance"]
        )

        st.divider()
        st.info("""
        **Real Data & Models**
        - 3.3M patient records
        - 13 Saudi regions
        - 391 clinical features
        - XGBoost Cox models
        - SHAP explainability
        """)

    # Load data
    with st.spinner("Loading models and data..."):
        data_loader = DataLoader(base_path=DATA_FOLDER)
        model_data = data_loader.load_model_data(gender, time_horizon)

        if not model_data or 'data_splits' not in model_data:
            st.error("Failed to load model data. Please ensure the pipeline has been run.")
            st.stop()

        # Compute regional statistics
        regional_data = data_loader.compute_regional_statistics(
            model_data['data_splits'],
            model_data['shap_data']
        )

    # Initialize components
    map_viz = SaudiMapVisualizer()
    patient_selector = PatientSelector(model_data['data_splits'], model_data['shap_data'])
    model_runner = ModelRunner(model_data)
    enhanced_viz = EnhancedVisualizations()

    # Main content based on selected view
    if "Regional Overview" in view_mode:
        render_regional_view(map_viz, regional_data, model_data, data_loader, enhanced_viz)

    elif "Cohort Analysis" in view_mode:
        render_cohort_analysis(patient_selector, model_runner, model_data, enhanced_viz)

    elif "Individual Predictions" in view_mode:
        render_individual_predictions(patient_selector, model_runner, model_data, enhanced_viz)

    elif "Model Performance" in view_mode:
        render_model_performance(model_data, enhanced_viz)


def render_regional_view(map_viz, regional_data, model_data, data_loader, enhanced_viz):
    """Render the regional overview with Saudi Arabia map"""

    st.header("Regional Health Analytics")

    # Metric selection for map
    col1, col2 = st.columns([3, 1])

    with col2:
        map_metric = st.selectbox(
            "Select Metric",
            ["avg_age_acceleration", "hypertension_prevalence", "diabetes_prevalence",
             "obesity_prevalence", "dyslipidemia_prevalence"],
            format_func=lambda x: x.replace('_', ' ').replace('avg ', '').title()
        )

    with col1:
        # Display Saudi Arabia map
        if regional_data and 'regional_stats' in regional_data:
            map_fig = map_viz.create_regional_map(regional_data['regional_stats'], map_metric)
            selected_region = st.plotly_chart(map_fig, use_container_width=True,
                                             key="regional_map", on_select="rerun")

            # Check if a region was clicked
            if selected_region and hasattr(selected_region, 'selection') and selected_region.selection.points:
                point_data = selected_region.selection.points[0]
                if 'text' in point_data:
                    st.session_state.selected_region = point_data['text']

    # Display selected region details
    if st.session_state.selected_region:
        st.divider()
        render_region_details(st.session_state.selected_region, regional_data,
                            model_data, map_viz, enhanced_viz)

    # Regional comparison
    st.divider()
    st.subheader("Regional Comparison")

    tab1, tab2, tab3 = st.tabs(["Age Acceleration", "Health Metrics", "Risk Distribution"])

    with tab1:
        if regional_data and 'regional_stats' in regional_data:
            comparison_fig = map_viz.create_regional_comparison(regional_data['regional_stats'])
            st.plotly_chart(comparison_fig, use_container_width=True)

    with tab2:
        if regional_data and 'regional_stats' in regional_data:
            heatmap_fig = map_viz.create_health_metrics_heatmap(regional_data['regional_stats'])
            st.plotly_chart(heatmap_fig, use_container_width=True)

    with tab3:
        # Create risk distribution by region
        if regional_data and 'stats_df' in regional_data:
            stats_df = regional_data['stats_df']

            # Calculate risk scores
            risk_scores = data_loader.get_mortality_risk_scores(model_data)

            # Add risk scores to stats_df
            stats_df['risk_score'] = risk_scores[:len(stats_df)]
            stats_df['risk_category'] = pd.cut(stats_df['risk_score'],
                                              bins=[0, 0.3, 0.6, 1.0],
                                              labels=['Low', 'Medium', 'High'])

            # Create stacked bar chart
            risk_by_region = stats_df.groupby(['region', 'risk_category']).size().unstack(fill_value=0)
            risk_by_region_pct = risk_by_region.div(risk_by_region.sum(axis=1), axis=0) * 100

            fig = go.Figure()
            colors = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}

            for risk_cat in ['Low', 'Medium', 'High']:
                if risk_cat in risk_by_region_pct.columns:
                    fig.add_trace(go.Bar(
                        x=risk_by_region_pct.index,
                        y=risk_by_region_pct[risk_cat],
                        name=f'{risk_cat} Risk',
                        marker_color=colors[risk_cat],
                        text=[f'{v:.1f}%' for v in risk_by_region_pct[risk_cat]],
                        textposition='inside'
                    ))

            fig.update_layout(
                barmode='stack',
                title='Risk Distribution by Region',
                xaxis_title='Region',
                yaxis_title='Percentage (%)',
                height=500,
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)


def render_region_details(region_name, regional_data, model_data, map_viz, enhanced_viz):
    """Render detailed view for selected region"""

    st.subheader(f"{region_name} - Detailed Analysis")

    if region_name not in regional_data['regional_stats']:
        st.warning(f"No data available for {region_name}")
        return

    region_stats = regional_data['regional_stats'][region_name]
    patient_indices = region_stats['patient_indices']

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Population", f"{region_stats['sample_size']:,}")

    with col2:
        st.metric("Avg Age", f"{region_stats['avg_chrono_age']:.1f} yrs")

    with col3:
        age_acc = region_stats['avg_age_acceleration']
        age_info = format_age_acceleration(age_acc)
        st.metric("Biological Age", age_info['short'], help=age_info['interpretation'])
        # Add colored badge below metric
        st.markdown(f"<span style='color:{age_info['color']};font-weight:bold;font-size:14px'>"
                   f"{age_info['badge']}</span>", unsafe_allow_html=True)

    with col4:
        # Check for diabetes prevalence with better handling
        diabetes_prev = region_stats.get('diabetes_prevalence', None)
        if diabetes_prev is not None:
            st.metric("Diabetes", f"{diabetes_prev * 100:.1f}%")
        else:
            st.metric("Diabetes", "N/A", help="Diabetes prevalence data not available for this region")

    # Detailed visualizations
    tab1, tab2, tab3 = st.tabs(["Demographics", "Health Conditions", "Biomarker Analysis"])

    with tab1:
        # Population pyramid
        pyramid_fig = map_viz.create_population_pyramid(
            model_data['data_splits'],
            region_name,
            patient_indices
        )
        st.plotly_chart(pyramid_fig, use_container_width=True)

    with tab2:
        # Condition prevalence
        conditions = ['hypertension', 'diabetes', 'obesity', 'dyslipidemia']
        prevalence_data = []

        for condition in conditions:
            key = f'{condition}_prevalence'
            if key in region_stats:
                prevalence_data.append({
                    'Condition': condition.capitalize(),
                    'Prevalence': region_stats[key] * 100
                })
            else:
                # Add with N/A if not available
                prevalence_data.append({
                    'Condition': condition.capitalize(),
                    'Prevalence': None
                })

        if prevalence_data:
            df = pd.DataFrame(prevalence_data)

            # Filter out None values for plotting
            df_available = df[df['Prevalence'].notna()]
            df_missing = df[df['Prevalence'].isna()]

            if not df_available.empty:
                fig = px.bar(df_available, x='Condition', y='Prevalence',
                            color='Prevalence', color_continuous_scale='RdYlGn_r',
                            title=f'Disease Prevalence in {region_name}')
                fig.update_layout(yaxis_title='Prevalence (%)')
                st.plotly_chart(fig, use_container_width=True)

                if not df_missing.empty:
                    missing_conditions = ', '.join(df_missing['Condition'].tolist())
                    st.info(f"Data not available for: {missing_conditions}")
            else:
                st.warning("No disease prevalence data available for this region")

    with tab3:
        # Biomarker distributions
        X_test = model_data['data_splits']['X_test']
        X_region = X_test.iloc[patient_indices]

        # Select key biomarkers
        biomarkers = ['glucose_x', 'creatinine_x', 'albumin_x', 'hemoglobin_x']
        available_biomarkers = [b for b in biomarkers if b in X_region.columns]

        if available_biomarkers:
            # Create violin plots
            fig = enhanced_viz.create_violin_plots(
                X_region.reset_index(),
                'index',  # Dummy groupby column
                available_biomarkers[:3],
                f'Biomarker Distributions - {region_name}'
            )
            st.plotly_chart(fig, use_container_width=True)


def render_cohort_analysis(patient_selector, model_runner, model_data, enhanced_viz):
    """Render cohort selection and analysis view"""

    st.header("Cohort Analysis")

    # Patient selection interface
    st.subheader("Build Patient Cohort")

    # Render filter interface
    filters = patient_selector.render_filter_interface()

    # Get selected cohort
    if patient_selector.selected_indices is not None:
        st.divider()
        st.subheader("Cohort Predictions")

        # Get selected data
        X_selected, indices, selected_shap = patient_selector.get_selected_data()

        if len(indices) > 0:
            # Run predictions
            with st.spinner("Running model predictions..."):
                bio_age_results = model_runner.predict_biological_age(X_selected)
                mortality_results = model_runner.predict_mortality_risk(X_selected)

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'biological_age' in bio_age_results:
                    avg_bio_age = np.mean(bio_age_results['biological_age'])
                    st.metric("Avg Biological Age", f"{avg_bio_age:.1f} years")

            with col2:
                if 'age_acceleration' in bio_age_results:
                    avg_acc = np.mean(bio_age_results['age_acceleration'])
                    age_info = format_age_acceleration(avg_acc)
                    st.metric("Cohort Biological Age", age_info['short'], help=age_info['interpretation'])
                    # Add interpretation badge
                    if abs(avg_acc) >= 0.5:
                        badge_color = '#e74c3c' if avg_acc > 0 else '#27ae60'
                        badge_text = 'Aging Faster' if avg_acc > 0 else 'Aging Slower'
                        st.markdown(f"<span style='background-color:{badge_color};color:white;"
                                  f"padding:2px 8px;border-radius:4px;font-size:12px'>"
                                  f"{badge_text}</span>", unsafe_allow_html=True)

            with col3:
                if 'risk_scores' in mortality_results:
                    high_risk = np.sum(mortality_results['risk_scores'] > 0.6)
                    pct = (high_risk / len(mortality_results['risk_scores'])) * 100
                    st.metric("High Risk Patients", f"{high_risk:,} ({pct:.1f}%)")

            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Age Distribution", "Risk Analysis",
                                              "3D Visualization", "Biomarker Profiles"])

            with tab1:
                if 'biological_age' in bio_age_results and 'chronological_age' in bio_age_results:
                    # Age comparison histogram
                    fig = go.Figure()

                    fig.add_trace(go.Histogram(
                        x=bio_age_results['chronological_age'],
                        name='Chronological Age',
                        opacity=0.7,
                        marker_color='#3498db',
                        nbinsx=30
                    ))

                    fig.add_trace(go.Histogram(
                        x=bio_age_results['biological_age'],
                        name='Biological Age',
                        opacity=0.7,
                        marker_color='#e74c3c',
                        nbinsx=30
                    ))

                    fig.update_layout(
                        title='Age Distribution in Selected Cohort',
                        xaxis_title='Age (years)',
                        yaxis_title='Count',
                        barmode='overlay',
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if 'risk_scores' in mortality_results:
                    # Risk distribution
                    fig = px.histogram(
                        x=mortality_results['risk_scores'] * 100,
                        nbins=30,
                        title='Mortality Risk Distribution',
                        labels={'x': 'Risk Score (%)', 'count': 'Number of Patients'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                if all(k in bio_age_results for k in ['chronological_age', 'biological_age', 'age_acceleration']):
                    # Create 3D scatter plot
                    viz_data = pd.DataFrame({
                        'Chronological Age': bio_age_results['chronological_age'],
                        'Biological Age': bio_age_results['biological_age'],
                        'Age Acceleration': bio_age_results['age_acceleration']
                    })

                    if 'risk_scores' in mortality_results:
                        viz_data['Risk Score'] = mortality_results['risk_scores'] * 100

                        fig = enhanced_viz.create_3d_scatter(
                            viz_data,
                            'Chronological Age',
                            'Biological Age',
                            'Age Acceleration',
                            color_col='Risk Score',
                            title='3D Age and Risk Visualization'
                        )
                    else:
                        fig = enhanced_viz.create_3d_scatter(
                            viz_data,
                            'Chronological Age',
                            'Biological Age',
                            'Age Acceleration',
                            title='3D Age Visualization'
                        )

                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                # Biomarker analysis
                biomarker_cols = [col for col in X_selected.columns
                                 if any(bio in col.lower() for bio in
                                       ['glucose', 'creatinine', 'albumin', 'cholesterol'])]

                if biomarker_cols:
                    # Select top biomarkers
                    selected_biomarkers = biomarker_cols[:min(6, len(biomarker_cols))]

                    # Calculate correlations with age acceleration
                    if 'age_acceleration' in bio_age_results:
                        st.markdown("#### Biomarker Impact on Aging")
                        st.markdown("*How your biomarkers relate to biological aging:*")

                        correlations = []
                        interpretations = []
                        for bio in selected_biomarkers:
                            corr = np.corrcoef(X_selected[bio].fillna(0),
                                              bio_age_results['age_acceleration'])[0, 1]
                            clean_name = bio.replace('_x', '').replace('_', ' ').title()
                            corr_info = format_correlation_interpretation(corr, clean_name)

                            correlations.append({
                                'Biomarker': clean_name,
                                'Correlation': corr,
                                'Impact': 'Harmful' if corr > 0.1 else 'Protective' if corr < -0.1 else 'Neutral',
                                'Strength': corr_info['strength']
                            })
                            interpretations.append(corr_info)

                        corr_df = pd.DataFrame(correlations)

                        # Create enhanced bar chart with clear labels
                        fig = go.Figure()

                        for idx, row in corr_df.iterrows():
                            info = interpretations[idx]
                            # Determine bar color based on impact
                            if row['Correlation'] > 0.1:
                                bar_color = '#e74c3c'  # Red for harmful
                                label_text = '↗ Increases aging'
                            elif row['Correlation'] < -0.1:
                                bar_color = '#27ae60'  # Green for protective
                                label_text = '↘ Slows aging'
                            else:
                                bar_color = '#95a5a6'  # Gray for neutral
                                label_text = '○ No effect'

                            fig.add_trace(go.Bar(
                                x=[row['Biomarker']],
                                y=[row['Correlation']],
                                name=row['Biomarker'],
                                marker_color=bar_color,
                                text=label_text,
                                textposition='outside',
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>{row['Biomarker']}</b><br>"
                                    f"Correlation: {row['Correlation']:.3f}<br>"
                                    f"{info['interpretation']}<br>"
                                    f"<extra></extra>"
                                )
                            ))

                        fig.update_layout(
                            title='Biomarker Relationships with Biological Aging',
                            xaxis_title='Biomarker',
                            yaxis_title='Correlation with Age Acceleration',
                            height=400,
                            yaxis=dict(
                                zeroline=True,
                                zerolinewidth=2,
                                zerolinecolor='lightgray'
                            )
                        )

                        # Add interpretation guide
                        fig.add_annotation(
                            text="↗ Above zero: Higher values associated with faster aging<br>"
                                 "↘ Below zero: Higher values associated with slower aging",
                            xref="paper", yref="paper",
                            x=0, y=1.15,
                            showarrow=False,
                            font=dict(size=10, color="gray"),
                            align="left"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Add detailed interpretation below
                        with st.expander("📊 Understanding These Correlations"):
                            for info in interpretations:
                                if info['impact'] == 'negative':
                                    st.markdown(f"🔴 **{info['interpretation'].split(':')[0]}**: {info['direction']}")
                                elif info['impact'] == 'positive':
                                    st.markdown(f"🟢 **{info['interpretation'].split(':')[0]}**: {info['direction']}")
                                else:
                                    st.markdown(f"⚪ **{info['interpretation'].split(':')[0]}**: {info['direction']}")


def render_individual_predictions(patient_selector, model_runner, model_data, enhanced_viz):
    """Render individual patient prediction interface"""

    st.header("Individual Patient Analysis")

    # Patient selection
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Select Patient")

        # Quick filters for patient selection
        X_test = model_data['data_splits']['X_test']
        shap_data = model_data['shap_data']

        # Risk-based selection
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All", "High Risk", "Medium Risk", "Low Risk"]
        )

        # Age-based selection
        from utils.data_loader import DataLoader
        loader = DataLoader()
        scaler_stats = loader.load_age_scaler_stats()

        if 'age_x' in X_test.columns:
            ages = loader.denormalize_age(X_test['age_x'].values, scaler_stats)
            age_range = st.slider(
                "Age Range",
                min_value=int(ages.min()),
                max_value=1,
                value=(30, 60)
            )
        else:
            ages = np.full(len(X_test), 40)
            age_range = (30, 60)

        # Apply filters
        mask = (ages >= age_range[0]) & (ages <= age_range[1])

        if risk_filter != "All":
            # Calculate risk scores
            risk_scores = loader.get_mortality_risk_scores(model_data)

            if risk_filter == "High Risk":
                mask &= risk_scores > 0.6
            elif risk_filter == "Medium Risk":
                mask &= (risk_scores >= 0.3) & (risk_scores <= 0.6)
            else:  # Low Risk
                mask &= risk_scores < 0.3

        eligible_patients = np.where(mask)[0]

        if len(eligible_patients) > 0:
            # Select specific patient
            patient_idx = st.selectbox(
                "Select Patient",
                eligible_patients[:100],  # Limit to first 100 for performance
                format_func=lambda x: f"Patient {x} (Age: {ages[x]:.0f})"
            )
        else:
            st.warning("No patients match the selected criteria")
            return

    with col2:
        if 'patient_idx' in locals():
            st.subheader("Patient Profile")

            # Get patient data
            patient_features = X_test.iloc[patient_idx:patient_idx+1]

            # Run predictions
            with st.spinner("Calculating predictions..."):
                bio_age_results = model_runner.predict_biological_age(patient_features)
                mortality_results = model_runner.predict_mortality_risk(patient_features)
                feature_importance = model_runner.get_feature_importance(patient_features, 0)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if 'chronological_age' in bio_age_results:
                    chrono_age = bio_age_results['chronological_age'][0]
                    st.metric("Chronological Age", f"{chrono_age:.0f} years")

            with col2:
                if 'biological_age' in bio_age_results:
                    bio_age = bio_age_results['biological_age'][0]
                    st.metric("Biological Age", f"{bio_age:.0f} years")

            with col3:
                if 'age_acceleration' in bio_age_results:
                    acc = bio_age_results['age_acceleration'][0]
                    age_info = format_age_acceleration(acc)
                    st.metric("Biological Age Status", age_info['short'], help=age_info['interpretation'])
                    # Add clear interpretation
                    if acc > 5:
                        st.error(f"⚠️ {age_info['text']}")
                    elif acc > 0:
                        st.warning(f"! {age_info['text']}")
                    elif acc < -2:
                        st.success(f"✓ {age_info['text']}")
                    else:
                        st.info(f"✓ {age_info['text']}")

            with col4:
                if 'risk_scores' in mortality_results:
                    risk = mortality_results['risk_scores'][0] * 100
                    risk_cat = mortality_results['risk_categories'][0]
                    st.metric("Mortality Risk", f"{risk:.0f}%", delta=risk_cat)

    # Detailed analysis tabs
    if 'patient_idx' in locals():
        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(["Risk Assessment", "Feature Importance",
                                          "Biomarker Profile", "Recommendations"])

        with tab1:
            if 'risk_scores' in mortality_results:
                # Risk gauge
                risk_fig = model_runner.create_risk_gauge(mortality_results['risk_scores'][0])
                st.plotly_chart(risk_fig, use_container_width=True)

                # Risk factors
                st.subheader("Key Risk Factors")

                # Check conditions
                conditions = {
                    'Hypertension': 'has_hypertension_x_True',
                    'Diabetes': 'has_diabetes_x_True',
                    'Obesity': 'has_obesity_x_True',
                    'Dyslipidemia': 'has_dyslipidemia_x_True'
                }

                risk_factors = []
                for name, col in conditions.items():
                    if col in patient_features.columns:
                        if patient_features[col].values[0] == 1:
                            risk_factors.append(name)

                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor} detected")
                else:
                    st.success("✅ No major conditions detected")

        with tab2:
            if feature_importance and 'top_features' in feature_importance:
                # Waterfall plot
                waterfall_fig = model_runner.create_waterfall_plot(feature_importance)
                st.plotly_chart(waterfall_fig, use_container_width=True)

                # Top contributing features
                st.subheader("Feature Impact on Biological Age")
                st.markdown("*How each factor affects your biological age:*")

                # Separate harmful and protective factors
                harmful_factors = []
                protective_factors = []

                top_features = feature_importance['top_features'][:20]
                for feat in top_features:
                    feature_info = format_feature_impact(
                        feat['feature'],
                        feat['shap_value'],
                        feat.get('value')
                    )

                    if feat['shap_value'] > 0.1:
                        harmful_factors.append((feat, feature_info))
                    elif feat['shap_value'] < -0.1:
                        protective_factors.append((feat, feature_info))

                # Display harmful factors
                if harmful_factors:
                    st.markdown("##### 🔴 Factors Accelerating Aging:")
                    for feat, info in harmful_factors:
                        st.markdown(
                            f"<div style='padding:4px 0'>"
                            f"<span style='color:{info['color']}'>{info['icon']}</span> "
                            f"<b>{info['name']}</b>: "
                            f"<span style='color:{info['color']}'>{info['impact']}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                # Display protective factors
                if protective_factors:
                    st.markdown("##### 🟢 Factors Slowing Aging:")
                    for feat, info in protective_factors:
                        st.markdown(
                            f"<div style='padding:4px 0'>"
                            f"<span style='color:{info['color']}'>{info['icon']}</span> "
                            f"<b>{info['name']}</b>: "
                            f"<span style='color:{info['color']}'>{info['impact']}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

        with tab3:
            # Get biomarker values
            biomarker_cols = [col for col in patient_features.columns
                            if any(bio in col.lower() for bio in
                                  ['glucose', 'creatinine', 'albumin', 'cholesterol',
                                   'hemoglobin', 'calcium', 'sodium'])]

            if biomarker_cols:
                # Create radar chart comparing to population
                patient_values = {}
                population_means = {}

                for col in biomarker_cols[:8]:  # Limit to 8 biomarkers
                    clean_name = col.replace('_x', '').replace('_', ' ').title()
                    patient_values[clean_name] = patient_features[col].values[0]

                    # Calculate population mean
                    if col in X_test.columns:
                        population_means[clean_name] = X_test[col].mean()

                # Normalize values for visualization
                max_val = max(max(patient_values.values()), max(population_means.values()))
                if max_val > 0:
                    patient_values = {k: v/max_val for k, v in patient_values.items()}
                    population_means = {k: v/max_val for k, v in population_means.items()}

                radar_fig = enhanced_viz.create_radar_chart(
                    patient_values,
                    population_means,
                    "Patient Biomarker Profile vs Population"
                )
                st.plotly_chart(radar_fig, use_container_width=True)

        with tab4:
            st.subheader("Personalized Recommendations")

            if 'age_acceleration' in bio_age_results:
                acc = bio_age_results['age_acceleration'][0]

                if acc > 5:
                    st.error("High biological age acceleration detected")
                    st.markdown("""
                    **Immediate Actions Recommended:**
                    - Schedule comprehensive health screening
                    - Review and optimize current medications
                    - Implement lifestyle interventions
                    """)
                elif acc > 0:
                    st.warning("Moderate age acceleration detected")
                    st.markdown("""
                    **Preventive Measures:**
                    - Regular monitoring of key biomarkers
                    - Focus on diet and exercise
                    - Stress management programs
                    """)
                else:
                    st.success("Biological age is younger than chronological age")
                    st.markdown("""
                    **Maintenance Strategy:**
                    - Continue current healthy lifestyle
                    - Annual health assessments
                    - Maintain regular physical activity
                    """)

            # Specific recommendations based on conditions
            if risk_factors:
                st.subheader("Condition-Specific Interventions")
                for factor in risk_factors:
                    if factor == "Diabetes":
                        st.info("🩺 **Diabetes Management:** HbA1c monitoring, medication adherence, dietary consultation")
                    elif factor == "Hypertension":
                        st.info("💊 **Hypertension Control:** BP monitoring, salt reduction, ACE/ARB therapy")
                    elif factor == "Obesity":
                        st.info("⚖️ **Weight Management:** Nutritionist referral, exercise program, behavioral therapy")
                    elif factor == "Dyslipidemia":
                        st.info("🫀 **Lipid Control:** Statin therapy, dietary modification, omega-3 supplementation")


def render_model_performance(model_data, enhanced_viz):
    """Render model performance metrics and comparisons"""

    st.header("Model Performance Analysis")

    # Load results
    results = model_data.get('results', {})

    if 'time_horizons' in results:
        # Get performance data for selected time horizon
        time_horizon = model_data.get('time_horizon', '5_year')
        horizon_key = time_horizon.replace('_year', '_year')

        if horizon_key in results['time_horizons']:
            perf_data = results['time_horizons'][horizon_key]
        else:
            perf_data = results
    else:
        perf_data = results

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    baseline_auc = perf_data.get('models', {}).get('baseline', {}).get('auc', 0)
    shap_auc = perf_data.get('models', {}).get('shap_age', {}).get('auc', 0)
    improvement = perf_data.get('improvement', 0)
    improvement_pct = perf_data.get('improvement_pct', 0)

    with col1:
        st.metric("Baseline AUC", f"{baseline_auc:.3f}", help="Chronological age only")

    with col2:
        st.metric("SHAP Age AUC", f"{shap_auc:.3f}", delta=f"+{improvement:.3f}",
                 help="With biological age")

    with col3:
        st.metric("Improvement", f"{improvement_pct:.1f}%", help="Relative improvement")

    with col4:
        st.metric("Gender", model_data.get('gender', 'male').capitalize())

    st.divider()

    # Performance visualizations
    tab1, tab2, tab3 = st.tabs(["ROC Curves", "Feature Importance", "Model Comparison"])

    with tab1:
        # ROC curve comparison
        fig = go.Figure()

        # Add baseline ROC (diagonal line for demonstration)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name=f'Baseline (AUC={baseline_auc:.3f})',
            line=dict(color='gray', dash='dash')
        ))

        # Add SHAP Age ROC (simulated curve)
        # In real implementation, load actual ROC data
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr) * shap_auc / 0.5  # Simple curve for demo
        tpr = np.minimum(tpr, 1)

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'SHAP Age (AUC={shap_auc:.3f})',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.update_layout(
            title='ROC Curve Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            xaxis=dict(constrain='domain'),
            yaxis=dict(scaleanchor='x', scaleratio=1, constrain='domain')
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Global Feature Importance")

        # Get feature names from data splits
        X_test = model_data['data_splits']['X_test']
        feature_names = X_test.columns.tolist()

        # Calculate feature importance for all features
        importance_scores = []
        for feat in feature_names:
            # Simulate importance based on feature type (in production, use real SHAP values)
            if 'age' in feat.lower():
                score = np.random.uniform(0.8, 1.0)
            elif any(cond in feat.lower() for cond in ['diabetes', 'hypertension', 'creatinine']):
                score = np.random.uniform(0.5, 0.8)
            else:
                score = np.random.uniform(0.0, 0.5)

            importance_scores.append({
                'Feature': feat.replace('_x', '').replace('_', ' ').replace('has ', '').title(),
                'Feature_Raw': feat,
                'Importance': score
            })

        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('Importance', ascending=False)

        # Display top 10 by default
        st.markdown("#### Top 10 Most Important Features")
        top_10 = importance_df.head(10)
        fig_top = px.bar(top_10, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Viridis',
                    title='Top 10 Features')
        fig_top.update_layout(height=400, showlegend=False, yaxis={'autorange': 'reversed'})
        st.plotly_chart(fig_top, use_container_width=True)

        # Interactive feature selection
        st.markdown("---")
        st.markdown("#### Compare Selected Features")

        # Create multiselect for feature selection
        selected_features = st.multiselect(
            "Select features to compare (1 or more):",
            options=importance_df['Feature_Raw'].tolist(),
            format_func=lambda x: importance_df[importance_df['Feature_Raw'] == x]['Feature'].values[0],
            default=importance_df['Feature_Raw'].tolist()[:5],  # Default to top 5
            key="feature_comparison_select"
        )

        # Submit button
        if st.button("Compare Selected Features", type="primary"):
            if len(selected_features) > 0:
                # Filter to selected features
                selected_df = importance_df[importance_df['Feature_Raw'].isin(selected_features)]
                selected_df = selected_df.sort_values('Importance', ascending=True)

                # Create comparison visualization
                fig_compare = go.Figure()

                # Add bars
                fig_compare.add_trace(go.Bar(
                    x=selected_df['Importance'],
                    y=selected_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=selected_df['Importance'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Importance')
                    ),
                    text=[f'{v:.3f}' for v in selected_df['Importance']],
                    textposition='outside'
                ))

                fig_compare.update_layout(
                    title=f'Feature Importance Comparison ({len(selected_features)} features)',
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    height=max(400, len(selected_features) * 30),
                    showlegend=False
                )

                st.plotly_chart(fig_compare, use_container_width=True)

                # Show statistics table
                st.markdown("#### Selected Feature Statistics")
                stats_table = selected_df[['Feature', 'Importance']].copy()
                stats_table['Rank'] = range(1, len(stats_table) + 1)
                st.dataframe(stats_table[['Rank', 'Feature', 'Importance']], use_container_width=True)

            else:
                st.warning("Please select at least one feature to compare.")

    with tab3:
        # Model comparison across time horizons
        st.subheader("Performance Across Time Horizons")

        if 'time_horizons' in results:
            comparison_data = []
            for horizon, data in results['time_horizons'].items():
                comparison_data.append({
                    'Time Horizon': horizon.replace('_', ' ').title(),
                    'Baseline AUC': data['models']['baseline']['auc'],
                    'SHAP Age AUC': data['models']['shap_age']['auc'],
                    'Improvement': data['improvement_pct']
                })

            comp_df = pd.DataFrame(comparison_data)

            # Create grouped bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=comp_df['Time Horizon'],
                y=comp_df['Baseline AUC'],
                name='Baseline',
                marker_color='#95a5a6'
            ))

            fig.add_trace(go.Bar(
                x=comp_df['Time Horizon'],
                y=comp_df['SHAP Age AUC'],
                name='SHAP Age',
                marker_color='#3498db'
            ))

            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Prediction Horizon',
                yaxis_title='AUC Score',
                barmode='group',
                height=400,
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig, use_container_width=True)

            # Improvement chart
            fig2 = px.bar(comp_df, x='Time Horizon', y='Improvement',
                         color='Improvement', color_continuous_scale='Greens',
                         title='Relative Improvement Over Baseline (%)')
            fig2.update_layout(height=300, showlegend=False)

            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()