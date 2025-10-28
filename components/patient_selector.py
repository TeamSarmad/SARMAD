"""
Advanced Patient Selection and Filtering System
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px


class PatientSelector:
    """Advanced patient filtering and selection interface"""

    def __init__(self, data_splits: Dict, shap_data: Dict):
        self.X_test = data_splits['X_test']
        self.y_test = data_splits.get('y_test', np.zeros(len(self.X_test)))
        self.shap_data = shap_data
        self.selected_indices = None
        self.filters = {}

        # Extract feature categories
        self._categorize_features()

    def _categorize_features(self):
        """Categorize features for better organization"""
        columns = self.X_test.columns.tolist()

        self.feature_categories = {
            'demographics': [],
            'regions': [],
            'conditions': [],
            'biomarkers': [],
            'medications': [],
            'utilization': [],
            'others': []
        }

        for col in columns:
            col_lower = col.lower()
            if 'region_en' in col_lower:  # Handle both 'cat__region_en_x_' and 'region_en_x_'
                self.feature_categories['regions'].append(col)
            elif 'has_' in col or any(cond in col for cond in ['hypertension', 'diabetes', 'obesity', 'dyslipidemia']):
                self.feature_categories['conditions'].append(col)
            elif 'gender' in col_lower or 'age' in col_lower or 'nationality' in col_lower:
                self.feature_categories['demographics'].append(col)
            elif any(med in col_lower for med in ['medication', 'drug', 'gliptin', 'formin', 'statin', 'blocker']):
                self.feature_categories['medications'].append(col)
            elif any(util in col_lower for util in ['visit', 'outpatient', 'inpatient', 'emergency']):
                self.feature_categories['utilization'].append(col)
            elif any(bio in col_lower for bio in ['glucose', 'creatinine', 'cholesterol', 'hemoglobin',
                                                   'albumin', 'bilirubin', 'calcium', 'sodium', 'potassium',
                                                   'wbc', 'rdw', 'lymphocyte', 'iron', 'tsh', 'hba1c']):
                self.feature_categories['biomarkers'].append(col)
            else:
                self.feature_categories['others'].append(col)

    def render_filter_interface(self) -> Dict[str, Any]:
        """Render the patient filtering interface"""

        st.markdown("### Patient Cohort Builder")

        # Create tabs for different filter categories
        tabs = st.tabs(["Demographics", "Clinical Conditions", "Biomarkers",
                       "Healthcare Utilization", "Medications", "Advanced"])

        with tabs[0]:  # Demographics
            self._render_demographic_filters()

        with tabs[1]:  # Clinical Conditions
            self._render_condition_filters()

        with tabs[2]:  # Biomarkers
            self._render_biomarker_filters()

        with tabs[3]:  # Healthcare Utilization
            self._render_utilization_filters()

        with tabs[4]:  # Medications
            self._render_medication_filters()

        with tabs[5]:  # Advanced
            self._render_advanced_filters()

        # Apply filters button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("Apply Filters", type="primary", use_container_width=True):
                self.selected_indices = self._apply_filters()
                st.success(f"Selected {len(self.selected_indices):,} patients")

        with col2:
            if st.button("Reset Filters", use_container_width=True):
                self.filters = {}
                self.selected_indices = None
                st.rerun()

        with col3:
            if st.button("Save Cohort", use_container_width=True):
                if self.selected_indices is not None:
                    self._save_cohort()

        # Show cohort statistics
        if self.selected_indices is not None:
            self._display_cohort_statistics()

        return self.filters

    def _render_demographic_filters(self):
        """Render demographic filter controls"""

        col1, col2 = st.columns(2)

        with col1:
            # Age filter (need to denormalize first)
            from utils.data_loader import DataLoader
            loader = DataLoader()
            scaler_stats = loader.load_age_scaler_stats()

            if 'age_x' in self.X_test.columns:
                ages = loader.denormalize_age(self.X_test['age_x'].values, scaler_stats)
                age_range = st.slider(
                    "Age Range",
                    min_value=int(ages.min()),
                    max_value=int(ages.max()),
                    value=(int(ages.min()), int(ages.max())),
                    key="age_filter"
                )
                self.filters['age_range'] = age_range

            # Gender filter
            gender_cols = [col for col in self.feature_categories['demographics']
                          if 'gender' in col.lower()]
            if gender_cols:
                gender_options = ["All", "Male", "Female"]
                gender = st.selectbox("Gender", gender_options, key="gender_filter")
                if gender != "All":
                    self.filters['gender'] = gender

        with col2:
            # Region filter
            region_cols = self.feature_categories['regions']
            if region_cols:
                # Handle different prefixes for region names
                region_names = []
                for col in region_cols:
                    if 'cat__region_en_x_' in col:
                        region_names.append(col.replace('cat__region_en_x_', ''))
                    elif 'region_en_x_' in col:
                        region_names.append(col.replace('region_en_x_', ''))
                    else:
                        region_names.append(col.split('_')[-1])

                selected_regions = st.multiselect(
                    "Regions",
                    options=region_names,
                    default=None,
                    key="region_filter"
                )
                if selected_regions:
                    self.filters['regions'] = selected_regions

            # Nationality filter (if needed)
            nationality_cols = [col for col in self.feature_categories['demographics']
                               if 'nationality' in col.lower()]
            if nationality_cols and len(nationality_cols) < 50:  # Only show if reasonable number
                nationality_names = [col.replace('nationality_en_x_', '') for col in nationality_cols]
                selected_nationalities = st.multiselect(
                    "Nationalities",
                    options=nationality_names[:20],  # Limit to top 20
                    default=None,
                    key="nationality_filter"
                )
                if selected_nationalities:
                    self.filters['nationalities'] = selected_nationalities

    def _render_condition_filters(self):
        """Render clinical condition filters"""

        st.markdown("#### Select Clinical Conditions")

        condition_cols = self.feature_categories['conditions']

        # Group conditions
        conditions = {
            'Hypertension': [col for col in condition_cols if 'hypertension' in col.lower()],
            'Diabetes': [col for col in condition_cols if 'diabetes' in col.lower()],
            'Obesity': [col for col in condition_cols if 'obesity' in col.lower()],
            'Dyslipidemia': [col for col in condition_cols if 'dyslipidemia' in col.lower()]
        }

        col1, col2, col3, col4 = st.columns(4)

        cols = [col1, col2, col3, col4]
        for i, (condition_name, condition_cols_list) in enumerate(conditions.items()):
            with cols[i]:
                if condition_cols_list:
                    # Find the True column for this condition
                    true_col = None
                    for col in condition_cols_list:
                        if 'True' in col:
                            true_col = col
                            break

                    if true_col:
                        filter_option = st.selectbox(
                            condition_name,
                            ["Any", "Yes", "No"],
                            key=f"{condition_name}_filter"
                        )
                        if filter_option != "Any":
                            self.filters[f'has_{condition_name.lower()}'] = (filter_option == "Yes")

        # Additional conditions
        st.markdown("#### Additional Filters")
        col1, col2 = st.columns(2)

        with col1:
            if 'is_dead' in self.X_test.columns:
                mortality_filter = st.selectbox(
                    "Mortality Status",
                    ["Any", "Alive", "Deceased"],
                    key="mortality_filter"
                )
                if mortality_filter != "Any":
                    self.filters['is_dead'] = (mortality_filter == "Deceased")

        with col2:
            if 'visit_pregnancy_clinic' in self.X_test.columns:
                pregnancy_filter = st.selectbox(
                    "Pregnancy Clinic Visit",
                    ["Any", "Yes", "No"],
                    key="pregnancy_filter"
                )
                if pregnancy_filter != "Any":
                    self.filters['visit_pregnancy_clinic'] = (pregnancy_filter == "Yes")

    def _render_biomarker_filters(self):
        """Render biomarker range filters"""

        st.markdown("#### Filter by Biomarker Ranges")

        biomarker_cols = self.feature_categories['biomarkers']

        if not biomarker_cols:
            st.info("No biomarker data available for filtering")
            return

        # Group biomarkers by category
        biomarker_groups = {
            'Metabolic': ['glucose', 'hba1c', 'cholesterol', 'triglycerides', 'hdl', 'ldl'],
            'Renal': ['creatinine', 'uric_acid', 'egfr'],
            'Liver': ['albumin', 'bilirubin', 'alt', 'ast', 'alkaline_phosphatase'],
            'Hematology': ['hemoglobin', 'wbc', 'rdw', 'lymphocyte'],
            'Electrolytes': ['sodium', 'potassium', 'calcium', 'magnesium']
        }

        selected_group = st.selectbox("Select Biomarker Category", list(biomarker_groups.keys()))

        # Display filters for selected group
        relevant_cols = []
        for keyword in biomarker_groups[selected_group]:
            relevant_cols.extend([col for col in biomarker_cols if keyword in col.lower()])

        if relevant_cols:
            # Show up to 4 biomarkers in columns
            cols = st.columns(min(4, len(relevant_cols)))
            for i, col in enumerate(relevant_cols[:8]):  # Limit to 8 biomarkers
                with cols[i % 4]:
                    if col in self.X_test.columns:
                        values = self.X_test[col].dropna()
                        if len(values) > 0:
                            # Use percentiles for range
                            min_val = float(values.quantile(0.01))
                            max_val = float(values.quantile(0.99))
                            mean_val = float(values.mean())

                            biomarker_name = col.replace('_x', '').replace('_', ' ').title()
                            range_val = st.slider(
                                biomarker_name,
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"biomarker_{col}",
                                help=f"Mean: {mean_val:.2f}"
                            )

                            if range_val != (min_val, max_val):
                                self.filters[f'biomarker_{col}'] = range_val
        else:
            st.info(f"No {selected_group} biomarkers available")

    def _render_utilization_filters(self):
        """Render healthcare utilization filters"""

        st.markdown("#### Healthcare Utilization Patterns")

        utilization_cols = self.feature_categories['utilization']

        if not utilization_cols:
            st.info("No healthcare utilization data available")
            return

        col1, col2 = st.columns(2)

        with col1:
            # Outpatient visits
            outpatient_col = next((col for col in utilization_cols if 'outpatient' in col.lower()), None)
            if outpatient_col and outpatient_col in self.X_test.columns:
                values = self.X_test[outpatient_col].dropna()
                if len(values) > 0:
                    outpatient_range = st.slider(
                        "Outpatient Visits",
                        min_value=int(values.min()),
                        max_value=min(int(values.quantile(0.99)), 100),
                        value=(int(values.min()), min(int(values.quantile(0.99)), 100)),
                        key="outpatient_filter"
                    )
                    if outpatient_range != (int(values.min()), min(int(values.quantile(0.99)), 100)):
                        self.filters['outpatient_visits'] = outpatient_range

            # Inpatient visits
            inpatient_col = next((col for col in utilization_cols if 'inpatient' in col.lower()), None)
            if inpatient_col and inpatient_col in self.X_test.columns:
                values = self.X_test[inpatient_col].dropna()
                if len(values) > 0:
                    inpatient_range = st.slider(
                        "Inpatient Admissions",
                        min_value=int(values.min()),
                        max_value=min(int(values.quantile(0.99)), 20),
                        value=(int(values.min()), min(int(values.quantile(0.99)), 20)),
                        key="inpatient_filter"
                    )
                    if inpatient_range != (int(values.min()), min(int(values.quantile(0.99)), 20)):
                        self.filters['inpatient_visits'] = inpatient_range

        with col2:
            # Emergency visits
            emergency_col = next((col for col in utilization_cols if 'emergency' in col.lower()), None)
            if emergency_col and emergency_col in self.X_test.columns:
                values = self.X_test[emergency_col].dropna()
                if len(values) > 0:
                    emergency_range = st.slider(
                        "Emergency Visits",
                        min_value=int(values.min()),
                        max_value=min(int(values.quantile(0.99)), 50),
                        value=(int(values.min()), min(int(values.quantile(0.99)), 50)),
                        key="emergency_filter"
                    )
                    if emergency_range != (int(values.min()), min(int(values.quantile(0.99)), 50)):
                        self.filters['emergency_visits'] = emergency_range

            # Total medications
            medication_col = next((col for col in utilization_cols
                                 if 'unique_medication' in col.lower() or 'medication_dose' in col.lower()), None)
            if medication_col and medication_col in self.X_test.columns:
                values = self.X_test[medication_col].dropna()
                if len(values) > 0:
                    medication_range = st.slider(
                        "Number of Medications",
                        min_value=int(values.min()),
                        max_value=min(int(values.quantile(0.99)), 30),
                        value=(int(values.min()), min(int(values.quantile(0.99)), 30)),
                        key="medication_count_filter"
                    )
                    if medication_range != (int(values.min()), min(int(values.quantile(0.99)), 30)):
                        self.filters['medication_count'] = medication_range

    def _render_medication_filters(self):
        """Render medication filters"""

        st.markdown("#### Filter by Medications")

        medication_cols = self.feature_categories['medications']

        if not medication_cols:
            st.info("No medication data available")
            return

        # Group medications by type
        med_groups = {
            'Diabetes': ['metformin', 'gliptin', 'gliflozin', 'glp1', 'insulin', 'dapagliflozin', 'empagliflozin'],
            'Cardiovascular': ['statin', 'blocker', 'ace_inhibitor', 'arb', 'aspirin', 'clopidogrel', 'warfarin'],
            'Others': []
        }

        # Categorize medication columns
        categorized_meds = {group: [] for group in med_groups}
        for col in medication_cols:
            col_lower = col.lower()
            categorized = False
            for group, keywords in med_groups.items():
                if group != 'Others':
                    for keyword in keywords:
                        if keyword in col_lower:
                            categorized_meds[group].append(col)
                            categorized = True
                            break
                    if categorized:
                        break
            if not categorized:
                categorized_meds['Others'].append(col)

        # Display medication filters by group
        selected_med_group = st.selectbox("Select Medication Category",
                                         [g for g in categorized_meds.keys() if categorized_meds[g]])

        if selected_med_group and categorized_meds[selected_med_group]:
            meds = categorized_meds[selected_med_group][:20]  # Limit to 20 medications
            med_names = [col.replace('medication_', '').replace('_x', '').replace('_', ' ').title()
                        for col in meds]

            selected_meds = st.multiselect(
                f"Select {selected_med_group} Medications",
                options=list(zip(meds, med_names)),
                format_func=lambda x: x[1],
                key=f"medication_{selected_med_group}"
            )

            if selected_meds:
                self.filters['medications'] = [med[0] for med in selected_meds]

    def _render_advanced_filters(self):
        """Render advanced filtering options"""

        st.markdown("#### Advanced Filtering Options")

        col1, col2 = st.columns(2)

        with col1:
            # Biological age acceleration filter
            if 'age_acc_test' in self.shap_data:
                from utils.data_loader import DataLoader
                loader = DataLoader()
                scaler_stats = loader.load_age_scaler_stats()

                # Denormalize age acceleration
                age_acc = self.shap_data['age_acc_test'] * scaler_stats['iqr']
                age_acc_range = st.slider(
                    "Biological Age Acceleration (years)",
                    min_value=float(age_acc.min()),
                    max_value=float(age_acc.max()),
                    value=(float(age_acc.min()), float(age_acc.max())),
                    key="age_acc_filter",
                    help="Difference between biological and chronological age"
                )
                if age_acc_range != (float(age_acc.min()), float(age_acc.max())):
                    self.filters['age_acceleration'] = age_acc_range

        with col2:
            # Risk score filter (if available)
            if 'pred_shap' in self.shap_data:
                risk_scores = self.shap_data.get('pred_shap', np.zeros(len(self.X_test)))
                risk_range = st.slider(
                    "Risk Score Range (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),
                    key="risk_filter",
                    help="Predicted mortality/hospitalization risk"
                )
                if risk_range != (0, 100):
                    self.filters['risk_score'] = (risk_range[0]/100, risk_range[1]/100)

        # Custom query builder
        st.markdown("#### Custom Query")
        custom_query = st.text_area(
            "Enter custom filter expression (pandas query syntax)",
            placeholder="e.g., (glucose_x > 7) & (has_diabetes_x_True == 1)",
            key="custom_query"
        )
        if custom_query:
            self.filters['custom_query'] = custom_query

    def _apply_filters(self) -> np.ndarray:
        """Apply all filters and return selected patient indices"""

        # Start with all patients
        mask = np.ones(len(self.X_test), dtype=bool)

        # Apply age filter
        if 'age_range' in self.filters and 'age_x' in self.X_test.columns:
            from utils.data_loader import DataLoader
            loader = DataLoader()
            scaler_stats = loader.load_age_scaler_stats()
            ages = loader.denormalize_age(self.X_test['age_x'].values, scaler_stats)
            age_range = self.filters['age_range']
            mask &= (ages >= age_range[0]) & (ages <= age_range[1])

        # Apply gender filter
        if 'gender' in self.filters:
            gender = self.filters['gender']
            if gender == "Male" and 'gender_x_male' in self.X_test.columns:
                mask &= (self.X_test['gender_x_male'] == 1)
            elif gender == "Female" and 'gender_x_female' in self.X_test.columns:
                mask &= (self.X_test['gender_x_female'] == 1)

        # Apply region filter
        if 'regions' in self.filters:
            region_mask = np.zeros(len(self.X_test), dtype=bool)
            for region in self.filters['regions']:
                # Try different possible column names
                possible_cols = [
                    f'cat__region_en_x_{region}',
                    f'region_en_x_{region}',
                    f'region_{region}'
                ]
                for region_col in possible_cols:
                    if region_col in self.X_test.columns:
                        region_mask |= (self.X_test[region_col] == 1)
                        break
            mask &= region_mask

        # Apply condition filters
        for condition in ['hypertension', 'diabetes', 'obesity', 'dyslipidemia']:
            filter_key = f'has_{condition}'
            if filter_key in self.filters:
                col_name = f'has_{condition}_x_True'
                if col_name in self.X_test.columns:
                    if self.filters[filter_key]:
                        mask &= (self.X_test[col_name] == 1)
                    else:
                        mask &= (self.X_test[col_name] == 0)

        # Apply biomarker filters
        for key, value in self.filters.items():
            if key.startswith('biomarker_'):
                col_name = key.replace('biomarker_', '')
                if col_name in self.X_test.columns:
                    mask &= (self.X_test[col_name] >= value[0]) & (self.X_test[col_name] <= value[1])

        # Apply utilization filters
        utilization_mappings = {
            'outpatient_visits': 'total_outpatient_visits',
            'inpatient_visits': 'total_inpatient_visits',
            'emergency_visits': 'total_emergency_visits',
            'medication_count': 'unique_medications'
        }

        for filter_key, col_pattern in utilization_mappings.items():
            if filter_key in self.filters:
                # Find matching column
                matching_col = next((col for col in self.X_test.columns if col_pattern in col.lower()), None)
                if matching_col:
                    range_val = self.filters[filter_key]
                    mask &= (self.X_test[matching_col] >= range_val[0]) & (self.X_test[matching_col] <= range_val[1])

        # Apply medication filters
        if 'medications' in self.filters:
            med_mask = np.zeros(len(self.X_test), dtype=bool)
            for med_col in self.filters['medications']:
                if med_col in self.X_test.columns:
                    med_mask |= (self.X_test[med_col] == 1)
            mask &= med_mask

        # Apply advanced filters
        if 'age_acceleration' in self.filters and 'age_acc_test' in self.shap_data:
            from utils.data_loader import DataLoader
            loader = DataLoader()
            scaler_stats = loader.load_age_scaler_stats()
            age_acc = self.shap_data['age_acc_test'] * scaler_stats['iqr']
            age_acc_range = self.filters['age_acceleration']
            mask &= (age_acc >= age_acc_range[0]) & (age_acc <= age_acc_range[1])

        if 'risk_score' in self.filters and 'pred_shap' in self.shap_data:
            risk_scores = self.shap_data.get('pred_shap', np.zeros(len(self.X_test)))
            risk_range = self.filters['risk_score']
            mask &= (risk_scores >= risk_range[0]) & (risk_scores <= risk_range[1])

        # Apply custom query
        if 'custom_query' in self.filters:
            try:
                custom_mask = self.X_test.eval(self.filters['custom_query'])
                mask &= custom_mask
            except Exception as e:
                st.error(f"Error in custom query: {e}")

        # Return indices where mask is True
        return np.where(mask)[0]

    def _display_cohort_statistics(self):
        """Display statistics for the selected cohort"""

        if self.selected_indices is None or len(self.selected_indices) == 0:
            st.warning("No patients selected")
            return

        st.markdown("---")
        st.markdown("### Cohort Statistics")

        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Patients", f"{len(self.selected_indices):,}")

        with col2:
            total_patients = len(self.X_test)
            percentage = (len(self.selected_indices) / total_patients) * 100
            st.metric("% of Total", f"{percentage:.1f}%")

        with col3:
            # Gender distribution
            if 'gender_x_male' in self.X_test.columns:
                male_count = self.X_test.iloc[self.selected_indices]['gender_x_male'].sum()
                female_count = len(self.selected_indices) - male_count
                st.metric("M/F Ratio", f"{male_count:,}/{female_count:,}")

        with col4:
            # Average age
            if 'age_x' in self.X_test.columns:
                from utils.data_loader import DataLoader
                loader = DataLoader()
                scaler_stats = loader.load_age_scaler_stats()
                ages = loader.denormalize_age(
                    self.X_test.iloc[self.selected_indices]['age_x'].values,
                    scaler_stats
                )
                st.metric("Avg Age", f"{ages.mean():.1f} years")

        # Condition prevalence in cohort
        st.markdown("#### Condition Prevalence in Cohort")
        conditions = ['hypertension', 'diabetes', 'obesity', 'dyslipidemia']
        prevalence_data = []

        for condition in conditions:
            col_name = f'has_{condition}_x_True'
            if col_name in self.X_test.columns:
                prevalence = self.X_test.iloc[self.selected_indices][col_name].mean() * 100
                prevalence_data.append({
                    'Condition': condition.capitalize(),
                    'Prevalence (%)': prevalence
                })

        if prevalence_data:
            df_prevalence = pd.DataFrame(prevalence_data)
            fig = px.bar(df_prevalence, x='Condition', y='Prevalence (%)',
                        color='Prevalence (%)', color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    def _save_cohort(self):
        """Save the current cohort definition"""
        if self.selected_indices is not None:
            # Create cohort summary
            cohort_data = {
                'indices': self.selected_indices.tolist(),
                'filters': self.filters,
                'size': len(self.selected_indices),
                'timestamp': pd.Timestamp.now().isoformat()
            }

            # Allow user to name the cohort
            cohort_name = st.text_input("Enter cohort name:", key="cohort_name_input")
            if cohort_name and st.button("Confirm Save"):
                # In a real app, you would save this to a database or file
                st.success(f"Cohort '{cohort_name}' saved successfully!")
                st.json(cohort_data)

    def get_selected_data(self) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """Get data for selected patients"""

        if self.selected_indices is None:
            return self.X_test, np.arange(len(self.X_test)), {}

        X_selected = self.X_test.iloc[self.selected_indices]
        y_selected = self.y_test[self.selected_indices] if len(self.y_test) > 0 else None

        selected_shap_data = {}
        for key in self.shap_data:
            if isinstance(self.shap_data[key], np.ndarray) and len(self.shap_data[key]) == len(self.X_test):
                selected_shap_data[key] = self.shap_data[key][self.selected_indices]

        return X_selected, self.selected_indices, selected_shap_data