"""
Display Helper Functions for User-Friendly Metrics
Provides consistent formatting for age, risk, and health metrics
to improve user understanding and avoid confusion with negative values
"""

import streamlit as st
from typing import Tuple, Optional, Dict, Any
import numpy as np


def format_age_acceleration(value: float, show_interpretation: bool = True) -> Dict[str, Any]:
    """
    Format age acceleration value for user-friendly display.

    Args:
        value: Age acceleration in years (positive = older, negative = younger)
        show_interpretation: Whether to include interpretation text

    Returns:
        Dict with 'text', 'color', 'badge', and 'interpretation' keys
    """
    abs_value = abs(value)

    if abs_value < 0.5:
        return {
            'text': 'Age-appropriate',
            'color': 'gray',
            'badge': '✓ Normal',
            'interpretation': 'Your biological age matches your chronological age',
            'short': 'On track'
        }
    elif value > 0:
        # Aging faster than expected
        return {
            'text': f'{abs_value:.1f} years older than expected',
            'color': '#e74c3c',  # Red
            'badge': '⚠️ Aging Faster',
            'interpretation': f'Your body shows signs of accelerated aging by {abs_value:.1f} years',
            'short': f'{abs_value:.1f} yrs older'
        }
    else:
        # Aging slower than expected (younger)
        return {
            'text': f'{abs_value:.1f} years younger than expected',
            'color': '#27ae60',  # Green
            'badge': '✓ Aging Slower',
            'interpretation': f'Your body is aging more slowly, appearing {abs_value:.1f} years younger',
            'short': f'{abs_value:.1f} yrs younger'
        }


def format_risk_level(score: float, risk_type: str = "Mortality") -> Dict[str, Any]:
    """
    Format risk score for clear user understanding.

    Args:
        score: Risk score between 0 and 1
        risk_type: Type of risk (Mortality, Hospitalization, etc.)

    Returns:
        Dict with formatted risk information
    """
    percentage = score * 100

    if score < 0.3:
        return {
            'level': 'Low Risk',
            'color': '#27ae60',  # Green
            'icon': '✓',
            'percentage': f'{percentage:.0f}%',
            'interpretation': f'Low {risk_type.lower()} risk - Keep up the good work!',
            'badge_color': 'success'
        }
    elif score < 0.6:
        return {
            'level': 'Moderate Risk',
            'color': '#f39c12',  # Orange
            'icon': '!',
            'percentage': f'{percentage:.0f}%',
            'interpretation': f'Moderate {risk_type.lower()} risk - Consider preventive measures',
            'badge_color': 'warning'
        }
    else:
        return {
            'level': 'High Risk',
            'color': '#e74c3c',  # Red
            'icon': '⚠️',
            'percentage': f'{percentage:.0f}%',
            'interpretation': f'High {risk_type.lower()} risk - Medical consultation recommended',
            'badge_color': 'error'
        }


def format_feature_impact(feature_name: str, shap_value: float,
                         feature_value: Optional[float] = None) -> Dict[str, Any]:
    """
    Format feature importance for clear impact understanding.

    Args:
        feature_name: Name of the feature
        shap_value: SHAP value (impact on biological age)
        feature_value: Optional actual feature value

    Returns:
        Dict with formatted feature impact information
    """
    abs_impact = abs(shap_value)

    # Clean up feature name
    clean_name = (feature_name.replace('_x', '')
                             .replace('_', ' ')
                             .replace('has ', '')
                             .replace(' True', '')
                             .replace(' False', '')
                             .title())

    if abs_impact < 0.1:
        return {
            'name': clean_name,
            'impact': 'Minimal impact',
            'icon': '○',
            'color': 'gray',
            'description': 'Negligible effect on biological age'
        }
    elif shap_value > 0:
        # Feature increases biological age (harmful)
        return {
            'name': clean_name,
            'impact': f'Accelerates aging by {abs_impact:.1f} years',
            'icon': '⚠️',
            'color': '#e74c3c',
            'description': f'{clean_name} is contributing to faster aging',
            'short': f'+{abs_impact:.1f} yrs'
        }
    else:
        # Feature decreases biological age (protective)
        return {
            'name': clean_name,
            'impact': f'Slows aging by {abs_impact:.1f} years',
            'icon': '✓',
            'color': '#27ae60',
            'description': f'{clean_name} is protecting against aging',
            'short': f'-{abs_impact:.1f} yrs'
        }


def format_biomarker_status(biomarker_name: str, value: float,
                           reference_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Format biomarker values with status indicators.

    Args:
        biomarker_name: Name of the biomarker
        value: Current biomarker value
        reference_range: Optional tuple of (min, max) for normal range

    Returns:
        Dict with formatted biomarker information
    """
    # Default reference ranges for common biomarkers (normalized values)
    default_ranges = {
        'glucose': (-1.0, 1.0),
        'creatinine': (-1.0, 1.0),
        'albumin': (-0.5, 1.5),
        'cholesterol': (-1.0, 1.0),
        'hemoglobin': (-1.0, 1.0)
    }

    # Clean biomarker name
    clean_name = biomarker_name.lower().replace('_x', '').replace('_', ' ')

    # Get reference range
    if reference_range is None:
        for key in default_ranges:
            if key in clean_name:
                reference_range = default_ranges[key]
                break

    if reference_range:
        min_val, max_val = reference_range
        if value < min_val:
            status = 'Below Normal'
            color = '#3498db'  # Blue
            icon = '↓'
        elif value > max_val:
            status = 'Above Normal'
            color = '#e74c3c'  # Red
            icon = '↑'
        else:
            status = 'Normal'
            color = '#27ae60'  # Green
            icon = '✓'
    else:
        status = 'Value recorded'
        color = 'gray'
        icon = '•'

    return {
        'name': clean_name.title(),
        'value': value,
        'status': status,
        'color': color,
        'icon': icon
    }


def format_improvement_metric(current_value: float, improved_value: float,
                             metric_type: str = "age") -> Dict[str, Any]:
    """
    Format improvement metrics for interventions.

    Args:
        current_value: Current metric value
        improved_value: Projected improved value
        metric_type: Type of metric (age, risk, etc.)

    Returns:
        Dict with formatted improvement information
    """
    improvement = current_value - improved_value
    abs_improvement = abs(improvement)

    if metric_type == "age":
        if improvement > 0:
            return {
                'text': f'Reduces biological age by {abs_improvement:.1f} years',
                'color': '#27ae60',
                'icon': '✓',
                'percentage': None
            }
        else:
            return {
                'text': 'No significant age reduction',
                'color': 'gray',
                'icon': '○',
                'percentage': None
            }
    elif metric_type == "risk":
        pct_reduction = (improvement / current_value) * 100 if current_value > 0 else 0
        if pct_reduction > 0:
            return {
                'text': f'Reduces risk by {abs(pct_reduction):.0f}%',
                'color': '#27ae60',
                'icon': '✓',
                'percentage': f'{abs(pct_reduction):.0f}%'
            }
        else:
            return {
                'text': 'No risk reduction',
                'color': 'gray',
                'icon': '○',
                'percentage': '0%'
            }

    return {
        'text': 'No change',
        'color': 'gray',
        'icon': '○',
        'percentage': None
    }


def render_age_metric(chrono_age: float, bio_age: float,
                     col: Optional[st.delta_generator.DeltaGenerator] = None):
    """
    Render age metrics in Streamlit with proper formatting.

    Args:
        chrono_age: Chronological age
        bio_age: Biological age
        col: Optional Streamlit column to render in
    """
    age_acc = bio_age - chrono_age
    age_info = format_age_acceleration(age_acc)

    target = col if col else st

    # Display the metrics without confusing deltas
    target.metric(
        label="Biological Age Status",
        value=age_info['short'],
        help=age_info['interpretation']
    )

    # Add colored badge
    target.markdown(
        f"<span style='color:{age_info['color']};font-weight:bold'>"
        f"{age_info['badge']}</span>",
        unsafe_allow_html=True
    )


def render_risk_metric(risk_score: float, risk_type: str = "Mortality",
                      col: Optional[st.delta_generator.DeltaGenerator] = None):
    """
    Render risk metrics in Streamlit with proper formatting.

    Args:
        risk_score: Risk score (0-1)
        risk_type: Type of risk
        col: Optional Streamlit column to render in
    """
    risk_info = format_risk_level(risk_score, risk_type)

    target = col if col else st

    # Display the metric without confusing delta
    target.metric(
        label=f"{risk_type} Risk",
        value=risk_info['percentage'],
        help=risk_info['interpretation']
    )

    # Add colored badge
    if risk_info['badge_color'] == 'success':
        target.success(f"{risk_info['icon']} {risk_info['level']}")
    elif risk_info['badge_color'] == 'warning':
        target.warning(f"{risk_info['icon']} {risk_info['level']}")
    else:
        target.error(f"{risk_info['icon']} {risk_info['level']}")


def format_correlation_interpretation(correlation: float,
                                     biomarker: str) -> Dict[str, Any]:
    """
    Format correlation values with user-friendly interpretation.

    Args:
        correlation: Correlation coefficient (-1 to 1)
        biomarker: Name of the biomarker

    Returns:
        Dict with interpretation of correlation
    """
    abs_corr = abs(correlation)

    if abs_corr < 0.1:
        strength = "No relationship"
        color = "gray"
    elif abs_corr < 0.3:
        strength = "Weak relationship"
        color = "#95a5a6"
    elif abs_corr < 0.5:
        strength = "Moderate relationship"
        color = "#f39c12"
    else:
        strength = "Strong relationship"
        color = "#e74c3c" if correlation > 0 else "#27ae60"

    if correlation > 0.1:
        direction = "Higher values associated with faster aging"
        impact = "negative"
    elif correlation < -0.1:
        direction = "Higher values associated with slower aging"
        impact = "positive"
    else:
        direction = "No significant association with aging"
        impact = "neutral"

    return {
        'strength': strength,
        'direction': direction,
        'impact': impact,
        'color': color,
        'value': correlation,
        'interpretation': f"{biomarker}: {strength.lower()} - {direction.lower()}"
    }