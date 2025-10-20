from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from io import BytesIO
import secrets
import math
from shapely.geometry import Polygon

# Import the DataExtractor from analyzer.py
sys.path.insert(0, os.path.dirname(__file__))
from analyzer import DataExtractor
from auth import (
    register_user, login_user, logout_user, 
    validate_session, login_required, get_user_by_id
)
# Import wind analysis function from wind.py
from wind import wind_suitability_score

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Generate secure secret key
CORS(app, supports_credentials=True)

# Load ML models
try:
    solar_rf, solar_features = joblib.load("solar_model.pkl")
    wind_rf, wind_features = joblib.load("wind_model.pkl")
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load models - {e}")
    solar_rf, wind_rf = None, None
    solar_features, wind_features = [], []

# Initialize data extractor
extractor = DataExtractor()


def analyze_solar_site_api(site_data):
    """Analyze solar site using loaded model"""
    if solar_rf is None:
        return "Unknown", 0, {"error": "Model not loaded"}
    
    df_site = pd.DataFrame([site_data])
    
    # Add missing columns with mean values
    for col in solar_features:
        if col not in df_site.columns:
            df_site[col] = 0
    
    df_site = df_site[solar_features]
    
    # Predict
    prediction = solar_rf.predict(df_site)[0]
    label = "Yes" if prediction == 1 else "No"
    
    # Calculate score
    thresholds = {
        "GHI (kWh/m²/day)": 4.0,
        "DNI (kWh/m²/day)": 4.0,
        "DHI (% of GHI)": (15, 25),
        "Snowfall (mm/year)": 100,
        "Ambient temperature": (15, 35),
        "Relative humidity": 70,
        "Precipitation": 1500,
        "Sunshine duration": 6,
        "YearlyCloud cover": 50
    }
    
    score = 0
    suggestions = {}
    
    for param, threshold in thresholds.items():
        if param not in df_site.columns:
            continue
        value = float(df_site[param].iloc[0])
        
        if isinstance(threshold, tuple):
            low, high = threshold
            if low <= value <= high:
                score += 1
            else:
                if value < low:
                    suggestions[param] = f"Increase by {round(low - value, 2)}"
                else:
                    suggestions[param] = f"Reduce by {round(value - high, 2)}"
        else:
            if param in ["Relative humidity", "Precipitation", "Snowfall (mm/year)", "YearlyCloud cover"]:
                if value < threshold:
                    score += 1
                else:
                    suggestions[param] = f"Reduce by {round(value - threshold, 2)}"
            else:
                if value >= threshold:
                    score += 1
                else:
                    suggestions[param] = f"Increase by {round(threshold - value, 2)}"
    
    total_params = len(thresholds)
    suitability_percent = round((score / total_params) * 100, 2)
    
    return label, suitability_percent, suggestions


def analyze_wind_site_api(site_data):
    """
    Analyze wind site using the actual wind.py model and logic
    Returns:
        label: "Yes" / "No"
        suitability_percent: 0-100
        suggestions: dict of parameter adjustments
    """
    try:
        # Call the actual wind_suitability_score from wind.py
        label, suitability_percent, suggestions = wind_suitability_score(site_data)
        
        print(f"🌬️ Wind Analysis Result: {label}, Score: {suitability_percent}%")
        
        return label, suitability_percent, suggestions
        
    except Exception as e:
        print(f"❌ Error in wind analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Unknown", 0, {"error": str(e)}


def calculate_polygon_area(coordinates):
    """Calculate area of polygon in square meters using Shapely"""
    try:
        # Convert to (lon, lat) format for Shapely
        coords = [(coord['lng'], coord['lat']) for coord in coordinates]
        polygon = Polygon(coords)
        
        # Get area in square degrees
        area_degrees = polygon.area
        
        # Convert to square meters (approximate)
        # At equator: 1 degree ≈ 111km
        # This is approximate, for precise calculation use pyproj
        lat_avg = sum([c['lat'] for c in coordinates]) / len(coordinates)
        meters_per_degree_lat = 111000
        meters_per_degree_lon = 111000 * math.cos(math.radians(lat_avg))
        
        area_m2 = area_degrees * meters_per_degree_lat * meters_per_degree_lon
        return area_m2
    except Exception as e:
        print(f"Error calculating area: {e}")
        return 0


def calculate_solar_impact_and_cost(area_m2, suitability_score):
    """
    Calculate solar panel installation capacity, cost, and environmental impact
    
    Args:
        area_m2: Area in square meters
        suitability_score: 0-100 score
    
    Returns:
        dict with installation details, costs, and environmental impact
    """
    # Constants
    PANEL_AREA = 2.0  # m² per panel (typical 400W panel)
    PANEL_CAPACITY = 0.4  # kW per panel
    USABLE_AREA_RATIO = 0.6  # 60% of area usable (spacing, access roads, etc.)
    INSTALLATION_COST_PER_KW = 40000  # Rs40,000 per kW (India average)
    MAINTENANCE_COST_YEARLY = 0.02  # 2% of installation cost per year
    CAPACITY_FACTOR = 0.19  # 19% average for India
    COAL_CO2_PER_KWH = 0.85  # kg CO2 per kWh from coal
    COAL_POLLUTION_PER_KWH = 0.02  # kg pollutants per kWh
    
    # Adjust capacity factor based on suitability
    adjusted_capacity_factor = CAPACITY_FACTOR * (suitability_score / 100)
    
    # Calculate installation capacity
    usable_area = area_m2 * USABLE_AREA_RATIO
    num_panels = int(usable_area / PANEL_AREA)
    total_capacity_kw = num_panels * PANEL_CAPACITY
    total_capacity_mw = total_capacity_kw / 1000
    
    # Calculate energy generation
    hours_per_year = 365 * 24
    annual_generation_kwh = total_capacity_kw * adjusted_capacity_factor * hours_per_year
    annual_generation_mwh = annual_generation_kwh / 1000
    
    # Calculate costs
    installation_cost = total_capacity_kw * INSTALLATION_COST_PER_KW
    annual_maintenance = installation_cost * MAINTENANCE_COST_YEARLY
    
    # Calculate environmental impact (compared to coal)
    co2_reduction_yearly = annual_generation_kwh * COAL_CO2_PER_KWH  # kg
    co2_reduction_yearly_tons = co2_reduction_yearly / 1000
    pollution_reduction_yearly = annual_generation_kwh * COAL_POLLUTION_PER_KWH  # kg
    
    # Equivalent trees (1 tree absorbs ~20kg CO2/year)
    equivalent_trees = int(co2_reduction_yearly / 20)
    
    # Coal dependency reduction (assuming 0.5 kg coal per kWh)
    coal_saved_yearly_kg = annual_generation_kwh * 0.5
    coal_saved_yearly_tons = coal_saved_yearly_kg / 1000
    
    return {
        "installation": {
            "area_m2": round(area_m2, 2),
            "usable_area_m2": round(usable_area, 2),
            "num_panels": num_panels,
            "total_capacity_kw": round(total_capacity_kw, 2),
            "total_capacity_mw": round(total_capacity_mw, 2),
            "capacity_factor": round(adjusted_capacity_factor * 100, 2)
        },
        "generation": {
            "annual_kwh": round(annual_generation_kwh, 2),
            "annual_mwh": round(annual_generation_mwh, 2),
            "daily_avg_kwh": round(annual_generation_kwh / 365, 2)
        },
        "cost": {
            "installation_inr": round(installation_cost, 2),
            "installation_crore": round(installation_cost / 10000000, 2),
            "annual_maintenance_inr": round(annual_maintenance, 2),
            "annual_maintenance_lakh": round(annual_maintenance / 100000, 2)
        },
        "environmental_impact": {
            "co2_reduction_yearly_tons": round(co2_reduction_yearly_tons, 2),
            "co2_reduction_25years_tons": round(co2_reduction_yearly_tons * 25, 2),
            "equivalent_trees": equivalent_trees,
            "pollution_reduction_yearly_kg": round(pollution_reduction_yearly, 2),
            "coal_saved_yearly_tons": round(coal_saved_yearly_tons, 2)
        }
    }


def calculate_wind_impact_and_cost(area_m2, suitability_score, avg_wind_speed):
    """
    Calculate wind turbine installation capacity, cost, and environmental impact
    
    Args:
        area_m2: Area in square meters
        suitability_score: 0-100 score
        avg_wind_speed: Average wind speed in m/s
    
    Returns:
        dict with installation details, costs, and environmental impact
    """
    # Constants
    TURBINE_SPACING = 250000  # m² per turbine (500m x 500m spacing)
    TURBINE_CAPACITY = 2500  # kW per turbine (2.5 MW)
    INSTALLATION_COST_PER_KW = 60000  # Rs60,000 per kW (India average)
    MAINTENANCE_COST_YEARLY = 0.03  # 3% of installation cost per year
    BASE_CAPACITY_FACTOR = 0.25  # 25% average for India
    COAL_CO2_PER_KWH = 0.85  # kg CO2 per kWh from coal
    COAL_POLLUTION_PER_KWH = 0.02  # kg pollutants per kWh
    
    # Adjust capacity factor based on wind speed and suitability
    wind_factor = min(avg_wind_speed / 6.5, 1.5)  # Optimal at 6.5 m/s
    adjusted_capacity_factor = BASE_CAPACITY_FACTOR * wind_factor * (suitability_score / 100)
    
    # Calculate installation capacity
    num_turbines = max(1, int(area_m2 / TURBINE_SPACING))
    total_capacity_kw = num_turbines * TURBINE_CAPACITY
    total_capacity_mw = total_capacity_kw / 1000
    
    # Calculate energy generation
    hours_per_year = 365 * 24
    annual_generation_kwh = total_capacity_kw * adjusted_capacity_factor * hours_per_year
    annual_generation_mwh = annual_generation_kwh / 1000
    
    # Calculate costs
    installation_cost = total_capacity_kw * INSTALLATION_COST_PER_KW
    annual_maintenance = installation_cost * MAINTENANCE_COST_YEARLY
    
    # Calculate environmental impact
    co2_reduction_yearly = annual_generation_kwh * COAL_CO2_PER_KWH
    co2_reduction_yearly_tons = co2_reduction_yearly / 1000
    pollution_reduction_yearly = annual_generation_kwh * COAL_POLLUTION_PER_KWH
    
    equivalent_trees = int(co2_reduction_yearly / 20)
    
    coal_saved_yearly_kg = annual_generation_kwh * 0.5
    coal_saved_yearly_tons = coal_saved_yearly_kg / 1000
    
    return {
        "installation": {
            "area_m2": round(area_m2, 2),
            "num_turbines": num_turbines,
            "turbine_capacity_kw": TURBINE_CAPACITY,
            "turbine_capacity_mw": TURBINE_CAPACITY / 1000,
            "total_capacity_kw": round(total_capacity_kw, 2),
            "total_capacity_mw": round(total_capacity_mw, 2),
            "capacity_factor": round(adjusted_capacity_factor * 100, 2),
            "avg_wind_speed": round(avg_wind_speed, 2)
        },
        "generation": {
            "annual_kwh": round(annual_generation_kwh, 2),
            "annual_mwh": round(annual_generation_mwh, 2),
            "daily_avg_kwh": round(annual_generation_kwh / 365, 2)
        },
        "cost": {
            "installation_inr": round(installation_cost, 2),
            "installation_crore": round(installation_cost / 10000000, 2),
            "annual_maintenance_inr": round(annual_maintenance, 2),
            "annual_maintenance_lakh": round(annual_maintenance / 100000, 2)
        },
        "environmental_impact": {
            "co2_reduction_yearly_tons": round(co2_reduction_yearly_tons, 2),
            "co2_reduction_25years_tons": round(co2_reduction_yearly_tons * 25, 2),
            "equivalent_trees": equivalent_trees,
            "pollution_reduction_yearly_kg": round(pollution_reduction_yearly, 2),
            "coal_saved_yearly_tons": round(coal_saved_yearly_tons, 2)
        }
    }


def convert_wind_suggestions_to_simple_language(suggestions):
    """
    Convert wind technical suggestions from wind.py to simple English statements
    Removes duplicates by grouping similar parameters
    
    Args:
        suggestions: Dict of parameter suggestions from wind.py
    
    Returns:
        List of unique simple English statements
    """
    simple_statements = []
    seen_base_params = set()  # Track base parameters to avoid duplicates
    
    for param, suggestion_text in suggestions.items():
        # Extract base parameter name (e.g., "WindSpeed" from "Q1-WindSpeed")
        if '-' in param:
            base_param = param.split('-', 1)[1]
        else:
            base_param = param
        
        # Skip if we've already added a suggestion for this base parameter
        if base_param in seen_base_params:
            continue
        
        seen_base_params.add(base_param)
        
        # The suggestion text from wind.py already contains the warning message
        # Just clean it up and add to simple statements
        simple_statements.append(suggestion_text)
    
    return simple_statements


def convert_suggestions_to_simple_language(suggestions, analysis_type):
    """
    Convert technical suggestions to simple English statements
    
    Args:
        suggestions: Dict of parameter suggestions
        analysis_type: 'solar' or 'wind'
    
    Returns:
        List of simple English statements
    """
    if analysis_type == 'wind':
        # Use specialized wind converter that handles wind.py output
        return convert_wind_suggestions_to_simple_language(suggestions)
    
    # Solar-specific conversion
    simple_statements = []
    
    # Parameter name mappings to simple language
    param_descriptions = {
        "GHI (kWh/m²/day)": "sunlight intensity",
        "DNI (kWh/m²/day)": "direct sunlight",
        "DHI (% of GHI)": "diffused sunlight ratio",
        "Snowfall (mm/year)": "snowfall levels",
        "Cloud cover": "cloud coverage",
        "Sunshine duration": "daily sunshine hours",
        "Ambient temperature": "temperature",
        "Relative humidity": "humidity",
        "Precipitation": "rainfall",
    }
    
    for param, suggestion in suggestions.items():
        # Get simple description
        simple_name = param_descriptions.get(param, param.lower())
        
        # Determine if increase or decrease needed
        if "Increase" in suggestion or "increase" in suggestion:
            statement = f"The {simple_name} is lower than expected for optimal {analysis_type} energy generation."
        elif "Reduce" in suggestion or "reduce" in suggestion:
            statement = f"The {simple_name} is higher than ideal for {analysis_type} energy setup."
        else:
            statement = f"The {simple_name} needs adjustment for better {analysis_type} performance."
        
        simple_statements.append(statement)
    
    return simple_statements


def generate_pdf_report(analysis_data):
    """Generate detailed PDF report with environmental impact and cost analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=20,
        alignment=1
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#4a5568'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    # Title
    story.append(Paragraph("Renewable Energy Site Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Timestamp
    story.append(Paragraph(
        f"<b>Generated:</b> {datetime.now().strftime('%d %B %Y, %I:%M %p')}", 
        styles['Normal']
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Check if restricted
    if analysis_data.get('restricted'):
        restriction_style = ParagraphStyle(
            'Restriction',
            parent=styles['Normal'],
            fontSize=14,
            textColor=colors.red,
            spaceAfter=15
        )
        story.append(Paragraph("<b>⚠️ SITE RESTRICTED FOR RENEWABLE ENERGY PROJECT</b>", restriction_style))
        story.append(Spacer(1, 0.15*inch))
        
        violations = analysis_data['restriction_details']['violations']
        if violations:
            story.append(Paragraph("<b>Safety Violations Detected:</b>", styles['Heading3']))
            story.append(Spacer(1, 0.1*inch))
            for v in violations:
                story.append(Paragraph(
                    f"• <b>{v['facility']}</b> ({v['type']}): Currently {v['distance']} km away. " +
                    f"Minimum required distance is {v['required_distance']} km. " +
                    f"The site is {v['shortage']} km too close.",
                    styles['Normal']
                ))
                story.append(Spacer(1, 0.05*inch))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    # Site Area Information
    if 'area' in analysis_data:
        story.append(Paragraph("<b>Site Area</b>", heading2_style))
        area_data = [
            ['Measurement', 'Value'],
            ['Square Meters', f"{analysis_data['area']['square_meters']:,.2f} m²"],
            ['Hectares', f"{analysis_data['area']['hectares']:,.2f} ha"],
            ['Acres', f"{analysis_data['area']['acres']:,.2f} acres"]
        ]
        area_table = Table(area_data, colWidths=[2.5*inch, 3.5*inch])
        area_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(area_table)
        story.append(Spacer(1, 0.25*inch))
    
    # Metadata Section
    story.append(Paragraph("<b>Site Location</b>", heading2_style))
    metadata = analysis_data['metadata']
    meta_data = [
        ['Property', 'Value'],
        ['Coordinates', f"{metadata['centroid'][0]:.4f}°N, {metadata['centroid'][1]:.4f}°E"],
        ['Weather Data Source', metadata['weather_source']],
    ]
    meta_table = Table(meta_data, colWidths=[2.5*inch, 3.5*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Infrastructure Analysis
    if 'infrastructure_analysis' in analysis_data:
        story.append(Paragraph("<b>Infrastructure Connectivity</b>", heading2_style))
        infra = analysis_data['infrastructure_analysis']
        
        score = infra.get('connectivity_score', 0)
        score_color = colors.green if score >= 70 else colors.orange if score >= 40 else colors.red
        
        story.append(Paragraph(
            f"<b>Connectivity Score: <font color='{score_color.hexval()}'>{score}/100</font></b>",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.15*inch))
        
        infra_data = [['Infrastructure Type', 'Nearest Distance', 'Facility Name']]
        for key, value in infra.items():
            if isinstance(value, dict) and 'distance_km' in value and value['distance_km']:
                infra_data.append([
                    key.replace('_', ' ').title(),
                    f"{value['distance_km']} km",
                    value.get('nearest', 'N/A')
                ])
        
        if len(infra_data) > 1:
            infra_table = Table(infra_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            infra_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(infra_table)
        story.append(PageBreak())
    
    # Solar Analysis
    if 'solar' in analysis_data:
        solar = analysis_data['solar']
        
        story.append(Paragraph("☀️ <b>Solar Energy Analysis</b>", heading2_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Summary
        feasible_color = colors.green if solar['feasible'] == 'Yes' else colors.red
        story.append(Paragraph(
            f"<b>Feasibility:</b> <font color='{feasible_color.hexval()}'>{solar['feasible']}</font> | " +
            f"<b>Suitability Score:</b> {solar['score']}% | " +
            f"<b>Category:</b> {solar['category']}",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Installation Details
        if 'installation' in solar:
            story.append(Paragraph("<b>Installation Capacity</b>", styles['Heading3']))
            inst = solar['installation']
            install_data = [
                ['Parameter', 'Value'],
                ['Number of Solar Panels', f"{inst['num_panels']:,}"],
                ['Usable Area', f"{inst['usable_area_m2']:,.2f} m²"],
                ['Total Capacity', f"{inst['total_capacity_mw']:.2f} MW ({inst['total_capacity_kw']:,.2f} kW)"],
                ['Capacity Factor', f"{inst['capacity_factor']}%"]
            ]
            install_table = Table(install_data, colWidths=[2.5*inch, 3.5*inch])
            install_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff4e6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(install_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Energy Generation
        if 'generation' in solar:
            story.append(Paragraph("<b>Energy Generation Potential</b>", styles['Heading3']))
            gen = solar['generation']
            gen_data = [
                ['Period', 'Generation'],
                ['Annual Generation', f"{gen['annual_mwh']:,.2f} MWh ({gen['annual_kwh']:,.2f} kWh)"],
                ['Daily Average', f"{gen['daily_avg_kwh']:,.2f} kWh"]
            ]
            gen_table = Table(gen_data, colWidths=[2.5*inch, 3.5*inch])
            gen_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff4e6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(gen_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Cost Analysis
        if 'cost' in solar:
            story.append(Paragraph("<b>Cost Analysis</b>", styles['Heading3']))
            cost = solar['cost']
            cost_data = [
                ['Cost Component', 'Amount (Rs)', 'Amount'],
                ['Installation Cost', f"Rs{cost['installation_inr']:,.2f}", f"Rs{cost['installation_crore']:.2f} Crore"],
                ['Annual Maintenance', f"Rs{cost['annual_maintenance_inr']:,.2f}", f"Rs{cost['annual_maintenance_lakh']:.2f} Lakh"]
            ]
            cost_table = Table(cost_data, colWidths=[2*inch, 2*inch, 2*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e9')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(cost_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Environmental Impact
        if 'environmental_impact' in solar:
            story.append(Paragraph("<b>Environmental Impact (vs Coal Power)</b>", styles['Heading3']))
            env = solar['environmental_impact']
            story.append(Paragraph(
                f"This solar installation can replace coal-based power generation and provide significant environmental benefits:",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.1*inch))
            
            env_data = [
                ['Impact Category', 'Annual', '25-Year Lifetime'],
                ['CO2 Emissions Avoided', f"{env['co2_reduction_yearly_tons']:,.2f} tons", 
                 f"{env['co2_reduction_25years_tons']:,.2f} tons"],
                ['Coal Dependency Reduced', f"{env['coal_saved_yearly_tons']:,.2f} tons", 
                 f"{env['coal_saved_yearly_tons'] * 25:,.2f} tons"],
                ['Air Pollution Reduced', f"{env['pollution_reduction_yearly_kg']:,.2f} kg", 
                 f"{env['pollution_reduction_yearly_kg'] * 25:,.2f} kg"],
                ['Equivalent Trees Planted', f"{env['equivalent_trees']:,} trees", 
                 f"{env['equivalent_trees'] * 25:,} trees"]
            ]
            env_table = Table(env_data, colWidths=[2*inch, 2*inch, 2*inch])
            env_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e9')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(env_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Simple Suggestions
        if solar.get('simple_suggestions'):
            story.append(Paragraph("<b>Site Conditions Assessment</b>", styles['Heading3']))
            story.append(Paragraph(
                "The following site conditions may affect solar energy generation:",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.1*inch))
            for suggestion in solar['simple_suggestions']:
                story.append(Paragraph(f"• {suggestion}", styles['Normal']))
                story.append(Spacer(1, 0.05*inch))
        else:
            story.append(Paragraph(
                "✅ <b>All site conditions are within optimal ranges for solar energy generation.</b>",
                styles['Normal']
            ))
        
        story.append(PageBreak())
    
    # Wind Analysis
    if 'wind' in analysis_data:
        wind = analysis_data['wind']
        
        story.append(Paragraph("💨 <b>Wind Energy Analysis</b>", heading2_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Summary
        feasible_color = colors.green if wind['feasible'] == 'Yes' else colors.red
        story.append(Paragraph(
            f"<b>Feasibility:</b> <font color='{feasible_color.hexval()}'>{wind['feasible']}</font> | " +
            f"<b>Suitability Score:</b> {wind['score']}% | " +
            f"<b>Category:</b> {wind['category']}",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Installation Details
        if 'installation' in wind:
            story.append(Paragraph("<b>Installation Capacity</b>", styles['Heading3']))
            inst = wind['installation']
            install_data = [
                ['Parameter', 'Value'],
                ['Number of Wind Turbines', f"{inst['num_turbines']}"],
                ['Turbine Capacity (each)', f"{inst['turbine_capacity_mw']} MW"],
                ['Total Capacity', f"{inst['total_capacity_mw']:.2f} MW ({inst['total_capacity_kw']:,.2f} kW)"],
                ['Average Wind Speed', f"{inst['avg_wind_speed']} m/s"],
                ['Capacity Factor', f"{inst['capacity_factor']}%"]
            ]
            install_table = Table(install_data, colWidths=[2.5*inch, 3.5*inch])
            install_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f7ff')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(install_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Energy Generation
        if 'generation' in wind:
            story.append(Paragraph("<b>Energy Generation Potential</b>", styles['Heading3']))
            gen = wind['generation']
            gen_data = [
                ['Period', 'Generation'],
                ['Annual Generation', f"{gen['annual_mwh']:,.2f} MWh ({gen['annual_kwh']:,.2f} kWh)"],
                ['Daily Average', f"{gen['daily_avg_kwh']:,.2f} kWh"]
            ]
            gen_table = Table(gen_data, colWidths=[2.5*inch, 3.5*inch])
            gen_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f7ff')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(gen_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Cost Analysis
        if 'cost' in wind:
            story.append(Paragraph("<b>Cost Analysis</b>", styles['Heading3']))
            cost = wind['cost']
            cost_data = [
                ['Cost Component', 'Amount (Rs)', 'Amount'],
                ['Installation Cost', f"Rs{cost['installation_inr']:,.2f}", f"Rs{cost['installation_crore']:.2f} Crore"],
                ['Annual Maintenance', f"Rs{cost['annual_maintenance_inr']:,.2f}", f"Rs{cost['annual_maintenance_lakh']:.2f} Lakh"]
            ]
            cost_table = Table(cost_data, colWidths=[2*inch, 2*inch, 2*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e9')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(cost_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Environmental Impact
        if 'environmental_impact' in wind:
            story.append(Paragraph("<b>Environmental Impact (vs Coal Power)</b>", styles['Heading3']))
            env = wind['environmental_impact']
            story.append(Paragraph(
                f"This wind farm can replace coal-based power generation and provide significant environmental benefits:",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.1*inch))
            
            env_data = [
                ['Impact Category', 'Annual', '25-Year Lifetime'],
                ['CO2 Emissions Avoided', f"{env['co2_reduction_yearly_tons']:,.2f} tons", 
                 f"{env['co2_reduction_25years_tons']:,.2f} tons"],
                ['Coal Dependency Reduced', f"{env['coal_saved_yearly_tons']:,.2f} tons", 
                 f"{env['coal_saved_yearly_tons'] * 25:,.2f} tons"],
                ['Air Pollution Reduced', f"{env['pollution_reduction_yearly_kg']:,.2f} kg", 
                 f"{env['pollution_reduction_yearly_kg'] * 25:,.2f} kg"],
                ['Equivalent Trees Planted', f"{env['equivalent_trees']:,} trees", 
                 f"{env['equivalent_trees'] * 25:,} trees"]
            ]
            env_table = Table(env_data, colWidths=[2*inch, 2*inch, 2*inch])
            env_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e9')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(env_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Simple Suggestions
        if wind.get('simple_suggestions'):
            story.append(Paragraph("<b>Site Conditions Assessment</b>", styles['Heading3']))
            story.append(Paragraph(
                "The following site conditions may affect wind energy generation:",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.1*inch))
            for suggestion in wind['simple_suggestions']:
                story.append(Paragraph(f"• {suggestion}", styles['Normal']))
                story.append(Spacer(1, 0.05*inch))
        else:
            story.append(Paragraph(
                "✅ <b>All site conditions are within optimal ranges for wind energy generation.</b>",
                styles['Normal']
            ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


@app.route('/')
def landing():
    """Serve the landing page"""
    # If already logged in, redirect to dashboard
    session_token = session.get('session_token')
    if session_token:
        validation = validate_session(session_token)
        if validation['valid']:
            return redirect(url_for('index'))
    return render_template('landing.html')


@app.route('/index')
@login_required
def index():
    """Serve the main map/analysis page (protected route)"""
    return render_template('index.html')


# Authentication Routes
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        company = data.get('company')
        
        # Validation
        if not name or not email or not password:
            return jsonify({
                'success': False,
                'error': 'Name, email, and password are required'
            }), 400
        
        if len(password) < 6:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 6 characters'
            }), 400
        
        # Register user
        result = register_user(name, email, password, company)
        
        if not result['success']:
            return jsonify(result), 400
        
        # Auto-login after registration
        login_result = login_user(email, password)
        
        if login_result['success']:
            session['session_token'] = login_result['session_token']
            session['user'] = login_result['user']
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'user': login_result['user']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'error': 'Email and password are required'
            }), 400
        
        result = login_user(email, password)
        
        if not result['success']:
            return jsonify(result), 401
        
        # Store session
        session['session_token'] = result['session_token']
        session['user'] = result['user']
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': result['user']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    try:
        session_token = session.get('session_token')
        
        if session_token:
            logout_user(session_token)
        
        session.clear()
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current user information"""
    try:
        return jsonify({
            'success': True,
            'user': request.current_user
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    """Main endpoint for site analysis with environmental impact and cost (protected route)"""
    try:
        data = request.get_json()
        
        # Extract coordinates and analysis type
        coordinates = data.get('coordinates', [])
        analysis_type = data.get('analysis_type', 'both')
        
        if not coordinates or len(coordinates) < 3:
            return jsonify({
                'success': False,
                'error': 'Please provide at least 3 coordinates for polygon'
            }), 400
        
        # Calculate area
        area_m2 = calculate_polygon_area(coordinates)
        area_hectares = area_m2 / 10000
        area_acres = area_m2 / 4047
        
        # Convert to required format [(lat, lon), ...]
        polygon_coords = [(coord['lat'], coord['lng']) for coord in coordinates]
        
        # Extract data using analyzer
        print(f"📍 Analyzing polygon with {len(polygon_coords)} points...")
        print(f"   Area: {area_hectares:.2f} hectares ({area_acres:.2f} acres)")
        print(f"   Analysis Type: {analysis_type}")
        print(f"   User: {request.current_user['email']}")
        
        extracted_data = extractor.extract_from_polygon(
            polygon_coords, 
            save_to_file=False
        )
        
        # Check if site is restricted
        if not extracted_data.get('success', True):
            return jsonify({
                'success': False,
                'restricted': True,
                'restriction_details': extracted_data.get('restriction_details', {}),
                'message': extracted_data.get('message', 'Site is restricted')
            })
        
        # Determine categories
        def get_category(score):
            if score >= 75:
                return "Excellent"
            elif score >= 60:
                return "Good"
            elif score >= 40:
                return "Marginal"
            else:
                return "Poor"
        
        # Prepare response
        response = {
            'success': True,
            'restricted': False,
            'analysis_type': analysis_type,
            'area': {
                'square_meters': round(area_m2, 2),
                'hectares': round(area_hectares, 2),
                'acres': round(area_acres, 2)
            },
            'metadata': {
                'centroid': extracted_data['metadata']['centroid'],
                'weather_source': extracted_data['metadata']['source'],
                'osm_elements': len(extracted_data.get('infrastructure_data', {}).get('elements', []))
            },
            'infrastructure_analysis': extracted_data.get('infrastructure_analysis', {}),
            'restriction_check': extracted_data.get('restriction_check', {})
        }
        
        # Analyze based on type
        if analysis_type in ['solar', 'both']:
            solar_label, solar_score, solar_suggestions = analyze_solar_site_api(
                extracted_data['solar_site']
            )
            
            # Convert suggestions to simple language
            simple_suggestions = convert_suggestions_to_simple_language(solar_suggestions, 'solar')
            
            # Calculate environmental impact and cost
            solar_impact = calculate_solar_impact_and_cost(area_m2, solar_score)
            
            response['solar'] = {
                'feasible': solar_label,
                'score': solar_score,
                'category': get_category(solar_score),
                'suggestions': solar_suggestions,
                'simple_suggestions': simple_suggestions,
                'raw_data': extracted_data['solar_site'],
                'installation': solar_impact['installation'],
                'generation': solar_impact['generation'],
                'cost': solar_impact['cost'],
                'environmental_impact': solar_impact['environmental_impact']
            }
        
        if analysis_type in ['wind', 'both']:
            wind_label, wind_score, wind_suggestions = analyze_wind_site_api(
                extracted_data['wind_site']
            )
            
            # Convert suggestions to simple language (handles deduplication)
            simple_suggestions = convert_suggestions_to_simple_language(wind_suggestions, 'wind')
            
            # Get average wind speed from the data
            avg_wind_speed = extracted_data['wind_site'].get('Yearly-WindSpeed', 5.0)
            
            # Calculate environmental impact and cost
            wind_impact = calculate_wind_impact_and_cost(area_m2, wind_score, avg_wind_speed)
            
            response['wind'] = {
                'feasible': wind_label,
                'score': wind_score,
                'category': get_category(wind_score),
                'suggestions': wind_suggestions,
                'simple_suggestions': simple_suggestions,
                'raw_data': extracted_data['wind_site'],
                'installation': wind_impact['installation'],
                'generation': wind_impact['generation'],
                'cost': wind_impact['cost'],
                'environmental_impact': wind_impact['environmental_impact']
            }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/download-report', methods=['POST'])
@login_required
def download_report():
    """Generate and download PDF report"""
    try:
        data = request.get_json()
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(data)
        
        # Send file
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'renewable_energy_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    
    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': solar_rf is not None and wind_rf is not None,
        'auth_enabled': True
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Renewable Energy Analysis Server")
    print("="*50)
    print("📊 Models:", "Loaded ✓" if solar_rf and wind_rf else "Not Loaded ⚠")
    print("🔐 Authentication:", "Enabled ✓")
    print("🌐 Server: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)