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

# Import the DataExtractor from analyzer.py
sys.path.insert(0, os.path.dirname(__file__))
from analyzer import DataExtractor
from auth import (
    register_user, login_user, logout_user, 
    validate_session, login_required, get_user_by_id
)

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
    Analyze wind site with quarter-wise evaluation.
    Returns:
        label: "Yes" / "No"
        suitability_percent: 0-100
        suggestions: dict of parameter adjustments per quarter
    """
    if wind_rf is None:
        return "Unknown", 0, {"error": "Model not loaded"}

    df_site = pd.DataFrame([site_data])

    # Ensure all expected features exist
    for col in wind_features:
        if col not in df_site.columns:
            df_site[col] = 0
    df_site = df_site[wind_features]

    # Predict using model (yearly data only)
    prediction = wind_rf.predict(df_site)[0]
    label = "Yes" if prediction == 1 else "No"

    # Define thresholds
    thresholds = {
        "WindSpeed": 4,
        "WindGustSpeed": 50,
        "AirTemperature": (-30, 40),
        "AirPressure": (950, 1050),
        "RelativeHumidity": 90,
        "Precipitation": 2000,
        "Elevation": 1500,
        "Slope": 15,
        "TurbulenceIntensity": 20
    }

    score = 0
    suggestions = {}

    # Evaluate each quarter + yearly
    for param, threshold in thresholds.items():
        matching_cols = [c for c in df_site.columns if param in c]
        for col in matching_cols:
            value = float(df_site[col].iloc[0])

            # Handle range thresholds (like AirTemperature, AirPressure)
            if isinstance(threshold, tuple):
                low, high = threshold
                if low <= value <= high:
                    score += 1
                else:
                    if value < low:
                        suggestions[col] = f"Increase by {round(low - value, 2)}"
                    else:
                        suggestions[col] = f"Reduce by {round(value - high, 2)}"
            else:
                # Single-value thresholds
                if param in ["RelativeHumidity", "Precipitation", "Slope", "Elevation", "TurbulenceIntensity"]:
                    if value < threshold:
                        score += 1
                    else:
                        suggestions[col] = f"Reduce by {round(value - threshold, 2)}"
                else:
                    if value >= threshold:
                        score += 1
                    else:
                        suggestions[col] = f"Increase by {round(threshold - value, 2)}"

    total_params = sum([len([c for c in df_site.columns if param in c]) for param in thresholds])
    suitability_percent = round((score / total_params) * 100, 2) if total_params > 0 else 0

    return label, suitability_percent, suggestions


def generate_pdf_report(analysis_data):
    """Generate detailed PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("Renewable Energy Site Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Timestamp
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Check if restricted
    if analysis_data.get('restricted'):
        restriction_data = [[Paragraph("<b>⚠️ SITE RESTRICTED</b>", styles['Heading2'])]]
        story.append(Table(restriction_data, colWidths=[6*inch]))
        story.append(Spacer(1, 0.2*inch))
        
        violations = analysis_data['restriction_details']['violations']
        if violations:
            story.append(Paragraph("<b>Safety Violations:</b>", styles['Heading3']))
            for v in violations:
                story.append(Paragraph(
                    f"• {v['facility']} ({v['type']}): {v['distance']} km (Required: {v['required_distance']} km)",
                    styles['Normal']
                ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    # Metadata Section
    story.append(Paragraph("<b>Site Information</b>", styles['Heading2']))
    metadata = analysis_data['metadata']
    meta_data = [
        ['Centroid', f"{metadata['centroid'][0]:.4f}, {metadata['centroid'][1]:.4f}"],
        ['Weather Source', metadata['weather_source']],
        ['OSM Elements', str(metadata['osm_elements'])]
    ]
    meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Infrastructure Analysis
    if 'infrastructure_analysis' in analysis_data:
        story.append(Paragraph("<b>Infrastructure Connectivity</b>", styles['Heading2']))
        infra = analysis_data['infrastructure_analysis']
        
        story.append(Paragraph(f"<b>Connectivity Score: {infra.get('connectivity_score', 'N/A')}/100</b>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        infra_data = []
        for key, value in infra.items():
            if isinstance(value, dict) and 'distance_km' in value:
                if value['distance_km']:
                    infra_data.append([
                        key.replace('_', ' ').title(),
                        f"{value['distance_km']} km",
                        value.get('nearest', 'N/A')
                    ])
        
        if infra_data:
            infra_table = Table([['Infrastructure', 'Distance', 'Nearest']] + infra_data, 
                               colWidths=[2*inch, 1.5*inch, 2.5*inch])
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
        story.append(Paragraph("<b>☀️ Solar Energy Analysis</b>", styles['Heading2']))
        solar = analysis_data['solar']
        
        solar_summary = [
            ['Feasibility', solar['feasible']],
            ['Suitability Score', f"{solar['score']}%"],
            ['Category', solar['category']]
        ]
        solar_table = Table(solar_summary, colWidths=[2*inch, 4*inch])
        solar_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff4e6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(solar_table)
        story.append(Spacer(1, 0.2*inch))
        
        if solar['suggestions']:
            story.append(Paragraph("<b>Improvement Suggestions:</b>", styles['Heading3']))
            for param, suggestion in solar['suggestions'].items():
                story.append(Paragraph(f"• <b>{param}:</b> {suggestion}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Wind Analysis
    if 'wind' in analysis_data:
        story.append(Paragraph("<b>💨 Wind Energy Analysis</b>", styles['Heading2']))
        wind = analysis_data['wind']
        
        wind_summary = [
            ['Feasibility', wind['feasible']],
            ['Suitability Score', f"{wind['score']}%"],
            ['Category', wind['category']]
        ]
        wind_table = Table(wind_summary, colWidths=[2*inch, 4*inch])
        wind_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f7ff')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(wind_table)
        story.append(Spacer(1, 0.2*inch))
        
        if wind['suggestions']:
            story.append(Paragraph("<b>Improvement Suggestions:</b>", styles['Heading3']))
            for param, suggestion in wind['suggestions'].items():
                story.append(Paragraph(f"• <b>{param}:</b> {suggestion}", styles['Normal']))
    
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
    """Main endpoint for site analysis (protected route)"""
    try:
        data = request.get_json()
        
        # Extract coordinates and analysis type
        coordinates = data.get('coordinates', [])
        analysis_type = data.get('analysis_type', 'both')  # 'solar', 'wind', or 'both'
        
        if not coordinates or len(coordinates) < 3:
            return jsonify({
                'success': False,
                'error': 'Please provide at least 3 coordinates for polygon'
            }), 400
        
        # Convert to required format [(lat, lon), ...]
        polygon_coords = [(coord['lat'], coord['lng']) for coord in coordinates]
        
        # Extract data using analyzer
        print(f"📍 Analyzing polygon with {len(polygon_coords)} points...")
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
            response['solar'] = {
                'feasible': solar_label,
                'score': solar_score,
                'category': get_category(solar_score),
                'suggestions': solar_suggestions,
                'raw_data': extracted_data['solar_site']
            }
        
        if analysis_type in ['wind', 'both']:
            wind_label, wind_score, wind_suggestions = analyze_wind_site_api(
                extracted_data['wind_site']
            )
            response['wind'] = {
                'feasible': wind_label,
                'score': wind_score,
                'category': get_category(wind_score),
                'suggestions': wind_suggestions,
                'raw_data': extracted_data['wind_site']
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


@app.route('/generate-report', methods=['POST'])
@login_required
def generate_report():
    """Generate and download PDF report (protected route)"""
    try:
        data = request.get_json()
        pdf_buffer = generate_pdf_report(data)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'renewable_energy_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    except Exception as e:
        print(f"❌ PDF Generation Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


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