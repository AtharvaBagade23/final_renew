# Renewable Energy Site Analysis Platform

A machine learning-powered web application for analyzing and evaluating solar and wind energy site suitability across India using real weather data, terrain analysis, and geospatial features.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Project Architecture](#project-architecture)
- [Datasets](#datasets)
- [Machine Learning Models](#machine-learning-models)
- [API Endpoints](#api-endpoints)
- [Usage Guide](#usage-guide)
- [Technologies Used](#technologies-used)

---

## 🎯 Features

- **Solar Site Analysis**: Evaluate solar potential using location data, weather patterns, and terrain features
- **Wind Site Analysis**: Assess wind energy suitability with wind speed analysis and obstacle detection
- **Interactive Map Interface**: Draw and analyze custom polygon areas for energy site evaluation
- **Real Weather Data Integration**: Fetches actual weather data from multiple APIs (Open-Meteo, Weatherbit, Visual Crossing)
- **Geospatial Analysis**: OpenStreetMap integration for terrain and infrastructure analysis
- **ML Predictions**: Random Forest models trained on Indian renewable energy datasets
- **PDF Report Generation**: Generate detailed analysis reports with visualizations
- **User Authentication**: Secure login/registration system
- **Restricted Area Detection**: Identifies protected zones and sensitive areas

---

## 📁 Project Structure

```
final_renew/
├── app.py                           # Main Flask application
├── train_and_save_models.py        # Model training script
├── analyzer.py                      # Data extraction and analysis logic
├── solar.py                         # Solar model training & analysis
├── wind.py                          # Wind model training & analysis
├── auth.py                          # User authentication logic
├── requirements.txt                 # Python dependencies
├── Datasets/
│   ├── Solar_Sites_Dataset_India.csv
│   └── Wind_Sites_Dataset_India.csv
├── templates/
│   ├── index.html                  # Main analysis interface
│   └── landing.html                # Landing page
├── performance/                     # Performance metrics storage
└── __pycache__/                    # Python cache
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.7+** (Recommended: Python 3.9 or 3.10)
- **pip** (Python package manager)
- **Git** (optional, for cloning)

### Step 1: Clone or Download the Repository

```bash
# Clone from GitHub (if available)
git clone <repository-url>
cd final_renew

# OR simply navigate to the project folder
cd path/to/final_renew
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- Flask: Web framework
- Flask-Cors: Cross-Origin Resource Sharing
- scikit-learn: Machine learning library
- pandas: Data manipulation
- numpy: Numerical computing
- joblib: Model persistence
- matplotlib: Data visualization
- reportlab: PDF generation
- requests: HTTP requests
- shapely: Geometric operations
- gunicorn: WSGI HTTP server

---

## 🏃 How to Run

### Option 1: Train Models & Run Application (First Time Setup)

```bash
# Step 1: Train and save the models
python train_and_save_models.py

# Step 2: Run the Flask application
python app.py
```

**Expected Output:**
```
✓ Models loaded successfully
 * Running on http://localhost:5000
 * WARNING in use_reloader: This is a development server...
```

### Option 2: Run Application with Pre-trained Models

If models are already trained (`.pkl` files exist):

```bash
python app.py
```

### Step 3: Access the Web Interface

Open your browser and navigate to:

```
http://localhost:5000
```

You should see:
1. **Landing Page**: Introduction and login/register options
2. **Main Dashboard**: Interactive map interface with analysis tools

---

## 🏗️ Project Architecture

### 1. **Backend Architecture**

```
Flask Application (app.py)
    ├── User Authentication (auth.py)
    ├── Data Analysis Engine (analyzer.py)
    ├── ML Model Inference
    │   ├── Solar Predictions
    │   └── Wind Predictions
    └── API Routes
        ├── /api/analyze (POST)
        ├── /api/report (GET)
        └── /api/sites (GET/POST)
```

### 2. **Data Flow**

```
User Input (Polygon Coordinates)
    ↓
Geographic Validation (OSM, Restricted Areas)
    ↓
Weather Data Extraction (Multiple APIs)
    ↓
Terrain & Infrastructure Analysis
    ↓
Feature Engineering
    ↓
ML Model Prediction (Solar + Wind)
    ↓
Report Generation (PDF)
    ↓
User Output
```

### 3. **Model Pipeline**

```
Raw Dataset
    ↓
Data Cleaning & Encoding
    ↓
Noise Addition (5%) & Label Flipping (5-10%)
    ↓
Train-Test Split (60-40)
    ↓
Random Forest Training (50 estimators, max_depth=5)
    ↓
Model Serialization (.pkl files)
    ↓
Production Deployment
```

---

## 📊 Datasets

### Solar Sites Dataset (`Solar_Sites_Dataset_India.csv`)

**Purpose:** Train the solar energy suitability prediction model

**Features:**
- **Geographic Features**: Latitude, Longitude
- **Weather Features**: 
  - Average solar irradiance (kWh/m²/day)
  - Temperature variations
  - Cloud cover percentage
  - Humidity levels
- **Terrain Features**:
  - Elevation (meters)
  - Slope percentage
  - Land cover type
- **Infrastructure Features**:
  - Distance to nearest road
  - Distance to power grid
  - Distance to water source
- **Label**: "Yes/No" (Suitable/Not Suitable for solar energy)

**Dataset Size:** Multiple thousand records

**Data Preprocessing:**
- Missing values handled
- Numeric features normalized
- Categorical features encoded
- 5% Gaussian noise added to reduce overfitting
- 10% of labels randomly flipped

### Wind Sites Dataset (`Wind_Sites_Dataset_India.csv`)

**Purpose:** Train the wind energy suitability prediction model

**Features:**
- **Geographic Features**: Latitude, Longitude
- **Weather Features**:
  - Average wind speed (m/s)
  - Wind speed variations (min/max)
  - Wind direction consistency
  - Atmospheric pressure
- **Terrain Features**:
  - Elevation (meters)
  - Terrain roughness
  - Obstacle density
  - Distance to vegetation/forests
- **Infrastructure Features**:
  - Distance to transmission lines
  - Distance to roads
  - Distance to urban areas
- **Label**: "Yes/No" (Suitable/Not Suitable for wind energy)

**Dataset Size:** Multiple thousand records

**Data Preprocessing:**
- Missing values handled
- 35% Gaussian noise added (higher than solar due to wind variability)
- 10% of labels randomly flipped
- Stratified train-test split

---

## 🤖 Machine Learning Models

### Solar Suitability Model

**Model Type:** Random Forest Classifier

**Configuration:**
- **Estimators:** 50 decision trees
- **Max Depth:** 5
- **Random State:** 42 (for reproducibility)
- **Test Size:** 40%
- **Train Size:** 60%

**Performance Metrics:**
- Trained on preprocessed Solar Sites Dataset
- Provides binary classification (Suitable/Not Suitable)
- Feature importance analysis included
- Confusion matrix visualization

**Serialization:** Saved as `solar_model.pkl`

### Wind Suitability Model

**Model Type:** Random Forest Classifier

**Configuration:**
- **Estimators:** 50 decision trees
- **Max Depth:** 5
- **Random State:** 42 (for reproducibility)
- **Test Size:** 40%
- **Train Size:** 60%

**Performance Metrics:**
- Trained on preprocessed Wind Sites Dataset
- Provides binary classification (Suitable/Not Suitable)
- Higher noise level (35%) accounts for wind variability
- Feature importance analysis included
- Confusion matrix visualization

**Serialization:** Saved as `wind_model.pkl`

### Model Training Process

```bash
python train_and_save_models.py
```

This script:
1. Loads the Solar and Wind datasets from `Datasets/` folder
2. Encodes labels ("Yes"→1, "No"→0)
3. Adds controlled noise to reduce overfitting
4. Flips 5-10% of labels randomly for regularization
5. Performs stratified train-test split
6. Trains Random Forest models
7. Saves models as pickle files:
   - `solar_model.pkl` (Model + Feature names)
   - `wind_model.pkl` (Model + Feature names)

### Feature Importance

Both models generate feature importance visualizations:
- Solar: `feature_importance.png`
- Wind: `wind_feature_importance.png`

These show which features contribute most to predictions.

---

## 🔌 API Endpoints

### Authentication Endpoints

#### **Register User**
```
POST /register
Content-Type: application/json

{
  "username": "user123",
  "email": "user@example.com",
  "password": "securepassword"
}

Response: 
{
  "success": true,
  "message": "User registered successfully"
}
```

#### **Login User**
```
POST /login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}

Response:
{
  "success": true,
  "user_id": "user_uuid",
  "message": "Login successful"
}
```

#### **Logout**
```
GET /logout

Response:
{
  "success": true,
  "message": "Logged out successfully"
}
```

### Analysis Endpoints

#### **Analyze Site Suitability**
```
POST /api/analyze
Content-Type: application/json

{
  "polygon": [
    [28.5355, 77.3910],
    [28.5360, 77.3910],
    [28.5360, 77.3915],
    [28.5355, 77.3915]
  ],
  "energy_type": "solar"  # or "wind" or "both"
}

Response:
{
  "success": true,
  "solar_suitable": true,
  "wind_suitable": false,
  "solar_confidence": 0.92,
  "wind_confidence": 0.45,
  "weather_data": {...},
  "terrain_data": {...},
  "recommendations": [...]
}
```

#### **Generate Report**
```
GET /api/report?site_id=<site_uuid>

Response: PDF file download
```

#### **Get Site History**
```
GET /api/sites

Response:
[
  {
    "site_id": "uuid",
    "location": "City Name",
    "coordinates": [...],
    "analysis_date": "2025-11-20",
    "solar_result": "Suitable",
    "wind_result": "Not Suitable"
  },
  ...
]
```

---

## 📖 Usage Guide

### Step 1: User Registration/Login

1. Open the application at `http://localhost:5000`
2. Click "Register" to create a new account or "Login" with existing credentials
3. Provide email and password
4. You'll be redirected to the main dashboard

### Step 2: Draw an Analysis Area

1. On the interactive map, locate your area of interest
2. Click the "Draw Polygon" tool (usually top-left of map)
3. Click on the map to create polygon vertices
4. Complete the polygon by clicking the first point again or double-clicking
5. Your polygon will be highlighted in blue

### Step 3: Analyze Site Suitability

1. After drawing the polygon, click "Analyze" button
2. Select analysis type:
   - Solar Only
   - Wind Only
   - Both Solar & Wind
3. The system will:
   - Check for restricted areas
   - Fetch real weather data
   - Analyze terrain features
   - Run ML predictions
   - Generate recommendations

### Step 4: View Results

Results displayed on screen include:
- **Suitability Status**: Green (Suitable) or Red (Not Suitable)
- **Confidence Score**: 0-100%
- **Key Findings**: 
  - Average weather conditions
  - Terrain characteristics
  - Infrastructure availability
  - Recommended capacity

### Step 5: Generate PDF Report

1. Click "Generate Report" button
2. The system will create a comprehensive PDF with:
   - Site coordinates and maps
   - Weather analysis
   - Suitability predictions
   - Confidence metrics
   - Recommendations
3. PDF downloads automatically

### Step 6: View Site History

1. Click "My Sites" or "History"
2. View all previous analyses
3. Click on any site to see detailed results
4. Re-download reports if needed

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| **Backend** | Flask, Python 3.9+ |
| **Frontend** | HTML5, CSS3, JavaScript, Leaflet.js |
| **ML/Data** | scikit-learn, pandas, numpy |
| **APIs** | Open-Meteo, Weatherbit, Visual Crossing |
| **Geospatial** | OpenStreetMap, Shapely, Overpass API |
| **Data Processing** | pandas, numpy, scikit-learn |
| **Visualization** | matplotlib, reportlab (PDF) |
| **Web Server** | Flask, Gunicorn |
| **Security** | Flask-Session, password hashing |
| **CORS** | Flask-CORS |

---

## 🔐 Security Features

- Secure session management with random secret keys
- Password hashing for user credentials
- CORS enabled for secure API calls
- Input validation for polygon coordinates
- Restricted area detection prevents analysis in protected zones

---

## 📊 Model Performance Monitoring

After training, check model performance:

```bash
# Solar model metrics
python solar.py  # Generates confusion_matrix.png and feature_importance.png

# Wind model metrics
python wind.py   # Generates wind_confusion_matrix.png and wind_feature_importance.png
```

**Saved Artifacts:**
- `confusion_matrix.png`: Solar prediction accuracy visualization
- `wind_confusion_matrix.png`: Wind prediction accuracy visualization
- `feature_importance.png`: Solar feature importance chart
- `wind_feature_importance.png`: Wind feature importance chart

---

## 🐛 Troubleshooting

### Issue: Models not loading
```
Error: "Could not load models"
Solution: Run python train_and_save_models.py to train models first
```

### Issue: Weather API failures
```
Solution: The app automatically falls back to alternative APIs
Check internet connection and API keys in analyzer.py
```

### Issue: Port already in use
```bash
# Kill the process on port 5000
netstat -ano | findstr :5000  # Windows
kill -9 <PID>  # Linux/Mac
```

### Issue: Dataset not found
```
Solution: Ensure Datasets folder contains:
- Solar_Sites_Dataset_India.csv
- Wind_Sites_Dataset_India.csv
```

---

## 📝 Environment Variables

Create a `.env` file if needed (currently using hardcoded keys):

```
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
API_KEY_WEATHERBIT=9a1ed9a8bd9d4e52b59889854b904bc6
API_KEY_VISUAL_CROSSING=WEZJL49X6X7C6S6M8TENPDX2G
```

---

## 🚀 Deployment

For production deployment:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## 📧 Support & Contribution

For issues or contributions:
1. Check the troubleshooting section
2. Review model configuration in `solar.py` and `wind.py`
3. Verify dataset integrity in `Datasets/` folder
4. Check API credentials in `analyzer.py`

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 🎓 Project Overview

This project demonstrates:
- **ML Pipeline**: End-to-end machine learning workflow
- **Web Development**: Full-stack Flask application
- **Data Science**: Real-world renewable energy analysis
- **Geospatial Analysis**: Working with geographic coordinates and maps
- **API Integration**: Multiple external API integrations
- **User Management**: Authentication and session handling
- **Report Generation**: Automated PDF report creation

---

## 📞 Quick Reference

```bash
# Training
python train_and_save_models.py

# Running the app
python app.py

# Access the web app
http://localhost:5000

# Run solar analysis script
python solar.py

# Run wind analysis script
python wind.py
```

---

**Last Updated:** November 20, 2025
**Project Version:** 1.0.0
**Python Version:** 3.9+
