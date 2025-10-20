import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------
# 1. Load Dataset
# -------------------------
DATA_PATH = "./Datasets/Wind_Sites_Dataset_India.csv"  # Path to the dataset
df = pd.read_csv(DATA_PATH)

# Encode Label
label_col = "Label"
df[label_col] = df[label_col].map({"Yes": 1, "No": 0})

# -------------------------
# 2. Add Noise (~5%) to Reduce Overfitting
# -------------------------
noise_level = 0.12  
numeric_cols = df.select_dtypes(include=np.number).columns.drop(label_col)

for col in numeric_cols:
    df[col] += np.random.normal(0, noise_level * df[col].std(), size=len(df))

# Slightly flip 5% of labels
flip_rate = 0.05
flip_indices = np.random.choice(df.index, size=int(len(df)*flip_rate), replace=False)
df.loc[flip_indices, label_col] = 1 - df.loc[flip_indices, label_col]

# -------------------------
# 3. Features and Target
# -------------------------
X = df.drop(label_col, axis=1)
y = df[label_col]

# -------------------------
# 4. Initialize and Train Model
# -------------------------
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

def train_model(verbose=True):
    """Train and evaluate the model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    
    rf.fit(X_train, y_train)
    
    if verbose:
        # Evaluate Model
        y_pred = rf.predict(X_test)
        print("======== MODEL PERFORMANCE ========")
        print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # Feature Importance
        importances = rf.feature_importances_
        features = X.columns
        plt.figure(figsize=(10,6))
        plt.barh(features, importances, color='skyblue')
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig("wind_feature_importance.png")
        print("\nFeature importance chart saved as 'wind_feature_importance.png'")

# Train model immediately with no output
train_model(verbose=False)

# If run directly, train again with output
if __name__ == "__main__":
    train_model(verbose=True)

# -------------------------
# 8. Wind Suitability Scoring Function (FIXED!)
# -------------------------
def wind_suitability_score(site_dict):
    """
    Input:
        site_dict (dict): Dictionary of site parameters matching dataset columns.
    Output:
        (label, score_percent, suggestions)
    
    FIXED: Preserves real API data, only adds missing columns
    """
    df_site = pd.DataFrame([site_dict])
    
    # DEBUG: Print what we received
    print(f"\n🔍 DEBUG wind.py - Received data:")
    print(f"   Keys received: {len(site_dict)}")
    for key in ['Q1-WindSpeed', 'Q2-WindSpeed', 'Q3-WindSpeed', 'Q4-WindSpeed', 'Yearly-WindSpeed']:
        if key in site_dict:
            print(f"   {key}: {site_dict[key]:.2f} m/s")
    
    # CRITICAL FIX: Only add columns that are TRULY missing
    # Do NOT overwrite existing columns!
    missing_cols = [col for col in X.columns if col not in df_site.columns]
    
    if missing_cols:
        print(f"    Adding {len(missing_cols)} missing columns with mean values:")
        for col in missing_cols[:5]:  # Show first 5
            print(f"      - {col}")
        
        for col in missing_cols:
            df_site[col] = X[col].mean()
    else:
        print(f"   ✅ All required columns present!")
    
    # Reorder to match training data
    df_site = df_site[X.columns]
    
    # DEBUG: Check values after alignment
    print(f"\n   After alignment:")
    for key in ['Q1-WindSpeed', 'Q2-WindSpeed', 'Yearly-WindSpeed']:
        if key in df_site.columns:
            print(f"   {key}: {df_site[key].iloc[0]:.2f} m/s")

    # Predict feasibility
    prediction = rf.predict(df_site)[0]
    label = "Yes" if prediction == 1 else "No"

    # CORRECTED THRESHOLDS
    thresholds = {
        "WindSpeed": {
            "min": 17.0,
            "optimal": 14.0,
            "type": "higher_better"
        },
        "WindGustSpeed": {
            "min": 8.0,
            "optimal": 15.0,
            "type": "higher_better"
        },
        "AirTemperature": {
            "min": -30,
            "max": 40,
            "type": "range"
        },
        "AirPressure": {
            "min": 900,
            "max": 1030,
            "optimal_min": 930,
            "optimal_max": 1025,
            "type": "range"
        },
        "RelativeHumidity": {
            "max": 95,
            "type": "lower_better"
        },
        "Precipitation": {
            "max": 3000,
            "type": "lower_better"
        },
        "Elevation": {
            "min": 100,
            "max": 2000,
            "type": "range"
        },
        "Slope": {
            "max": 20,
            "type": "lower_better"
        },
        "TurbulenceIntensity": {
            "max": 25,
            "type": "lower_better"
        }
    }

    score = 0
    suggestions = {}
    total_params = 0

    for param, config in thresholds.items():
        matching_cols = [c for c in df_site.columns if param in c]
        
        for col in matching_cols:
            total_params += 1
            value = float(df_site[col].iloc[0])
            
            param_type = config.get("type")
            
            if param_type == "range":
                min_val = config.get("min", float('-inf'))
                max_val = config.get("max", float('inf'))
                
                if min_val <= value <= max_val:
                    if "optimal_min" in config and "optimal_max" in config:
                        if config["optimal_min"] <= value <= config["optimal_max"]:
                            score += 1
                        else:
                            score += 0.7
                    else:
                        score += 1
                else:
                    if value < min_val:
                        suggestions[col] = f" {param} is too low: {value:.2f} (minimum: {min_val})"
                    else:
                        suggestions[col] = f" {param} is too high: {value:.2f} (maximum: {max_val})"
            
            elif param_type == "higher_better":
                min_val = config.get("min", 0)
                optimal_val = config.get("optimal", min_val)
                
                if value >= optimal_val:
                    score += 1
                elif value >= min_val:
                    score += 0.7
                else:
                    suggestions[col] = f" {param} below minimum: {value:.2f} m/s (need ≥{min_val} m/s)"
            
            elif param_type == "lower_better":
                max_val = config.get("max", float('inf'))
                
                if value <= max_val * 0.7:
                    score += 1
                elif value <= max_val:
                    score += 0.7
                else:
                    suggestions[col] = f" {param} too high: {value:.2f} (maximum: {max_val})"

    suitability_percent = round((score / total_params) * 100, 2) if total_params > 0 else 0

    print(f"\n   📊 Scoring complete: {score}/{total_params} = {suitability_percent}%")
    print(f"   🎯 Suggestions: {len(suggestions)} issues found\n")

    return label, suitability_percent, suggestions