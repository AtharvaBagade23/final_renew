
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------
# 1. Load Dataset
# -------------------------
DATA_PATH = "synthetic_wind_dataset_2000.csv"  # Path to the dataset
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
# 8. Wind Suitability Scoring Function
# -------------------------
def wind_suitability_score(site_dict):
    """
    Input:
        site_dict (dict): Dictionary of site parameters matching dataset columns.
    Output:
        (label, score_percent, suggestions)
    """
    df_site = pd.DataFrame([site_dict])
    
    # Align columns with training features
    missing_cols = [col for col in X.columns if col not in df_site.columns]
    for col in missing_cols:
        df_site[col] = X[col].mean()
    df_site = df_site[X.columns]

    # Predict feasibility
    prediction = rf.predict(df_site)[0]
    label = "Yes" if prediction == 1 else "No"

    # Thresholds for scoring suggestions
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

    for param, threshold in thresholds.items():
        # Find the corresponding column in site_dict
        matching_cols = [c for c in df_site.columns if param in c]
        for col in matching_cols:
            value = float(df_site[col].iloc[0])
            if isinstance(threshold, tuple):
                low, high = threshold
                if low <= value <= high:
                    score += 1
                else:
                    if value < low:
                        suggestions[col] = f"Increase {col} by {round(low - value, 2)}"
                    else:
                        suggestions[col] = f"Reduce {col} by {round(value - high, 2)}"
            else:
                if param in ["RelativeHumidity","Precipitation","Slope","Elevation","TurbulenceIntensity"]:
                    if value < threshold:
                        score += 1
                    else:
                        suggestions[col] = f"Reduce {col} by {round(value - threshold, 2)}"
                else:
                    if value >= threshold:
                        score += 1
                    else:
                        suggestions[col] = f"Increase {col} by {round(threshold - value, 2)}"

    total_params = sum([len([c for c in df_site.columns if param in c]) for param in thresholds])
    suitability_percent = round((score / total_params) * 100, 2)

    return label, suitability_percent, suggestions

# -------------------------
# 9. Main Analysis Function
# -------------------------
def analyze_wind_site(site_data):
    """
    Analyze a wind site's feasibility
    Args:
        site_data (dict): Dictionary containing wind site parameters
    Returns:
        tuple: (label, score, suggestions)
    """
    label, score, suggestions = wind_suitability_score(site_data)

    # Determine site category based on score
    if score >= 75:
        category = "Excellent"
    elif score >= 60:
        category = "Good"
    elif score >= 40:
        category = "Marginal"
    else:
        category = "Poor"

    print("\n======== WIND SITE ANALYSIS RESULT ========")
    print(f"Predicted Feasibility: {label}")
    print(f"Wind Suitability Score: {score}%")
    print(f"Site Category: {category}")

    if label == "Yes":
        print("\nNote: The site is predicted to be feasible based on historical data patterns,")
        if score < 50:
            print("but has a low suitability score. Consider addressing the suggestions below for optimal performance.")
    else:
        print("\nNote: The site is predicted to be unfeasible based on historical data patterns.")

    if suggestions:
        print("\nParameter Improvement Suggestions:")
        for p, s in suggestions.items():
            print("-", s)
    else:
        print("\n✅ This site meets all optimal conditions for wind plant setup.")

    return label, score, suggestions

if __name__ == "__main__":
    # Example site for standalone testing
    sample_wind_site = {
        "Slope": 17.051106539303152,
        "Elevation": 1230.857456194914,
        "TurbulenceIntensity": 17.884531822173724,
        "Q1-AirTemperature": 26.87,
        # ... (rest of the sample data)
    }
    analyze_wind_site(sample_wind_site)
