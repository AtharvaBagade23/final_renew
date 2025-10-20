
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------
# 2. Load Dataset
# -------------------------
DATA_PATH = "solar_synthetic_quarterly_yearly_2000.csv"
df = pd.read_csv(DATA_PATH)

label_col = "Label (Yes/No)"
if label_col not in df.columns:
    raise ValueError(f"Column '{label_col}' not found in dataset. Check CSV headers!")

# Encode Label
df[label_col] = df[label_col].map({"Yes": 1, "No": 0})

# -------------------------
# 2b. Add Controlled Noise (~5%) to Reduce Overfitting
# -------------------------
noise_level = 0.05  # 5% Gaussian noise
numeric_cols = df.select_dtypes(include=np.number).columns.drop(label_col)

for col in numeric_cols:
    df[col] += np.random.normal(0, noise_level * df[col].std(), size=len(df))

# Slightly flip 5% of the labels
flip_rate = 0.1
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
        
        # Feature Importance
        importances = rf.feature_importances_
        features = X.columns
        plt.figure(figsize=(10,6))
        plt.barh(features, importances, color='skyblue')
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        print("\nFeature importance chart saved as 'feature_importance.png'")

# Train model immediately with no output
train_model(verbose=False)

# If run directly, train again with output
if __name__ == "__main__":
    train_model(verbose=True)

# -------------------------
# 8. Solar Suitability Scoring Function
# -------------------------
def solar_suitability_score(site_dict):
    df_site = pd.DataFrame([site_dict])
    missing_cols = [col for col in X.columns if col not in df_site.columns]
    for col in missing_cols:
        df_site[col] = X[col].mean()
    df_site = df_site[X.columns]

    prediction = rf.predict(df_site)[0]
    label = "Yes" if prediction == 1 else "No"

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
                    suggestions[param] = f"Increase {param} by {round(low - value, 2)}"
                else:
                    suggestions[param] = f"Reduce {param} by {round(value - high, 2)}"
        else:
            if param in ["Relative humidity", "Precipitation", "Snowfall (mm/year)", "YearlyCloud cover"]:
                if value < threshold:
                    score += 1
                else:
                    suggestions[param] = f"Reduce {param} by {round(value - threshold, 2)}"
            else:
                if value >= threshold:
                    score += 1
                else:
                    suggestions[param] = f"Increase {param} by {round(threshold - value, 2)}"

    total_params = len(thresholds)
    suitability_percent = round((score / total_params) * 100, 2)

    return label, suitability_percent, suggestions

# -------------------------
# 9. Main Analysis Function
# -------------------------
def analyze_solar_site(site_data):
    """
    Analyze a solar site's feasibility
    Args:
        site_data (dict): Dictionary containing solar site parameters
    Returns:
        tuple: (label, score, suggestions)
    """
    label, score, suggestions = solar_suitability_score(site_data)
    
    # Determine site category based on score
    if score >= 75:
        category = "Excellent"
    elif score >= 60:
        category = "Good"
    elif score >= 40:
        category = "Marginal"
    else:
        category = "Poor"
    
    print("\n======== SOLAR SITE ANALYSIS RESULT ========")
    print(f"Predicted Feasibility: {label}")
    print(f"Solar Suitability Score: {score}%")
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
        print("\n✅ This site meets all optimal conditions for solar plant setup.")
    
    return label, score, suggestions