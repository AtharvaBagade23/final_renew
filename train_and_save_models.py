"""
Train and save solar and wind RandomForest models as .pkl files
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Solar Model
solar_df = pd.read_csv("./Datasets/Solar_Sites_Dataset_India.csv")
solar_label_col = "Label (Yes/No)"
solar_df[solar_label_col] = solar_df[solar_label_col].map({"Yes": 1, "No": 0})
solar_X = solar_df.drop(solar_label_col, axis=1)
solar_y = solar_df[solar_label_col]
solar_X_train, _, solar_y_train, _ = train_test_split(solar_X, solar_y, test_size=0.4, random_state=42, stratify=solar_y)
solar_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
solar_rf.fit(solar_X_train, solar_y_train)
joblib.dump((solar_rf, list(solar_X.columns)), "solar_model.pkl")
print("Saved solar_model.pkl")

# Wind Model
wind_df = pd.read_csv("./Datasets/Wind_Sites_Dataset_India.csv")
wind_label_col = "Label"
wind_df[wind_label_col] = wind_df[wind_label_col].map({"Yes": 1, "No": 0})
wind_X = wind_df.drop(wind_label_col, axis=1)
wind_y = wind_df[wind_label_col]
wind_X_train, _, wind_y_train, _ = train_test_split(wind_X, wind_y, test_size=0.4, random_state=42, stratify=wind_y)
wind_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
wind_rf.fit(wind_X_train, wind_y_train)
joblib.dump((wind_rf, list(wind_X.columns)), "wind_model.pkl")
print("Saved wind_model.pkl")
