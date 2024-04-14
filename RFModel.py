import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from joblib import dump
from math import sqrt

# load and print dataset
url = "https://raw.githubusercontent.com/Mateo486/Housing-Market-Capstone/main/miami-housing%5B1%5D.csv"
print("Loading data...")
data = pd.read_csv(url)

# prepare data features and target variable
X = data[['LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST', 'age', 'structure_quality']]
y = data['SALE_PRC']

# split dataset into training and test sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# setup and configure model pipeline
print("Setting up the model pipeline...")
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # handle missing values
    ('regressor', RandomForestRegressor(random_state=42))  # regression model
])

# fit model to training data
print("Fitting the model...")
pipeline.fit(X_train, y_train)

# setup grid search for hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 120],  # define number of trees
    'regressor__max_depth': [None, 7],  # define maximum depth of trees
    'regressor__min_samples_split': [2, 5],  # define minimum samples to split a node
    'regressor__min_samples_leaf': [1, 2]  # define minimum samples at each leaf node
}

# execute grid search to find best model parameters
print("Starting grid search...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# retrieve best model from grid search
best_model = grid_search.best_estimator_
print("Grid search complete.")

# save best model to file
model_path = 'C:\\Users\\mnm4m\\project\\best_model3.joblib'
print(f"Saving the model to {model_path}...")
dump(best_model, model_path)
print("Model saved.")
