import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from joblib import dump
from math import sqrt

# Load the dataset
url = "https://raw.githubusercontent.com/Mateo486/Housing-Market-Capstone/main/miami-housing%5B1%5D.csv"
data = pd.read_csv(url)

# Prepare the features and target
X = data[['LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST', 'age', 'structure_quality']]
y = data['SALE_PRC']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setup the pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Perform grid search for hyperparameter tuning
param_grid = {
    'regressor__max_depth': [2, 3, 4],
    'regressor__max_features': [2, 3, 4, 5, 6, 'sqrt', 'log2', None],
    'regressor__min_samples_split': [2, 5, 10, 30]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Save the best model to a file in the specified directory
model_path = 'C:\\Users\\mnm4m\\project\\best_model.joblib'
dump(best_model, model_path)
