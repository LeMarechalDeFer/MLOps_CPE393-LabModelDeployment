# save_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np


iris = load_iris()
X, y = iris.data, iris.target

iris_model = RandomForestClassifier(random_state=42, n_jobs=-1)
iris_model.fit(X, y)

with open("app/iris_model.pkl", "wb") as f:
    pickle.dump(iris_model, f)


# price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus
# price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus

# Continuous numeric: price, area
# Discrete numeric: bedrooms, bathrooms, stories, parking
# Boolean (categorical): mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea
# Ordinal (categorical): furnishingstatus


# Loading and binning for stratification
df = pd.read_csv("Housing.csv")
X = df.drop("price", axis=1)
y = df["price"]
y_binned = pd.qcut(y, q=5, labels=False)

# Binarization of yes/no
bool_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
X[bool_cols] = X[bool_cols].apply(lambda col: col.str.lower().map({"yes":1,"no":0}))

# New features
X["area_per_bed"] = X["area"] / X["bedrooms"].replace(0,1)
X["rooms_total"]  = X["bedrooms"] + X["bathrooms"]

# Column definitions
categorical_features = ["furnishingstatus"]
numerical_features = ["area","bedrooms","bathrooms","stories","parking",
                      "area_per_bed","rooms_total"] + bool_cols

# Ordinal preprocessor + passthrough
preprocessor = ColumnTransformer([
    ("ord", OrdinalEncoder(categories=[["unfurnished","semi-furnished","furnished"]]),
     ["furnishingstatus"]),
    ("pass", "passthrough", numerical_features)
])

# Model with log-target
ttr = TransformedTargetRegressor(
    regressor=RandomForestRegressor(random_state=42, n_jobs=-1),
    func=np.log1p, inverse_func=np.expm1
)
pipeline = Pipeline([("pre", preprocessor), ("reg", ttr)])

# Stratified and shuffled split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_binned
)

# Hyperparameter search (i did xboost before)
param_dist = {
    "reg__regressor__n_estimators": [100,300,500],
    "reg__regressor__max_depth": [None,10,20],
    "reg__regressor__min_samples_leaf": [1,2,5]
}
search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=12,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Optimal params:", search.best_params_)

preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds)) 
print("RMSE test:", rmse)
print("R² test:", r2_score(y_test, preds))

# Cross-validation on the entire sample
cv_scores = cross_val_score(best_model, X, y,
                            cv=5,
                            scoring="neg_root_mean_squared_error",
                            n_jobs=-1)
print("CV RMSE :", -cv_scores.mean(), "±", cv_scores.std())

# Save
import pickle
with open("app/housing_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
