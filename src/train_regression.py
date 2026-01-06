import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_biased_data, load_standard_data
from preprocess import preprocess_data

df = load_standard_data()

df["Total_score"] = (df["Midterm_Score"] + df["Final_Score"] + df["Projects_Score"]) / 3
y = df["Total_score"]

preprocessor, X = preprocess_data(df, target_cols= ["Total_score"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

joblib.dump(model, 'models/regression_model.joblib')