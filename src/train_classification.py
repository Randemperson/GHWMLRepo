import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils import load_biased_data, load_standard_data
from preprocess import preprocess_data

df = load_standard_data()

df["Total_score"] = (df["Midterm_Score"] + df["Final_Score"] + df["Projects_Score"]) / 3

def assign_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

df['grade'] = df['Total_score'].apply(assign_grade)
y = df['grade']

preprocessor, X = preprocess_data(df, target_cols= ['grade'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
joblib.dump(model, 'models/classification_model.joblib')