import joblib
from preprocess import preprocess_data

from utils import load_standard_data, load_biased_data

# Load your saved model
modelReg = joblib.load('models/regression_model.joblib')
modelClf = joblib.load('models/classification_model.joblib')


new_df = load_standard_data()

df_100 = new_df.head(100)
df_100c = new_df.tail(100)


# Prepare new data the same way you prepared training data

# Make predictions
predictions = modelReg.predict(df_100)
print(predictions)
df_100["Total_score"] = (df_100["Midterm_Score"] + df_100["Final_Score"] + df_100["Projects_Score"]) / 3
print(df_100["Total_score"])

