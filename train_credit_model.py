import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("credit_risk_dataset_10000.csv")

# Encode loan purpose
le = LabelEncoder()
df["loan_purpose"] = le.fit_transform(df["loan_purpose"])

# Features / Label
X = df.drop("default_risk", axis=1)
y = df["default_risk"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Model accuracy:", accuracy)

# Save model
pickle.dump(model, open("credit_risk_model.pkl", "wb"))

print("Model saved successfully")
