import numpy as np
from sklearn.ensemble import IsolationForest

# Train model khi import
data = np.random.rand(1000, 3)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data)

def detect_fraud(amount, frequency, location):

    sample = [[amount, frequency, location]]

    prediction = model.predict(sample)

    if prediction[0] == -1:
        return "⚠️ Suspicious Transaction"
    else:
        return "✅ Normal Transaction"
