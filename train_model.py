import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("patient_data.csv")

print("Dataset Columns:", data.columns)

# Rename column
data = data.rename(columns={"C": "Gender"})

# Remove duplicates
data = data.drop_duplicates()

# -----------------------------
# Encode all categorical columns
# -----------------------------

encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = encoder.fit_transform(data[column])

# -----------------------------
# Separate features and target
# -----------------------------

X = data.drop("Stages", axis=1)
y = data["Stages"]

# -----------------------------
# Feature scaling
# -----------------------------

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train model
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# Model accuracy
# -----------------------------

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# -----------------------------
# Save model
# -----------------------------

joblib.dump(model, "logreg_model.pkl")

print("Model trained and saved successfully")