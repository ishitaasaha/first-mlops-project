# Import pandas for reading and manipulating the dataset
import pandas as pd

# Import train_test_split to divide the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import RandomForestClassifier for building the machine learning model
from sklearn.ensemble import RandomForestClassifier

# Import joblib to save the trained model to a file
import joblib


# Load the diabetes dataset from a hosted GitHub source
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(url)

# Print the column names to verify that the dataset was loaded correctly
print("✅ Columns:", df.columns.tolist())


# -------------------------------
# Prepare the input and output data
# -------------------------------

# Select the features that will be used to train the model
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]

# Select the target column that the model needs to predict
# Outcome: 0 = No diabetes, 1 = Diabetes
y = df["Outcome"]


# -------------------------------
# Split the dataset
# -------------------------------

# Split the data into training and testing sets
# 80% of the data is used for training
# 20% is used for testing
# random_state=42 ensures the same split each time
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# Train the machine learning model
# -------------------------------

# Create a Random Forest classification model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(X_train, y_train)


# -------------------------------
# Save the trained model
# -------------------------------

# Save the trained model as a .pkl file
# This file can later be loaded without retraining the model
joblib.dump(model, "diabetes_model.pkl")

# Confirm that the model was successfully saved
print("✅ Model saved as diabetes_model.pkl")