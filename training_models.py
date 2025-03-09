import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset (replace with your dataset file name)
data = pd.read_csv('Phishing_validation_emails.csv')

# Ensure correct column names
X = data['Email Text']  # Email content
y = data['Email Type'].map({'trap': 1, 'trusted': 0})  # Convert labels to 0 and 1
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
y = data['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0})  # Convert labels
print(y.isnull().sum())  # Check again for NaN values
print(data.isnull().sum())  # Check for NaN values
data = data.dropna(subset=['Email Type'])  # Remove missing rows
# OR
data['Email Type'] = data['Email Type'].fillna('Safe Email')  # Fix inplace warning


# Vectorize email content
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vectorized, y)

# Save model and vectorizer
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model training complete. Files saved successfully.")
