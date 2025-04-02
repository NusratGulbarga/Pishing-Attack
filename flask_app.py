import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ Load dataset
df = pd.read_csv("phishing.csv")

# ✅ Standardize column names
df.columns = df.columns.str.strip().str.lower()

# ✅ Ensure 'URL' column exists and is lowercase
df.rename(columns={"url": "url"}, inplace=True)

# ✅ Feature Engineering
df['url_length'] = df['url'].apply(len)
df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith('https') else 0)

# ✅ Select relevant features (ensure they exist)
existing_features = [col for col in df.columns if "feature" in col]
features = existing_features + ['url_length', 'num_dots', 'has_https']

# ✅ Ensure target column exists
if 'is_phishing' not in df.columns:
    raise ValueError("Dataset must contain 'is_phishing' column.")

X = df[features]
y = df['is_phishing']

# ✅ Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "phishing_model.pkl")

# ✅ Flask App
app = Flask(__name__)
model = joblib.load("phishing_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400

    # ✅ Feature Extraction for Prediction
    test_data = {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "has_https": int(url.startswith("https"))
    }

    # ✅ Ensure all model features exist in test data
    for feature in features:
        if feature not in test_data:
            test_data[feature] = 0  # Assign default value

    df_input = pd.DataFrame([test_data])

    # ✅ Ensure feature order matches model training
    df_input = df_input[features]

    prediction = model.predict(df_input)[0]
    
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)