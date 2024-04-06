from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('Mall_Customers_with_Zone.csv')

# Preprocess gender column (convert to numerical)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Convert gender to numerical

# Select features (age, gender, annual income) and target variable (spending score)
X_score = df[['Age', 'Gender', 'Annual Income (k$)']]
y_score = df['Spending Score (1-100)']

# Select features (age, gender, annual income, spending score) and target variable (zone)
X_zone = df[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']]
y_zone = df['Zone']

# Initialize and train the Random Forest regression model for spending score prediction
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_score, y_score)

# Initialize and train the Random Forest classifier model for zone prediction
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_zone, y_zone)

# Get the range of input values from the dataset
age_range = df['Age'].min(), df['Age'].max()
income_range = df['Annual Income (k$)'].min(), df['Annual Income (k$)'].max()

@app.route('/')
def index():
    return render_template('index.html', predicted_score=None, predicted_zone=None, age_range=age_range, income_range=income_range)

@app.route('/predicted_score', methods=['POST'])
def predicted_score_endpoint():
    age = int(request.form['age'])
    gender = request.form['gender']
    annual_income = float(request.form['income'])

    # Check if input values are within the range of the dataset
    if age < age_range[0] or age > age_range[1] or annual_income < income_range[0] or annual_income > income_range[1] or age < 0 or annual_income < 0:
        return render_template('index.html', predicted_score="Input values out of range", predicted_zone=None, age_range=age_range, income_range=income_range)

    # Encode gender input
    gender_encoded = label_encoder.transform([gender])[0]

    # Predict spending score for the input data using the Random Forest model
    predicted_score = rf_regressor.predict([[age, gender_encoded, annual_income]])
    return render_template('index.html', predicted_score=predicted_score[0], predicted_zone=None, age_range=age_range, income_range=income_range)

@app.route('/predicted_zone', methods=['POST'])
def predicted_zone_endpoint():
    age = int(request.form['age_zone'])
    gender = request.form['gender_zone']
    annual_income = float(request.form['income_zone'])
    spending_score = float(request.form['spending_score_zone'])

    # Check if input values are within the range of the dataset
    if age < age_range[0] or age > age_range[1] or annual_income < income_range[0] or annual_income > income_range[1] or age < 0 or annual_income < 0:
        return render_template('index.html', predicted_score=None, predicted_zone="Input values out of range", age_range=age_range, income_range=income_range)

    # Encode gender input
    gender_encoded = label_encoder.transform([gender])[0]

    # Predict zone for the input data using the Random Forest model
    predicted_zone = rf_classifier.predict([[age, gender_encoded, annual_income, spending_score]])
    return render_template('index.html', predicted_score=None, predicted_zone=predicted_zone[0], age_range=age_range, income_range=income_range)

if __name__ == '__main__':
    app.run(debug=True, port=8000)

