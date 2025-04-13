import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from pymongo import MongoClient  # MongoDB import

app = Flask(__name__)
CORS(app)

# MongoDB connection setup (replace with your actual details)
client = MongoClient('mongodb://localhost:27017/reethu')
db = client['crop_yield_db']
collection = db['predictions']

# Load the dataset
data = pd.read_excel("C:/Users/a6020/OneDrive/Desktop/reethu/Dataset.xlsx")  # Ensure path is correct

# Rename columns for consistency
data = data.rename(columns={
    "State_Name": "State",
    "Crop_Type": "Season",
    "Crop": "Crop",
    "rainfall": "Rainfall",
    "temperature": "Temperature",
    "N": "N",
    "P": "P",
    "K": "K",
    "pH": "pH",
    "Area_in_hectares": "Area",
    "Production_in_tons": "Production",
    "Yield_ton_per_hec": "Yield"
})

# Remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['Rainfall', 'Temperature', 'pH', 'Area', 'N', 'P', 'K', 'Production', 'Yield']:
    data = remove_outliers(data, col)

# Features and targets
X = data.drop(columns=['Yield', 'Production', 'N', 'K', 'P'])
y = data['Yield']

# Preprocessing setup
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', numerical_imputer), ('scaler', scaler)]), numerical_columns),
        ('cat', Pipeline(steps=[('imputer', categorical_imputer), ('encoder', one_hot_encoder)]), categorical_columns)
    ]
)

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)

# Train model
model_yield = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_yield.fit(X_train_processed, y_train)

# Save the model
joblib.dump(model_yield, 'crop_yield_model.pkl')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    soil_quality = request.form.get('pH', '')
    rainfall = request.form.get('Rainfall', '')
    temperature = request.form.get('Temperature', '')
    area = request.form.get('Area', '')
    crop_type = request.form.get('Season', '')
    place = request.form.get('State', '')
    crop = request.form.get('Crop', '')

    # Handle missing values with defaults
    def try_float(val, default):
        try:
            return float(val)
        except:
            return default

    soil_quality = try_float(soil_quality, data['pH'].mean())
    rainfall = try_float(rainfall, data['Rainfall'].mean())
    temperature = try_float(temperature, data['Temperature'].mean())
    area = try_float(area, data['Area'].mean())
    crop_type = crop_type or data['Season'].mode()[0]
    place = place or data['State'].mode()[0]
    crop = crop or data['Crop'].mode()[0]

    # Prepare input
    features = [crop_type, place, crop, soil_quality, rainfall, temperature, area]
    user_input_df = pd.DataFrame([features], columns=X.columns)
    user_input_processed = preprocessor.transform(user_input_df)

    # Prediction
    predicted_yield = model_yield.predict(user_input_processed)
    predicted_yield_rounded = round(predicted_yield[0], 2)

    # Store in MongoDB
    record = {
        'State': place,
        'Season': crop_type,
        'Crop': crop,
        'pH': soil_quality,
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Area': area,
        'Predicted_Yield': predicted_yield_rounded
    }
    collection.insert_one(record)

    return render_template('index.html', predicted_yield=predicted_yield_rounded)

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(_name_)
CORS(app)

data = pd.read_excel("C:/Users/REETHU/flask/Dataset.xlsx")

data = data.rename(columns={
    "State_Name": "State",
    'Crop_Type': 'Season',
    'Crop': 'Crop',
    'rainfall': 'Rainfall',
    'temperature': 'Temperature',
    'N': 'N', 'P': 'P', 'K': 'K', 'pH': 'pH',
    'Area_in_hectares': 'Area',
    'Production_in_tons': 'Production',
    'Yield_ton_per_hec': 'Yield'
})

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal for numerical features
for col in ['Rainfall', 'Temperature', 'pH', 'Area', 'N', 'P', 'K', 'Production', 'Yield']:
    data = remove_outliers(data, col)

X = data.drop(columns=['Yield', 'Production', 'N', 'K', 'P'])
y = data['Yield']
z = data['Production']

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Encoding Categorical Variables
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

# Feature Scaling
scaler = StandardScaler()

# Combine Preprocessing Steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', numerical_imputer), ('scaler', scaler)]), numerical_columns),
        ('cat', Pipeline(steps=[('imputer', categorical_imputer), ('encoder', one_hot_encoder)]), categorical_columns)
    ]
)

# Split the Dataset
X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.2, random_state=42)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train the Gradient Boosting models
model_yield = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_production = GradientBoostingRegressor(n_estimators=100, random_state=42)

model_yield.fit(X_train_processed, y_train)

# Save the models for later use in the Flask app
joblib.dump(model_yield, 'crop_yield_model.pkl')

# Serve the HTML form (index.html should be present in the templates folder)
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for predictions (using form submission)
@app.route('/predict', methods=['POST'])
def predict():
    soil_quality = request.form['pH'].strip() if 'pH' in request.form else ''
    rainfall = request.form['Rainfall'].strip() if 'Rainfall' in request.form else ''
    temperature = request.form['Temperature'].strip() if 'Temperature' in request.form else ''
    area = request.form['Area'].strip() if 'Area' in request.form else ''
    crop_type = request.form['Season'].strip() if 'Season' in request.form else ''
    place = request.form['State'].strip() if 'State' in request.form else ''
    crop = request.form['Crop'].strip() if 'Crop' in request.form else ''

    if soil_quality == "":
        soil_quality = data['pH'].mean()  
    else:
        try:
            soil_quality = float(soil_quality)
        except ValueError:
            soil_quality = data['pH'].mean()  

    if rainfall == "":
        rainfall = data['Rainfall'].mean()  
    else:
        try:
            rainfall = float(rainfall)
        except ValueError:
            rainfall = data['Rainfall'].mean()  

    if temperature == "":
        temperature = data['Temperature'].mean()  
    else:
        try:
            temperature = float(temperature)
        except ValueError:
            temperature = data['Temperature'].mean()  

    if area == "":
        area = data['Area'].mean() 
    else:
        try:
            area = float(area)
        except ValueError:
            area = data['Area'].mean()  

    if crop_type == "":
        crop_type = data['Season'].mode()[0]  

    if place == "":
        place = data['State'].mode()[0] 

    if crop == "":
        crop = data['Crop'].mode()[0] 

    features = [crop_type, place, crop, soil_quality, rainfall, temperature, area]

    user_input_df = pd.DataFrame([features], columns=X.columns)

    user_input_processed = preprocessor.transform(user_input_df)

    predicted_yield = model_yield.predict(user_input_processed)

    predicted_yield_rounded = round(predicted_yield[0], 2)
    
    return render_template('index.html', predicted_yield=predicted_yield_rounded)
if _name_ == '_main_':
    app.run(debug=True)