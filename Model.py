import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv('crop_production.csv')
df_model = df.dropna()

# Encode categorical columns
le_crop = LabelEncoder()
le_state = LabelEncoder()
df_model['Crop'] = le_crop.fit_transform(df_model['Crop'])
df_model['State'] = le_state.fit_transform(df_model['State'])

# Features and Target
X = df_model[['Crop', 'State', 'Cost of Cultivation (`/Hectare) A2+FL']]
y = df_model['Yield (Quintal/ Hectare)']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = XGBRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Save files
joblib.dump(model, 'model.pkl')
joblib.dump(le_crop, 'le_crop.pkl')
joblib.dump(le_state, 'le_state.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model trained and saved successfully!")
