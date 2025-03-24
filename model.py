import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv('TourismAdvisorDataset.csv')

# Preprocessing
df['Destination'] = df['Destination'].str.strip().str.title()

# Imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['Type of Trip', 'Mode of Transport', 'Destination']] = cat_imputer.fit_transform(
    df[['Type of Trip', 'Mode of Transport', 'Destination']]
)

num_imputer = SimpleImputer(strategy='mean')
df['Budget'] = num_imputer.fit_transform(df[['Budget']])

# Encoding
trip_encoder = LabelEncoder()
transport_encoder = LabelEncoder()
destination_encoder = LabelEncoder()

# Update trip types and transport modes
valid_trip_types = ['Adventure', 'Relaxation', 'Cultural', 'Honeymoon', 'Historical', 'Hill Station', 'Beach', 'Nature']
valid_transport_modes = ['Road', 'Flight']

# Filter dataset to include only valid trip types and transport modes
df = df[df['Type of Trip'].isin(valid_trip_types) & df['Mode of Transport'].isin(valid_transport_modes)]

# Encode features
df['Type of Trip'] = trip_encoder.fit_transform(df['Type of Trip'])
df['Mode of Transport'] = transport_encoder.fit_transform(df['Mode of Transport'])
df['Destination'] = destination_encoder.fit_transform(df['Destination'])

# Feature selection
X = df[['Budget', 'Type of Trip', 'Mode of Transport']]
y = df['Destination']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5)
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
ensemble = VotingClassifier(estimators=[('rf', rf), ('ada', ada)], voting='hard')
ensemble.fit(X_train, y_train)

# Save artifacts
joblib.dump(trip_encoder, 'trip_encoder.pkl')
joblib.dump(transport_encoder, 'transport_encoder.pkl')
joblib.dump(destination_encoder, 'destination_encoder.pkl')
joblib.dump(ensemble, 'model.pkl')