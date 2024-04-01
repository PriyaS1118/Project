#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PriyaS1118/Project/blob/main/Untitled5.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated maintenance data
maintenance_data = {
    'timestamp': [datetime(2023, 11, 1, 8, 0), datetime(2023, 11, 2, 9, 0), datetime(2023, 11, 5, 10, 0)],
    'equipment_id': [1, 2, 1],
    'maintenance_type': ['Scheduled', 'Unscheduled', 'Scheduled']
}

# Simulated sensor data
sensor_data = {
    'timestamp': [datetime(2023, 11, 1, 8, 0), datetime(2023, 11, 1, 9, 0), datetime(2023, 11, 1, 10, 0),
                  datetime(2023, 11, 2, 9, 0), datetime(2023, 11, 2, 10, 0), datetime(2023, 11, 3, 8, 0)],
    'equipment_id': [1, 1, 1, 2, 2, 2],
    'sensor_value': [48, 60, 58, 45, 48, 42]
}

# Create DataFrames
maintenance_df = pd.DataFrame(maintenance_data)
sensor_df = pd.DataFrame(sensor_data)

# Merge data
merged_df = pd.merge(sensor_df, maintenance_df, on=['equipment_id', 'timestamp'], how='left')
merged_df['maintenance_type'].fillna('No Maintenance', inplace=True)

# Feature engineering
merged_df['hour'] = merged_df['timestamp'].dt.hour
merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Define target variable: 1 for maintenance, 0 for no maintenance
merged_df['target'] = merged_df['maintenance_type'].apply(lambda x: 1 if x != 'No Maintenance' else 0)

# Features and target
features = ['sensor_value', 'hour', 'day_of_week', 'is_weekend']
X = merged_df[features]
y = merged_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict maintenance events
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Use the trained model to predict maintenance events in real-time
# For example, you can use the latest sensor data to make predictions.
latest_sensor_data = pd.DataFrame({'sensor_value': [48], 'hour': [11], 'day_of_week': [1], 'is_weekend': [0]})
prediction = clf.predict(latest_sensor_data)
if prediction == 1:
    print('Predictive maintenance is needed.')
else:
    print('No maintenance needed.')


# In[ ]:


from datetime import datetime, timedelta
import pandas as pd

# Simulated sensor data
sensor_data = {
    'timestamp': [datetime(2023, 11, 1, 8, 0), datetime(2023, 11, 1, 9, 0), datetime(2023, 11, 1, 10, 0),
                  datetime(2023, 11, 2, 9, 0), datetime(2023, 11, 2, 10, 0), datetime(2023, 11, 3, 8, 0)],
    'equipment_id': [1, 2, 1, 6, 3, 1],
    'sensor_value': [55, 45, 58, 45, 48, 42]
}

# Create a DataFrame for sensor data
sensor_df = pd.DataFrame(sensor_data)

# Feature engineering
sensor_df['hour'] = sensor_df['timestamp'].dt.hour
sensor_df['day_of_week'] = sensor_df['timestamp'].dt.dayofweek
sensor_df['is_weekend'] = sensor_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Define a function to send an alert for sensors that need maintenance
def send_maintenance_alert(sensor_df):
    for index, row in sensor_df.iterrows():
        if row['sensor_value'] < 50:
            # Send an alert (you can replace this with your alert mechanism)
            print(f"ALERT: Sensor {row['equipment_id']} needs maintenance.")

# Check for sensors needing maintenance and send an alert
send_maintenance_alert(sensor_df)


# In[ ]:


from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated maintenance data
maintenance_data = {
    'timestamp': [datetime(2023, 11, 1, 8, 0), datetime(2023, 11, 2, 9, 0), datetime(2023, 11, 5, 10, 0)],
    'equipment_id': [1, 2, 1],
    'maintenance_type': ['Scheduled', 'Unscheduled', 'Scheduled']
}

# Simulated sensor data
sensor_data = {
    'timestamp': [datetime(2023, 11, 1, 8, 0), datetime(2023, 11, 1, 9, 0), datetime(2023, 11, 1, 10, 0),
                  datetime(2023, 11, 2, 9, 0), datetime(2023, 11, 2, 10, 0), datetime(2023, 11, 3, 8, 0)],
    'equipment_id': [1, 1, 1, 2, 2, 2],
    'sensor_value': [55, 60, 58, 45, 48, 42]
}

# Create DataFrames
maintenance_df = pd.DataFrame(maintenance_data)
sensor_df = pd.DataFrame(sensor_data)

# Merge data
merged_df = pd.merge(sensor_df, maintenance_df, on=['equipment_id', 'timestamp'], how='left')
merged_df['maintenance_type'].fillna('No Maintenance', inplace=True)

# Feature engineering
merged_df['hour'] = merged_df['timestamp'].dt.hour
merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Define target variable: 1 for maintenance, 0 for no maintenance
merged_df['target'] = merged_df['maintenance_type'].apply(lambda x: 1 if x != 'No Maintenance' else 0)

# Features and target
features = ['sensor_value', 'hour', 'day_of_week', 'is_weekend']
X = merged_df[features]
y = merged_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict maintenance events
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Use the trained model to predict maintenance events in real-time
# For example, you can use the latest sensor data to make predictions.
latest_sensor_data = pd.DataFrame({'sensor_value': [52], 'hour': [11], 'day_of_week': [1], 'is_weekend': [0]})
prediction = clf.predict(latest_sensor_data)

if prediction == 1:
    print('Maintenance needed: Predictive maintenance is required.')
else:
    print('No maintenance needed: The equipment is in good condition.')



# In[ ]:


import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import random

# Function to generate random data
def generate_random_data(seed):
    random.seed(seed)

    # Simulated maintenance data
    maintenance_data = {
        'timestamp': [datetime(2023, 11, random.randint(1, 30), random.randint(0, 23), random.randint(0, 59)) for _ in range(200)],
        'equipment_id': [random.choice([1, 2]) for _ in range(200)],
        'maintenance_type': [random.choice(['Scheduled', 'Unscheduled', 'No Maintenance']) for _ in range(200)]
    }

    # Simulated sensor data
    sensor_data = {
        'timestamp': [datetime(2023, 11, random.randint(1, 30), random.randint(0, 23), random.randint(0, 59)) for _ in range(200)],
        'equipment_id': [random.choice([1, 2]) for _ in range(200)],
        'sensor_value': [random.randint(40, 80) for _ in range(200)]
    }

    # Create DataFrames
    maintenance_df = pd.DataFrame(maintenance_data)
    sensor_df = pd.DataFrame(sensor_data)

    # Merge data
    merged_df = pd.merge(sensor_df, maintenance_df, on=['equipment_id', 'timestamp'], how='left')
    merged_df['maintenance_type'].fillna('No Maintenance', inplace=True)

    # Feature engineering
    merged_df['hour'] = merged_df['timestamp'].dt.hour
    merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Define target variable: 1 for maintenance, 0 for no maintenance
    merged_df['target'] = merged_df['maintenance_type'].apply(lambda x: 1 if x != 'No Maintenance' else 0)

    # Features and target
    features = ['sensor_value', 'hour', 'day_of_week', 'is_weekend']
    X = merged_df[features]
    y = merged_df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Initialize random seed for reproducibility
random.seed(0)

# Number of scenarios to run
num_scenarios = 5

# Lists to store accuracy results
rf_accuracies = []
xgb_accuracies = []

for scenario in range(num_scenarios):
    X_train, X_test, y_train, y_test = generate_random_data(scenario)

    # Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)

    # Train an XGBoost Classifier
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X_train, y_train)

    # Predict maintenance events using Random Forest
    rf_predictions = rf_classifier.predict(X_test)

    # Predict maintenance events using XGBoost
    xgb_predictions = xgb_classifier.predict(X_test)

    # Calculate accuracies
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    # Append accuracies to the lists
    rf_accuracies.append(rf_accuracy)
    xgb_accuracies.append(xgb_accuracy)

    print(f"Scenario {scenario + 1}:")
    print(f"Random Forest Classifier Accuracy: {rf_accuracy}")
    print(f"XGBoost Classifier Accuracy: {xgb_accuracy}")
    print(f"Accuracy Difference (Random Forest - XGBoost): {rf_accuracy - xgb_accuracy}")
    print()

# Calculate and print the average accuracy difference
avg_accuracy_difference = sum(rf_accuracies) / num_scenarios - sum(xgb_accuracies) / num_scenarios
print(f"Average Accuracy Difference (Random Forest - XGBoost): {avg_accuracy_difference}")



# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic sensor data
np.random.seed(0)
n_samples = 2000
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1000, 100, n_samples)
vibration = np.random.normal(2, 1, n_samples)
failure_indicator = np.random.randint(2, size=n_samples)

sensor_data = pd.DataFrame({'temperature': temperature, 'pressure': pressure, 'vibration': vibration, 'failure_indicator': failure_indicator})

# Prepare the data for training
condition_indicators = sensor_data[{'temperature', 'pressure', 'vibration'}]
labels = sensor_data['failure_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(condition_indicators, labels, test_size=0.25, random_state=42)

# Train a machine learning model (Linear Regression in this case)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = np.mean((y_pred > 0.5) == y_test)  # Assuming binary classification
print('Accuracy:', accuracy)

# Use the trained model to predict the failure of sensors in real time
# Collect sensor data from the sensors
new_temperature = 27.5
new_pressure = 950
new_vibration = 1.8

# Make a prediction on the new sensor data
new_sensor_data = [new_temperature, new_pressure, new_vibration]
prediction = model.predict([new_sensor_data])

# If the prediction is greater than 0.5 (assuming a binary classification threshold), then the sensor is likely to fail soon
if prediction[0] > 0.5:
    # Schedule maintenance for the sensor
    print('Sensor is likely to fail soon. Schedule maintenance.')
else:
    # The sensor is likely to be in good condition
    print('Sensor is in good condition.')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate synthetic sensor data
np.random.seed(0)
n_samples = 20000
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1000, 100, n_samples)
vibration = np.random.normal(2, 1, n_samples)
failure_indicator = np.random.randint(2, size=n_samples)

sensor_data = pd.DataFrame({'temperature': temperature, 'pressure': pressure, 'vibration': vibration, 'failure_indicator': failure_indicator})

# Prepare the data for training
condition_indicators = sensor_data[['temperature', 'pressure', 'vibration']]
labels = sensor_data['failure_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(condition_indicators, labels, test_size=0.25, random_state=42)

# Reshape input data to fit LSTM input shape (samples, time steps, features)
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

# Use the trained model to predict the failure of sensors in real time
# Collect sensor data from the sensors
new_temperature = 27.5
new_pressure = 950
new_vibration = 1.8

# Reshape the new sensor data
new_sensor_data = np.reshape(np.array([[new_temperature, new_pressure, new_vibration]]), (1, 1, 3))

# Make a prediction on the new sensor data
prediction = model.predict(new_sensor_data)

# If the prediction is greater than 0.5, then the sensor is likely to fail soon
if prediction[0][0] > 0.5:
    # Schedule maintenance for the sensor
    print('Sensor is likely to fail soon. Schedule maintenance.')
else:
    # The sensor is likely to be in good condition
    print('Sensor is in good condition.')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Generate synthetic sensor data
np.random.seed(0)
n_samples = 1000
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1000, 100, n_samples)
vibration = np.random.normal(2, 1, n_samples)
failure_indicator = np.random.randint(2, size=n_samples)

sensor_data = pd.DataFrame({'temperature': temperature, 'pressure': pressure, 'vibration': vibration, 'failure_indicator': failure_indicator})

# Prepare the data for training
condition_indicators = sensor_data[['temperature', 'pressure', 'vibration']]
labels = sensor_data['failure_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(condition_indicators, labels, test_size=0.25, random_state=42)

# Train a machine learning model (XGBoost Classifier in this case)
model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Use the trained model to predict the failure of sensors in real time
# Collect sensor data from the sensors
new_temperature = 27.5
new_pressure = 950
new_vibration = 1.8

# Make a prediction on the new sensor data
new_sensor_data = np.array([[new_temperature, new_pressure, new_vibration]])
prediction = model.predict(new_sensor_data)

# If the prediction is 1, then the sensor is likely to fail soon
if prediction[0] == 1:
    # Schedule maintenance for the sensor
    print('Sensor is likely to fail soon. Schedule maintenance.')
else:
    # The sensor is likely to be in good condition
    print('Sensor is in good condition.')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Generate synthetic sensor data
np.random.seed(0)
n_samples = 1000
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1000, 100, n_samples)
vibration = np.random.normal(2, 1, n_samples)
failure_indicator = np.random.randint(2, size=n_samples)

sensor_data = pd.DataFrame({'temperature': temperature, 'pressure': pressure, 'vibration': vibration, 'failure_indicator': failure_indicator})

# Prepare the data for training
condition_indicators = sensor_data[['temperature', 'pressure', 'vibration']]
labels = sensor_data['failure_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(condition_indicators, labels, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Support Vector Machine (SVM) model
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model performance
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Use the trained model to predict the failure of sensors in real time
# Collect sensor data from the sensors
new_temperature = 27.5
new_pressure = 950
new_vibration = 1.8

# Make a prediction on the new sensor data
new_sensor_data = scaler.transform([[new_temperature, new_pressure, new_vibration]])
prediction = model.predict(new_sensor_data)

# If the prediction is 1, then the sensor is likely to fail soon
if prediction[0] == 1:
    # Schedule maintenance for the sensor
    print('Sensor is likely to fail soon. Schedule maintenance.')
else:
    # The sensor is likely to be in good condition
    print('Sensor is in good condition.')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic sensor data
np.random.seed(0)
n_samples = 2000
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1000, 100, n_samples)
vibration = np.random.normal(2, 1, n_samples)
failure_indicator = np.random.randint(2, size=n_samples)

sensor_data = pd.DataFrame({'temperature': temperature, 'pressure': pressure, 'vibration': vibration, 'failure_indicator': failure_indicator})

# Prepare the data for training
condition_indicators = sensor_data[['temperature', 'pressure', 'vibration']]
labels = sensor_data['failure_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(condition_indicators, labels, test_size=0.25, random_state=42)

# Train a machine learning model (Random Forest Classifier in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Use the trained model to predict the failure of sensors in real time
# Collect sensor data from the sensors
new_temperature = 27.5
new_pressure = 950
new_vibration = 1.8

# Make a prediction on the new sensor data
new_sensor_data = [[new_temperature, new_pressure, new_vibration]]
prediction = model.predict(new_sensor_data)

# If the prediction is 1, then the sensor is likely to fail soon
if prediction[0] == 1:
    # Schedule maintenance for the sensor
    print('Sensor is likely to fail soon. Schedule maintenance.')
else:
    # The sensor is likely to be in good condition
    print('Sensor is in good condition.')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Generate synthetic sensor data
np.random.seed(0)
n_samples = 1000
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1000, 100, n_samples)
vibration = np.random.normal(2, 1, n_samples)
failure_indicator = np.random.randint(2, size=n_samples)

sensor_data = pd.DataFrame({'temperature': temperature, 'pressure': pressure, 'vibration': vibration, 'failure_indicator': failure_indicator})

# Prepare the data for training
condition_indicators = sensor_data[['temperature', 'pressure', 'vibration']]
labels = sensor_data['failure_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(condition_indicators, labels, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model performance
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Use the trained model to predict the failure of sensors in real time
# Collect sensor data from the sensors
new_temperature = 27.5
new_pressure = 950
new_vibration = 1.8

# Make a prediction on the new sensor data
new_sensor_data = scaler.transform([[new_temperature, new_pressure, new_vibration]])
prediction = model.predict(new_sensor_data)

# If the prediction is 1, then the sensor is likely to fail soon
if prediction[0] == 1:
    # Schedule maintenance for the sensor
    print('Sensor is likely to fail soon. Schedule maintenance.')
else:
    # The sensor is likely to be in good condition
    print('Sensor is in good condition.')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Generate synthetic sensor data
np.random.seed(0)
n_samples = 1000
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1000, 100, n_samples)
vibration = np.random.normal(2, 1, n_samples)
failure_indicator = np.random.randint(2, size=n_samples)

sensor_data = pd.DataFrame({'temperature': temperature, 'pressure': pressure, 'vibration': vibration, 'failure_indicator': failure_indicator})

# Prepare the data for training
condition_indicators = sensor_data[['temperature', 'pressure', 'vibration']]
labels = sensor_data['failure_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(condition_indicators, labels, test_size=0.25, random_state=42)

# Train a machine learning model (XGBoost Classifier in this case)
model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Use the trained model to predict the failure of sensors in real time
# Collect sensor data from the sensors
new_temperature = 27.5
new_pressure = 950
new_vibration = 1.8

# Make a prediction on the new sensor data
new_sensor_data = np.array([[new_temperature, new_pressure, new_vibration]])
prediction = model.predict(new_sensor_data)

# If the prediction is 1, then the sensor is likely to fail soon
if prediction[0] == 1:
    # Schedule maintenance for the sensor
    print('Sensor is likely to fail soon. Schedule maintenance.')
else:
    # The sensor is likely to be in good condition
    print('Sensor is in good condition.')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Generate synthetic sensor data
np.random.seed(0)
n_samples = 20000
temperature = np.random.normal(25, 5, n_samples)
pressure = np.random.normal(1000, 100, n_samples)
vibration = np.random.normal(2, 1, n_samples)
failure_indicator = np.random.randint(2, size=n_samples)

sensor_data = pd.DataFrame({'temperature': temperature, 'pressure': pressure, 'vibration': vibration, 'failure_indicator': failure_indicator})

# Prepare the data for training
condition_indicators = sensor_data[['temperature', 'pressure', 'vibration']]
labels = sensor_data['failure_indicator']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(condition_indicators, labels, test_size=0.25, random_state=42)

# Train a machine learning model (Decision Tree Classifier)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Use the trained model to predict the failure of sensors in real time
# Collect sensor data from the sensors
new_temperature = 27.5
new_pressure = 950
new_vibration = 1.8

# Make a prediction on the new sensor data
new_sensor_data = [[new_temperature, new_pressure, new_vibration]]
prediction = model.predict(new_sensor_data)

# If the prediction is 1, then the sensor is likely to fail soon
if prediction[0] == 1:
    # Schedule maintenance for the sensor
    print('Sensor is likely to fail soon. Schedule maintenance.')
else:
    # The sensor is likely to be in good condition
    print('Sensor is in good condition.')

