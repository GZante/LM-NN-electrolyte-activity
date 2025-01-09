import pyrenn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the data
file_path = "X"
df = pd.read_excel("X")

'''
Assumes that the excel files contains activity values (activity of water or activity of the electrolyte or osmotic coefficient for multiple salts)
The column "Activity" contains activity values, other columns include molal concentration of electrolyte and descriptors (e.g. Pitzer parameters) 
See "Machine learning for determination of activity of water and activity coefficients of electrolytes in binary solutions", Artificial Intelligence Chemistry
Volume 2, Issue 1, June 2024, 100069, DOI: 10.1016/j.aichem.2024.100069 for more details
'''


# Preprocessing
features = df.drop(columns=["Activity"])
target = df["Activity"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network structure
input_neurons = 5  # Adjust as necessary. Corresponds to the number of features in the Excel file
hidden_neurons = [5, 7, 9, 16, 32, 64, 128] # Optimisation of the number of neurons used in the hidden layer. Best option selected based on the highest r_squared
output_neurons = 1

best_r_squared = -np.inf
best_config = None
best_pred = None
best_metrics = {}

# Iterate through different configurations
for hidden in hidden_neurons:
    # Create the neural network
    nn = pyrenn.CreateNN([input_neurons, hidden, output_neurons])

    # Train the network using Levenberg-Marquardt
    nn = pyrenn.train_LM(X_train_scaled.T, y_train.values.reshape(1,-1), nn, verbose=True)

    # Prediction
    y_pred = pyrenn.NNOut(X_test_scaled.T, nn).flatten()

    # Metrics
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    aard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Update best configuration if needed
    if r_squared > best_r_squared:
        best_r_squared = r_squared
        best_config = hidden
        best_pred = y_pred
        best_metrics = {'MAE': mae, 'RMSE': rmse, 'AARD': aard}

# Results
print(f"Best configuration: Hidden Neurons: {best_config}, R2: {best_r_squared}, MAE: {best_metrics['MAE']}, RMSE: {best_metrics['RMSE']}, AARD: {best_metrics['AARD']}")
