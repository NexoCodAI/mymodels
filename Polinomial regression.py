import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tensorflow import keras
from tensorflow.keras import layers

# Open the CSV file
with open(r'C:\Users\jcvis\Downloads\Life-Expectancy-Data-Averaged.csv', mode='r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)

    # Read the header (if there is one)
    header = next(csvreader)
    print("Header:", header)
    
    # Read the rows
    age = []
    countries = []
    GDP_per_capita = []
    Schooling = []
    
    for row in csvreader:
        age.append(float(row[-1]))  # Assuming life expectancy is the last column
        countries.append(row[0])     # Assuming country name is the first column
        GDP_per_capita.append(float(row[13]))  # Convert GDP per capita to float
        Schooling.append(float(row[17]))  # Convert Schooling to float

# Convert lists to numpy arrays
y = np.array(age)
X = np.array(GDP_per_capita).reshape(-1, 1)  # Reshape for polynomial features

# Check lengths
print("Length of GDP_per_capita:", len(X))
print("Length of life_expectancy:", len(y))

# Ensure both arrays have the same length
if len(X) != len(y):
    print("Error: X and y must be the same size.")
else:
    # Create a scatter plot
    plt.scatter(X, y)
    plt.xlabel('GDP per Capita')
    plt.ylabel('Life Expectancy')
    plt.title('Scatter Plot of Life Expectancy vs GDP per Capita')
    plt.show()

# Load the CSV file into a DataFrame
data = pd.read_csv(r'C:\Users\jcvis\Downloads\Life-Expectancy-Data-Averaged.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Select features and target variable
X = data[['GDP_per_capita', 'Schooling', 'Infant_deaths', 'Under_five_deaths', 
           'Adult_mortality', 'Alcohol_consumption', 'Hepatitis_B', 'Measles', 
           'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 'Population_mln', 
           'Thinness_ten_nineteen_years', 'Thinness_five_nine_years', 
           'Economy_status']]  # Use a list of column names
y = data['Life_expectancy'].values  # Target (life expectancy)

# Create polynomial features
poly = PolynomialFeatures(degree=2)  # Change degree for higher order polynomials
X_poly = poly.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='swish'),
    layers.Dense(256, activation='swish'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='swish'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.3)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Plot predictions vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs Predicted Life Expectancy')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()


# Define the input data for a new country (example values)
input_data = {
    'Infant_deaths': 150, 
    'Under_five_deaths': 170,             
    'Adult_mortality': 500,             
    'Alcohol_consumption': 5,         
    'Hepatitis_B': 98,       
    'Measles': 95,       
    'BMI': 26,      
    'Polio': 98,
    'Diphtheria': 98,    
    'Incidents_HIV': 0.02,
    'GDP_per_capita': 30100, 
    'Population_mln': 3,  
    'Thinness_ten_nineteen_years': 2, 
    'Thinness_five_nine_years': 1,
    'Schooling': 12,
    'Economy_status': 1,  # Ensure this feature is valid
}

# Convert the input data to a DataFrame without wrapping in a list
input_df = pd.DataFrame([input_data])  # This is correct

# Check the shape and columns of the input DataFrame
print("Input DataFrame shape:", input_df.shape)
print("Input DataFrame columns:", input_df.columns)

# Scale the input data using the same scaler
input_scaled = scaler.transform(input_df)

# Make predictions using the trained model
predicted_life_expectancy = model.predict(input_scaled)

# Print the predicted life expectancy
print(f'Predicted Life Expectancy: {predicted_life_expectancy[0][0]}')