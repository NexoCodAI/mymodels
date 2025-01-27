import csv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA  # Import PCA
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from tensorflow.keras import Input

# Open the CSV file
with open(r'C:\Users\jcvis\Downloads\Life-Expectancy-Data-Averaged.csv', mode='r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    
    # Read the header (if there is one)
    header = next(csvreader)
    print("Header:", header)
    
    # Initialize lists to store data
    age = []
    countries = []
    population = []
    Infant_deaths = []
    Under_five_deaths = []           
    Adult_mortality = []          
    Alcohol_consumption = []        
    Hepatitis_B = []
    Measles = []
    BMI = []  
    Polio = []
    Diphtheria = []  
    Incidents_HIV = []
    GDP_per_capita = []
    Thinness_ten_nineteen_years = []
    Thinness_five_nine_years = []
    Schooling = []
    Economy_status = []
    Region = []
    
    # Read the rows
    for row in csvreader:
        try:
            age.append(float(row[-1]))
            countries.append(row[0])
            Region.append(row[1])
            Infant_deaths.append(row[3])
            Under_five_deaths.append(row[4])
            Adult_mortality.append(row[5])
            Alcohol_consumption.append(row[6])
            Hepatitis_B.append(row[7])
            Measles.append(row[8])
            BMI.append(row[9])
            Polio.append(row[10])
            Diphtheria.append(row[11])
            Incidents_HIV.append(row[12])
            GDP_per_capita.append(float(row[13]))  # Ensure this is float
            Thinness_ten_nineteen_years.append(float(row[15]))  # Ensure this is float
            Thinness_five_nine_years.append(float(row[16]))  # Ensure this is float
            Schooling.append(float(row[17]))  # Ensure this is float
            Economy_status.append(row[18])
            population.append(int(float(row[-6])))  # Convert to float first, then to int
        except ValueError as e:
            print(f"Error converting data for row {row}: {e}")

# Convert lists to numpy arrays
y = np.array(age)
x = np.array(GDP_per_capita)

# Check lengths
print("Length of GDP_per_capita:", len(x))
print("Length of life_expectancy:", len(y))

# Ensure both arrays have the same length
if len(x) != len(y):
    print("Error: x and y must be the same size.")
else:
    # Create a scatter plot
    plt.scatter(x, y)
    plt.xlabel('GDP per Capita')
    plt.ylabel('Life Expectancy')
    plt.title('Scatter Plot of Life Expectancy vs GDP per Capita')
    plt.show()

# Statistical analysis
print("The mean is: ", np.mean(age))
print("The standard deviation is: ", np.std(age))
print("The variance is: ", np.var(age))

# Create a list of tuples (country, life expectancy)
country_life_expectancy = list(zip(countries, age))

# Sort the list by life expectancy (second element of the tuple)
sorted_country_life_expectancy = sorted(country_life_expectancy, key=lambda x: x[1])

# Print sorted countries with their life expectancy
print("Countries sorted by life expectancy (low to high):")
for country, life_expectancy in sorted_country_life_expectancy:
    print(f"{country}: {life_expectancy}")

# Classify countries based on average life expectancy
average = np.mean(age)
# Classify countries based on average life expectancy
for country, life_expectancy in sorted_country_life_expectancy:    
    if life_expectancy > average:
        print("Wealthy country", country, life_expectancy)
    else:
        print("Not wealthy country", country, life_expectancy)
        
x_gdp = np.array(GDP_per_capita)
x_schooling = np.array(Schooling)
plt.scatter(x_gdp, x_schooling)

plt.xlabel('GDP')
plt.ylabel('Schooling')

# Optionally, set the title
plt.title('Scatter Plot of Life Expectancy')

# Show the plot
plt.show()

# Load the CSV file into a DataFrame
data = pd.read_csv(r'C:\Users\jcvis\Downloads\Life-Expectancy-Data-Averaged.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values (or you can fill them with a strategy)
data.dropna(inplace=True)


# Select only numeric columns for features
X = data.select_dtypes(include=[np.number])

# Drop the 'Year' column from the features
X = X.drop(columns=['Year', 'Life_expectancy'], errors='ignore')
y = data['Life_expectancy'].values  # Target (life expectancy)

import csv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA  # Import PCA
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from tensorflow.keras import Input

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
    population = []
    
    for row in csvreader:
        age.append(float(row[-1]))  # Assuming life expectancy is the last column
        countries.append(row[0])     # Assuming country name is the first column
        GDP_per_capita.append(float(row[13]))  # Convert GDP per capita to float
        Schooling.append(float(row[17]))

# Convert lists to numpy arrays
y = np.array(age)
x = np.array(GDP_per_capita)

# Check lengths
print("Length of GDP_per_capita:", len(x))
print("Length of life_expectancy:", len(y))

# Ensure both arrays have the same length
if len(x) != len(y):
    print("Error: x and y must be the same size.")
else:
    # Create a scatter plot
    plt.scatter(x, y)
    plt.xlabel('GDP per Capita')
    plt.ylabel('Life Expectancy')
    plt.title('Scatter Plot of Life Expectancy vs GDP per Capita')
    #plt.show()


print(age)
total_sum = sum(age)

average = total_sum / len(age)
print("The mean is: ", np.mean(age))
print("The standard deviation is: ",np.std(age))
print("The variance is: ",np.var(age))
print(average)

# Create a list of tuples (country, life expectancy)
country_life_expectancy = list(zip(countries, age))

# Sort the list by life expectancy (second element of the tuple)
sorted_country_life_expectancy = sorted(country_life_expectancy, key=lambda x: x[1])

# Print sorted countries with their life expectancy
print("Countries sorted by life expectancy (low to high):")
for country, life_expectancy in sorted_country_life_expectancy:
    print(f"{country}: {life_expectancy}")

# Classify countries based on average life expectancy
for country, life_expectancy in sorted_country_life_expectancy:    
    if life_expectancy > average:
        print("Wealthy country", country, life_expectancy)
    else:
        print("Not wealthy country", country, life_expectancy)
        
x_gdp = np.array(GDP_per_capita)
x_schooling = np.array(Schooling)
plt.scatter(x_gdp, x_schooling)

plt.xlabel('GDP')
plt.ylabel('Schooling')

# Optionally, set the title
plt.title('Scatter Plot of Life Expectancy')

# Show the plot
#plt.show()

# Load the CSV file into a DataFrame
data = pd.read_csv(r'C:\Users\jcvis\Downloads\Life-Expectancy-Data-Averaged.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values (or you can fill them with a strategy)
data.dropna(inplace=True)


# Select only numeric columns for features
X = data.select_dtypes(include=[np.number])

# Drop the 'Year' column from the features
X = X.drop(columns=['Year', 'Life_expectancy'], errors='ignore')
y = data['Life_expectancy'].values  # Target (life expectancy)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot the PCA results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Life Expectancy')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Life Expectancy Data')
plt.grid()
plt.show()

   
#plt.hist(Polio, 5)
#plt.show()

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot the PCA results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Life Expectancy')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Life Expectancy Data')
plt.grid()
plt.show()



# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu'),   
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='swish'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(128, activation='swish'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Create a KerasRegressor with default parameters
model = KerasRegressor(model=create_model, verbose=0, neurons = 64, activation = 'tanh')

# Define the parameter grid for Random Search
param_dist = {
    'activation': ['relu', 'tanh', 'swish'],
    'neurons': [32, 64, 128, 256],  # This parameter will be passed to create_model
    'batch_size': [8, 16, 32, 64],
    'epochs': [100, 200, 300, 400]
}
# Random Search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3)
random_search.fit(X_train, y_train)

# Print the best parameters from Random Search
print("Best parameters from Random Search:", random_search.best_params_)

# Evaluate the best model
best_model = random_search.best_estimator_
loss = best_model.score(X_test, y_test)
print(f'Test Loss: {loss}')


# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Fit the model to the training data
model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# Make predictions
predictions = model.predict(X_test)

# Plot predictions vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs Predicted Life Expectancy')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from sklearn.decomposition import PCA
from tensorflow.keras import Input
import joblib 
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

# Load the CSV file into a DataFrame
data = pd.read_csv(r'C:\Users\jcvis\Downloads\Life-Expectancy-Data-Averaged.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Select only numeric columns for features
X = data.select_dtypes(include=[np.number])

# Drop the 'Year' column and the target column from the features
X = X.drop(columns=['Year', 'Life_expectancy'], errors='ignore')
y = data['Life_expectancy'].values  # Target (life expectancy)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# Transform the training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\scaler.pkl')

# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=5)  # Reduce to 2 dimensions
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save the PCA model
joblib.dump(pca, r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\pca_model.pkl')

'''
# Plot the PCA results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Life Expectancy')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Life Expectancy Data')
plt.grid()
plt.show()
'''

# Define a function to create the model
def create_model(activation='relu', neurons=64):
    model = keras.Sequential([
        Input(shape=(X_train_pca.shape[1],)),
        layers.Dense(neurons, activation=activation),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(neurons, activation=activation, kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(64, activation='swish'),
        layers.Dense(128, activation='relu'),
        layers.Dense(neurons, activation=activation, kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='swish'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Create a KerasRegressor with default parameters
model = KerasRegressor(model=create_model, verbose=0, neurons = 128, activation = 'relu')

# Define the parameter grid for Random Search
param_dist = {
    'activation': ['relu', 'tanh', 'swish'],
    'neurons': [32, 64, 128, 256],
    'batch_size': [8, 16, 32, 64],
    'epochs': [100, 200, 300, 400]
}

# Random Search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3)
random_search.fit(X_train_pca, y_train)

# Print the best parameters from Random Search
print("Best parameters from Random Search:", random_search.best_params_)

# Evaluate the best model
best_model = random_search.best_estimator_
loss = best_model.score(X_test_pca, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = best_model.predict(X_test_pca)

# Assuming 'random_search' is your RandomizedSearchCV object
results = random_search.cv_results_

# Create a DataFrame for easier manipulation
results_df = pd.DataFrame(results)

# Plotting mean test scores vs. different parameters
plt.figure(figsize=(12, 6))

# Example: Plotting mean test score against 'neurons' parameter
plt.subplot(1, 2, 1)
plt.scatter(results_df['param_neurons'], results_df['mean_test_score'], alpha=0.5)
plt.xlabel('Number of Neurons')
plt.ylabel('Mean Test Score')
plt.title('Mean Test Score vs. Number of Neurons')

# Example: Plotting mean fit time vs. different parameters
plt.subplot(1, 2, 2)
plt.scatter(results_df['param_batch_size'], results_df['mean_fit_time'], alpha=0.5)
plt.xlabel('Batch Size')
plt.ylabel('Mean Fit Time (seconds)')
plt.title('Mean Fit Time vs. Batch Size')

plt.tight_layout()
plt.show()

# Plot predictions vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs Predicted Life Expectancy')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()

# Access the underlying Keras model from the best estimator
underlying_model = best_model.model_  # This should now be the Keras model

# Check the type of the underlying model
print(type(underlying_model))  # Should be <class 'tensorflow.python.keras.engine.sequential.Sequential'>

# Save the underlying Keras model
if isinstance(underlying_model, keras.Model):
    underlying_model.save(r'C:\Users\jcvis\OneDrive\Escritorio\Ejercicios Phyton\Life expectancy.keras')
else:
    print("The underlying model is not a Keras model.")