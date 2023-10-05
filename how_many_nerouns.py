# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to generate data
def generate_data(N, noise=0.1):
    X = np.linspace(-1, 1, N)
    y = X ** 3 + noise * np.random.randn(N)
    return X, y

# Function to create and train a model
def create_and_train_model(layers, neurons, X_train, y_train, X_val, y_val, epochs=100):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
    for _ in range(layers-1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=0)
    return history.history['val_loss'][-1]

# Generate and split data
N = 100
X, y = generate_data(N)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment with different number of layers and neurons
layers_list = [1, 2, 3, 4]
neurons_list = [2, 4, 8, 16, 32]
results = {}

for layers in layers_list:
    for neurons in neurons_list:
        val_loss = create_and_train_model(layers, neurons, X_train, y_train, X_val, y_val)
        results[(layers, neurons)] = val_loss
        print(f"Layers: {layers}, Neurons: {neurons}, Validation Loss: {val_loss}")

# Visualizing the results
labels = [f"{key[0]} layers\n{key[1]} neurons" for key in results.keys()]
losses = list(results.values())

plt.figure(figsize=(15, 8))
plt.barh(labels, losses)
plt.xlabel('Validation Loss')
plt.title('Effect of Number of Layers and Neurons')
plt.show()
