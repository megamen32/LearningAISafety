import numpy as np


# Функция активации (сигмоида) и её производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Входные данные
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
#Входной слой нейронной сети состоит из нейронов, каждый из которых принимает одну компоненту входного вектора данных. В данном случае, каждый входной вектор имеет две компоненты (или "фичи"). Например, в случае задачи XOR, у нас есть четыре возможных входных вектора: Каждый вектор имеет две компоненты, поэтому на входной слой поступают два сигнала.
#Итак, входной слой обозначается двумя числами:

#1. Количество образцов данных (в нашем случае 4: \([0, 0]\), \([0, 1]\), \([1, 0]\), \([1, 1]\)).
#2. Количество фич или атрибутов для каждого образца данных (в нашем случае 2, потому что каждый вектор состоит из двух чисел).

#и ожидаемые выходные значенияv
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Инициализация весов и смещений
input_layer_neurons = X.shape[1]  # количество входных нейронов X.shape[1] — это синтаксис NumPy для получения числа столбцов в двумерном массиве X. В контексте этой нейронной сети это означает количество фич (или атрибутов) во входных данных. В нашем случае X.shape[1] будет равно 2, что соответствует количеству нейронов на входном слое.
hidden_layer_neurons = 10  # количество скрытых нейронов, хавтило бы двух
output_neurons = 1  # количество выходных нейронов

# Рандомная инициализация весов и смещений
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

# Параметры обучения
learning_rate = 0.1
epochs = 1000000

# Обучение нейросети
for epoch in range(epochs):
    # Прямое распространение
    hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Вычисление ошибки
    error = y - predicted_output
    mse = np.mean(np.square(error))

    # Обратное распространение ошибок и обновление весов
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Обновление весов и смещений
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Вывод MSE каждые 1000 эпох
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, MSE: {mse}")
    if mse < 0.0001:
        break

print("Обучение завершено.")
print("Предсказанные значения:")
print(predicted_output.round())
