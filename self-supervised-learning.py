import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Генерация данных
x_train = np.random.rand(1000, 20)  # 1000 примеров, 20 фичей

# Параметры
input_dim = x_train.shape[1]  # Количество фичей
encoding_dim = 10  # Размерность скрытого слоя

# Создание модели
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)

# Компиляция модели
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
autoencoder.fit(x_train, x_train, epochs=5000, batch_size=256, shuffle=True)
