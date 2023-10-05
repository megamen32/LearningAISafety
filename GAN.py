import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Генерируем реальные данные
def real_data_samples(n):
    return np.random.normal(0, 1, (n, 1)).astype(np.float32)


# Генератор
def build_generator(z_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(z_dim,)),
        tf.keras.layers.Dense(1)
    ])
    return model


# Дискриминатор
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# Параметры
z_dim = 5  # Размерность шума
batch_size = 64
epochs = 5000

# Создаем модели


try:
    generator = load_model('models/best_generator_model.h5')
    discriminator = load_model('models/best_discriminator_model.h5')
except:
    generator = build_generator(z_dim)
    discriminator = build_discriminator()


# Компилируем дискриминатор
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Состязательная модель
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(z_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Инициализация переменной для хранения минимального значения функции потерь генератора
min_g_loss = np.inf
# Обучение
for epoch in range(epochs):
    # Генерируем реальные и фейковые данные
    real_data = real_data_samples(batch_size)
    noise = np.random.normal(0, 1, (batch_size, z_dim)).astype(np.float32)
    fake_data = generator.predict(noise)

    # Метки для данных
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Обучаем дискриминатор
    d_loss_real = discriminator.train_on_batch(real_data, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Обучаем генератор
    noise = np.random.normal(0, 1, (batch_size, z_dim)).astype(np.float32)
    g_loss = gan.train_on_batch(noise, real_labels)

    # Выводим потери и точность
    print(f"{epoch} [D loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]}] [G loss: {g_loss[0]}]")
    # Сохраняем модель, если функция потерь генератора уменьшилась
    if g_loss[0] < min_g_loss:
        min_g_loss = g_loss[0]
        generator.save('models/best_generator_model.h5')
        discriminator.save('models/best_discriminator_model.h5')
        print(f"Saved new best generator model with loss: {min_g_loss}")

# Проверяем результаты
noise = np.random.normal(0, 1, (1000, z_dim)).astype(np.float32)
generated_data = generator.predict(noise)
plt.hist(generated_data, bins=20)
plt.show()
