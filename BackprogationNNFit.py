import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Разница между fit и train_on_batch
# Уровень абстракции: fit является более высокоуровневым методом по сравнению с train_on_batch. В fit многие вещи, такие как перемешивание данных, разделение на батчи, сохранение модели и так далее, могут быть автоматизированы. train_on_batch предоставляет более гибкий, но и более ручной способ обучения.
#
# Скорость: fit может быть быстрее, если у вас нет специфических нужд, которые требуют ручного управления каждым батчем, так как он оптимизирован для работы с большими наборами данных.
#
# Гибкость: train_on_batch дает больше контроля над процессом обучения. Вы можете легко вмешаться в процесс обучения, добавить кастомную логику, условия и т.д.
#
# Мониторинг и сохранение: С fit легче использовать коллбеки для мониторинга процесса обучения и автоматического сохранения модели. С train_on_batch вам придется реализовывать такую логику вручную.
#
# Валидация: fit позволяет автоматически использовать часть данных для валидации модели в процессе обучения, что удобно для отслеживания переобучения. С train_on_batch это нужно делать вручную.
#
# В целом, выбор между fit и train_on_batch зависит от ваших конкретных нужд. Если вам нужен полный контроль и гибкость, train_on_batch может быть лучшим выбором. Если вы предпочитаете удобство и автоматизацию, fit будет удобнее.
# Генерация реальных данных
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
z_dim = 5
batch_size = 64
epochs = 5000

# Создание моделей
# Попытка загрузить модели
try:
    generator = load_model('best_generator_model.h5')
    discriminator = load_model('best_discriminator_model.h5')
except:
    generator = build_generator(z_dim)
    discriminator = build_discriminator()

# Коллбеки для сохранения
checkpoint_g = ModelCheckpoint('best_generator_model.h5', save_best_only=True, monitor='loss', mode='min')
checkpoint_d = ModelCheckpoint('best_discriminator_model.h5', save_best_only=True, monitor='loss', mode='min')


# Компиляция дискриминатора
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Состязательная модель
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(z_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение
for epoch in range(epochs):
    # Генерация реальных и фейковых данных
    real_data = real_data_samples(batch_size)
    noise = np.random.normal(0, 1, (batch_size, z_dim)).astype(np.float32)
    fake_data = generator.predict(noise)

    # Метки для данных
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Обучение дискриминатора
    d_loss_real = discriminator.fit(real_data, real_labels, verbose=0, callbacks=[checkpoint_d])
    d_loss_fake = discriminator.fit(fake_data, fake_labels, verbose=0, callbacks=[checkpoint_d])
    # Обучение генератора
    g_loss = gan.fit(noise, real_labels, verbose=0, callbacks=[checkpoint_g])

    # Вывод потерь и точности
    print(f"{epoch} [D loss: {d_loss_real.history['loss'][0]} | D Accuracy: {100 * d_loss_real.history['accuracy'][0]}] [G loss: {g_loss.history['loss'][0]}]")
