from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np



import glob
import os
import shutil

import matplotlib.pyplot as plt
# импорт пакетов
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} изображений".format(cl, len(images)))
    train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))
        shutil.move(t, os.path.join(base_dir, 'train', cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
        shutil.move(v, os.path.join(base_dir, 'val', cl))
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
batch_size = 100
IMG_SHAPE = 150

image_gen = ImageDataGenerator(
    rescale=1. / 255,  # нормализация значений пикселей
    horizontal_flip=True  # горизонтальный переворот
)

# Применение преобразования к изображениям из обучающего набора данных
train_data_gen = image_gen.flow_from_directory(
    train_dir,  # путь к директории с обучающим набором данных
    target_size=(IMG_SHAPE, IMG_SHAPE),  # целевой размер изображений
    batch_size=batch_size,  # размер обучающего блока
    class_mode='sparse',
    shuffle=True  # перемешивание изображений

)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen = ImageDataGenerator(
    rescale=1. / 255,  # нормализация значений пикселей
    rotation_range=25  # поворот на 25 градусов
)

# Применение преобразования к изображениям из обучающего набора данных
train_data_gen = image_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),  # целевой размер изображений
    batch_size=batch_size,  # размер обучающего блока
    class_mode='sparse',
    shuffle=True  # перемешивание изображений

)
augmented_images = [np.array(train_data_gen[0][0][0]) for _ in range(5)]
plotImages(augmented_images)
image_gen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.3,  # Увеличение изображения до 30%
)

# Применяем метод flow_from_directory для чтения данных из директории
train_data_gen = image_gen.flow_from_directory(
    train_dir,
    batch_size=batch_size,  # Размер блока данных
    shuffle=True,  # Перемешиваем все изображения
    target_size=(IMG_SHAPE, IMG_SHAPE),  # Целевой размер изображений
    class_mode='sparse'
)
augmented_images = [np.array(train_data_gen[0][0][0]) for _ in range(5)]
plotImages(augmented_images)
image_gen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True
)

train_data_gen = image_gen_train.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    shuffle=True,
    class_mode='sparse',
    target_size=(IMG_SHAPE, IMG_SHAPE)
)
augmented_images = [np.array(train_data_gen[0][0][0]) for _ in range(5)]
plotImages(augmented_images)
image_gen_val = ImageDataGenerator(rescale=1. / 255)

val_data_gen = image_gen_val.flow_from_directory(
    directory=val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size = batch_size,
    class_mode="sparse"
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])


# компилирование модели
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 80

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // train_data_gen.batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // val_data_gen.batch_size,
    verbose=1
)
