import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Đây chỉ là ví dụ dữ liệu giả
train_images = np.random.rand(100, 64, 64, 3)
train_labels = tf.keras.utils.to_categorical(np.random.randint(2, size=(100,)), num_classes=2)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)  # train thật thì dùng dữ liệu khoai thật

model.save("D:/Dem_Khoai/potato_cnn.h5")
