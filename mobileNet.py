import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

# Ссылка на задачу: https://www.kaggle.com/datasets/prasunroy/natural-images

folder_path = 'Natural Images/'
df = pd.DataFrame(columns=['Image', 'Label'])
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            image_path = os.path.join(root, file)
            label = os.path.basename(root)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            df = df.append({'Image': image, 'Label': label}, ignore_index=True)
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(df['Image'], df['Label'], test_size=0.2)

base_model = MobileNet(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

x_train_processed = np.array([cv2.resize(img, (224, 224)) for img in x_train])
x_test_processed = np.array([cv2.resize(img, (224, 224)) for img in x_test])

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)
y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

model.fit(x_train_processed, y_train_one_hot, epochs=10, batch_size=32)

test_loss, test_accuracy = model.evaluate(x_test_processed, y_test_one_hot)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)





image = cv2.imread('C:/Users/diana/Downloads/koshka.jpg')
image = cv2.resize(image, (224, 224))
preprocessed_image = preprocess_input(image)
input_data = np.expand_dims(preprocessed_image, axis=0)
predictions = model.predict(input_data)
predicted_class = np.argmax(predictions[0])
probability = predictions[0][predicted_class]
print(f'Predicted class: {predicted_class}')







