import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- 1. Settings ---
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10 

# --- 2. Professional Data Pipeline ---
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory('dataset/Train', 
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
val_gen = val_datagen.flow_from_directory('dataset/Validation', 
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

# --- 3. Building the Forensic Model ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = True
for layer in base_model.layers[:-30]: # Unfreeze top 30 layers for detail detection
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', metrics=['accuracy'])

# --- 4. Training & Saving ---
if not os.path.exists('model'): os.makedirs('model')
checkpoint = tf.keras.callbacks.ModelCheckpoint('model/deepfake_final.h5', save_best_only=True)

print("🚀 Starting Professional Training Run...")
model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=[checkpoint])