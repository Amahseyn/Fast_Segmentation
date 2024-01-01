import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, UpSampling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1  # Update to 1 for grayscale images

NUM_TEST_IMAGES = 10

# Data paths
train_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/train'
val_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/val'
test_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/test'

# Define data augmentation parameters
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

def gaussian_noise(image, noise_factor=0.5):
    """
    Add random noise to the image.
    """
    row, col, _ = image.shape
    gauss = np.random.normal(0, noise_factor, (row, col, 1))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy

def load_and_preprocess_data(data_dir, augment=False, add_noise=False):
    images = []
    masks = []

    for image_filename in os.listdir(os.path.join(data_dir, 'images')):
        image_path = os.path.join(data_dir, 'images', image_filename)
        mask_path = os.path.join(data_dir, 'masks', image_filename)

        # Read and preprocess image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Read and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask >= 10).astype(np.float32)  # Ensure the correct data type
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

        # # Apply noise if specified
        # if add_noise:
        #     image = gaussian_noise(image)

        images.append(image)
        masks.append(mask)

        # Apply data augmentation if specified
        if augment:
            seed = np.random.randint(1, 1000)
            image = datagen.random_transform(image, seed=seed)
            mask = datagen.random_transform(mask, seed=seed)

            # Ensure the values are still in the [0, 1] range
            image = np.clip(image, 0, 1)
            mask = np.clip(mask, 0, 1)

            images.append(image)
            masks.append(mask)

    return np.array(images), np.array(masks)

# Load and preprocess training data with augmentation and noise
train_images, train_masks = load_and_preprocess_data(train_data_dir, augment=True, add_noise=True)

# Load and preprocess validation data
val_images, val_masks = load_and_preprocess_data(val_data_dir)

# Load and preprocess test data
test_images, test_masks = load_and_preprocess_data(test_data_dir)

def conv_block(inputs, filters, kernel_size=3, strides=1, use_batch_norm=True):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def residual_block(inputs, filters):
    x = conv_block(inputs, filters)
    x = conv_block(x, filters)
    x = conv_block(x, filters, use_batch_norm=False)
    shortcut = conv_block(inputs, filters, kernel_size=1, use_batch_norm=False)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def FastSCNN(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1):
    inputs = Input(input_shape)

    # Encoder
    x = conv_block(inputs, 64, strides=2)
    x = conv_block(x, 96, strides=2)
    x = conv_block(x, 128, strides=2)

    # Middle
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = conv_block(x, 96)
    x = UpSampling2D(size=(2, 2))(x)
    x = conv_block(x, 64)
    x = UpSampling2D(size=(2, 2))(x)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='FastSCNN')
    return model

# Build FastSCNN model
fastscnn_model = FastSCNN(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1)

# Compile FastSCNN model
fastscnn_model.compile(optimizer=Adam(lr=1e-4), loss=binary_crossentropy, metrics=['accuracy'])

# Display FastSCNN model summary
fastscnn_model.summary()

# Checkpoint to save the best model during training
checkpoint = ModelCheckpoint('fastscnn_model.h5', save_best_only=True)

# Define a learning rate schedule
def lr_schedule(epoch):
    
    # if epoch <=400:
    #   lr = 1e-2
    # elif epoch > 400 and epoch <=100 :
    #     lr = 1e-3
    # elif epoch > 1000 and epoch <=2000 :
    #     lr = 1e-4
    # elif epoch > 2000 and epoch <=2500 :
    #     lr = 1e-5
    # else :
    lr = 1e-4
    # if epoch > 100 and epoch <300 :
    #     lr *= 0.1
 
    return lr

# Create a learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the FastSCNN model
fastscnn_history = fastscnn_model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=1500, batch_size=16, verbose=1,
    callbacks=[checkpoint, lr_scheduler]
)

# Evaluate the model on the test set
test_loss, test_accuracy = fastscnn_model.evaluate(test_images, test_masks)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

import pandas as pd
import numpy as np
import os
import time

import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from skimage.io import imread, imshow
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dropout, Lambda
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1

NUM_TEST_IMAGES = 10

# Data paths
train_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/train'
val_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/val'
test_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/test'

def load_and_preprocess_data(data_dir):
    images = []
    masks = []

    for image_filename in os.listdir(os.path.join(data_dir, 'images')):
        image_path = os.path.join(data_dir, 'images', image_filename)
        mask_path = os.path.join(data_dir, 'masks', image_filename)

        # Read and preprocess image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Read and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 10, 255, 0).astype(np.float32)
          # Ensure the correct data type
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)

# Load and preprocess training data
train_images, train_masks = load_and_preprocess_data(train_data_dir)

# Load and preprocess validation data
val_images, val_masks = load_and_preprocess_data(val_data_dir)

# Load and preprocess test data
test_images, test_masks = load_and_preprocess_data(test_data_dir)

a= time.time()
#model = load_model("/content/drive/MyDrive/MobileNet/relu_fast_unet_aug_epoch500.h5",compile = False)
predictions = fastscnn_model.predict(test_images)
print(time.time() - a)
# Visualize some predictions
def save_predictions(images, masks, predictions, output_dir, num_samples=25):
    for i in range(num_samples):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(images[i][:,:,:])
        plt.title('Input Image')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i][:, :, 0], cmap='viridis')
        plt.title('True Mask')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i][:, :, 0], cmap='viridis')
        plt.title('Predicted Mask')

        # Save the figure
        output_path = os.path.join(output_dir, f"prediction_{i + 1}.png")
        plt.show()

# Define the output directory
output_directory = '/content/drive/MyDrive/Unet_mobile_net/predictions_N'

# Make sure the output directory exists
#os.makedirs(output_directory, exist_ok=True)

# Save predictions on the test data
save_predictions(test_images, test_masks, predictions, output_directory)
