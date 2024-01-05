import os
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import random

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1  # Grayscale images have 1 channel

# Data paths
train_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/train'
val_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/val'
test_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/test'

# Define data augmentation parameters
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

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

        # Apply data augmentation if specified
        if augment:
            seed = np.random.randint(1, 1000)
            augmented = datagen.random_transform(np.concatenate([image, mask], axis=-1), seed=seed)

            # Split the augmented array back into image and mask
            image = augmented[:, :, :1]
            mask = augmented[:, :, 1:]

            # Ensure the values are still in the [0, 1] range
            image = np.clip(image, 0, 1)
            mask = np.clip(mask, 0, 1)

        # Apply noise if specified
        if add_noise:
            image = gaussian_noise(image)
            image = add_brightness(image)
        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

def gaussian_noise(image, noise_factor=.3):
    row, col, _ = image.shape
    gauss = np.random.normal(0, noise_factor, (row, col, 1))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy

def add_brightness(image, noise_factor=.3):
    row, col, _ = image.shape
    brightness = random.uniform(0, noise_factor)
    noisy = image + brightness
    noisy = np.clip(noisy, 0, 1)
    return noisy

# Build a simpler SegNet model with fewer layers
def segnet_model_simplified(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_size)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # Decreased layer
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)   # Decreased layer
    x = UpSampling2D((2, 2))(x)

    # Output layer
    outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Compile the simplified model
simplified_model = segnet_model_simplified()
simplified_model.compile(optimizer=Adam(lr=1e-6), loss=binary_crossentropy, metrics=['accuracy'])
simplified_model.summary()
# Load and preprocess data for the simplified model
simplified_train_images, simplified_train_masks = load_and_preprocess_data(train_data_dir, augment=True, add_noise=False)
simplified_val_images, simplified_val_masks = load_and_preprocess_data(val_data_dir, augment=False, add_noise=False)
simplified_test_images, simplified_test_masks = load_and_preprocess_data(test_data_dir, augment=False, add_noise=False)

# Train the simplified model
simplified_model_checkpoint = ModelCheckpoint('simplified_segnet_model.h5', save_best_only=True)
simplified_history = simplified_model.fit(simplified_train_images, simplified_train_masks, validation_data=(simplified_val_images, simplified_val_masks),
                                          batch_size=16, epochs=5000, callbacks=[simplified_model_checkpoint])

# Evaluate the simplified model on the test set
simplified_test_loss, simplified_test_accuracy = simplified_model.evaluate(simplified_test_images, simplified_test_masks)
print(f'Simplified Test Loss: {simplified_test_loss}, Simplified Test Accuracy: {simplified_test_accuracy}')
import matplotlib.pyplot as plt

predictions = simplified_model.predict(simplified_test_images)
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
        plt.imshow(predictions[i][:, :, 0], cmap='gray')
        plt.title('Predicted Mask')

        # Save the figure
        output_path = os.path.join(output_dir, f"prediction_{i + 1}.png")
        plt.show()

# Define the output directory
output_directory = '/content/drive/MyDrive/Unet_mobile_net/predictions_N'

# Make sure the output directory exists
#os.makedirs(output_directory, exist_ok=True)

# Save predictions on the test data
save_predictions(simplified_test_images, simplified_test_masks , predictions, output_directory)
simplified_model.save("/content/drive/MyDrive/MobileNet/5jan_l4_c64_lr6_ep5000.h5")
