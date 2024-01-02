import os
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, UpSampling2D
from keras.layers import MaxPooling2D, SpatialDropout2D, concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1  # Grayscale images have 1 channel

NUM_TEST_IMAGES = 10

# Data paths
train_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/train'
val_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/val'
test_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/test'

# Define data augmentation parameters
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    # rotation_range=30,
    # zoom_range=0.2,
    # shear_range=0.2,
    # fill_mode='nearest'
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

        # Resize mask to match the model's output size
        mask = cv2.resize(mask, (64, 64))
        mask = np.expand_dims(mask, axis=-1)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)


def gaussian_noise(image, noise_factor=0.5):
    """
    Add random noise to the image.
    """
    row, col, _ = image.shape
    gauss = np.random.normal(0, noise_factor, (row, col, 1))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy

def initial_block(inputs, filters, kernel_size=3, strides=2):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def bottleneck_block(inputs, filters, dilated=False, asymmetric=False, dropout_rate=0.1):
    internal = filters // 4

    # Main branch
    x = Conv2D(internal, 1, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if not dilated and not asymmetric:
        x = Conv2D(internal, 3, padding='same', use_bias=False)(x)
    elif asymmetric:
        x = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(x)
        x = Conv2D(internal, (asymmetric, 1), padding='same', use_bias=False)(x)
    elif dilated:
        x = Conv2D(internal, 3, padding='same', dilation_rate=(dilated, dilated), use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 1, use_bias=False)(x)
    x = BatchNormalization()(x)

    # Skip connection
    if inputs.shape[-1] == filters:
        # If the number of channels matches, add inputs directly to x (skip connection)
        res = Add()([x, inputs])
    else:
        # Use a convolutional layer to match the channels of the inputs
        res = Conv2D(filters, 1, use_bias=False)(inputs)
        res = BatchNormalization()(res)
        res = Add()([x, res])

    res = Activation('relu')(res)
    res = SpatialDropout2D(dropout_rate)(res)  # Dropout for regularization

    return res

def downsample_block(inputs, filters, dropout_rate=0.1):
    x = MaxPooling2D(pool_size=2, strides=2)(inputs)

    # Bottleneck block
    x = bottleneck_block(x, filters, dropout_rate=dropout_rate)

    return x

def upsample_block(inputs, filters, skip_connection, dropout_rate=0.1):
    x = Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(inputs)

    # Concatenate with skip connection
    x = concatenate([x, skip_connection], axis=-1)

    # Bottleneck block
    x = bottleneck_block(x, filters, dropout_rate=dropout_rate)

    return x

def ENet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1):
    inputs = Input(shape=input_shape)

    # Initial Block
    initial = initial_block(inputs, filters=13, kernel_size=3, strides=2)

    # Downsample Blocks
    downsample1 = downsample_block(initial, filters=64, dropout_rate=0.01)
    downsample2 = downsample_block(downsample1, filters=128, dropout_rate=0.01)
    downsample3 = downsample_block(downsample2, filters=128, dropout_rate=0.01)

    # Bottleneck Blocks
    bottleneck1 = bottleneck_block(downsample3, filters=128, dropout_rate=0.1)
    bottleneck2 = bottleneck_block(bottleneck1, filters=128, dropout_rate=0.1)
    bottleneck3 = bottleneck_block(bottleneck2, filters=128, dropout_rate=0.1)
    bottleneck4 = bottleneck_block(bottleneck3, filters=128, dropout_rate=0.1)
    bottleneck5 = bottleneck_block(bottleneck4, filters=128, dropout_rate=0.1)

    # Upsample Blocks
    upsample1 = upsample_block(bottleneck5, filters=64, skip_connection=downsample2, dropout_rate=0.1)
    upsample2 = upsample_block(upsample1, filters=13, skip_connection=downsample1, dropout_rate=0.1)

    outputs = Conv2DTranspose(num_classes, kernel_size=2, strides=2, padding='same', activation='sigmoid')(upsample2)

    model = Model(inputs=inputs, outputs=outputs, name='enet_model')

    return model

# Load and preprocess training data with augmentation and noise
train_images, train_masks = load_and_preprocess_data(train_data_dir, augment=True, add_noise=True)

# Load and preprocess validation data
val_images, val_masks = load_and_preprocess_data(val_data_dir)

# Load and preprocess test data
test_images, test_masks = load_and_preprocess_data(test_data_dir)

# Build ENet-like binary segmentation model
enet_model = ENet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1)

# Compile ENet-like binary segmentation model
enet_model.compile(optimizer=Adam(lr=1e-4), loss=binary_crossentropy, metrics=['accuracy'])

# Display ENet-like binary segmentation model summary
enet_model.summary()

# Checkpoint to save the best model during training
checkpoint = ModelCheckpoint('enet_model.h5', save_best_only=True)

# Train the ENet-like binary segmentation model
enet_history = enet_model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=2000, batch_size=32, verbose=1,
    callbacks=[checkpoint]
)

# Evaluate the model on the test set
test_loss, test_accuracy = enet_model.evaluate(test_images, test_masks)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

predictions = enet_model.predict(test_images)
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
