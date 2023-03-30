import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator


# NOTE: MUST UNZIP DATASET.ZIP PRIOR TO RUNNING
def conv_n_n(file):
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('training_set', target_size=(64, 64), batch_size=32,
                                                     class_mode='binary')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32,
                                                class_mode='binary')

    # BUILDING THE CNN

    # Initializing CNN
    cnn = tf.keras.models.Sequential()

    # Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    # Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Second layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # TRAINING THE CNN

    # Compiling CNN
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

    # Training the CNN on Training Set, evaluating on Test set
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)

    # Making single prediction
    test_image = load_img('you-ve-got-to-be-kitten-me.jpg', target_size=(64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    indices = training_set.class_indices
    print(indices)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    print(prediction)

