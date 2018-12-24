import  tensorflow as tf
from tensorflow import keras

import  numpy as np
import cv2 as cv

# Show mat or image in python
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def main():
    print("Get fashion train and test data")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print('train_images dim = ', train_images.ndim)
    print('train_images very dim = ', train_images.shape)

    print('train_labels dim = ', train_labels.ndim)
    print('train_labels very dim = ', train_labels.shape)

    # cv.imshow("train_images first img", train_images[0])  # show first image
    # cv.waitKey(0)

    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)

    train_images, test_images = train_images/255.0, test_images/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Start train, fit train data to verify data
    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)
    print('model.predict(test_images) = ', predictions)

    predictions_single = model.predict(test_images[1].reshape(1, 28, 28))
    print('model.predict(test_images[0]) = ', predictions_single)

    maxv = np.argmax(predictions_single)
    print('maxv = ', maxv)
    print('maxv class = ', class_names[maxv])

if __name__ == "__main__":
    print(tf.__version__)
    print('start verify fashion mnist')
    main()