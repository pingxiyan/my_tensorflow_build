import  tensorflow as tf
import  numpy as np

import cv2 as cv

def main():
    print("start test classify mnist")
    mnist = tf.keras.datasets.mnist
    print("type mnist = ", type(mnist))

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('x_train dim = ', x_train.ndim)
    print('x_train very dim = ', x_train.shape)

    # cv.imshow("img", x_train[0])  # show first image
    # cv.waitKey(0)

    x_train, x_test = x_train/255.0, x_test/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()