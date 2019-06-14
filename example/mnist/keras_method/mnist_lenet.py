import  tensorflow as tf
import  numpy as np

import cv2 as cv

def old_main():
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

# keras way
from keras.datasets import mnist
import matplotlib.pyplot as plt

class MyLenet(tf.keras.Model):
    def __init__(self, num_classes=10):
        supper(MyLenet, self).__init__(name='my_model')
        self.num_classes=num_classes
        # Define your layers here.
        self.conv1=tf.keras.layers.Conv2D(filters=6, 
            kernel_size=(5, 5),strides=(1,1), padding='same', 
            data_format='channels_first', # nchw, channels_last=nhwc;
            activation='relu')
        self.conv2=tf.keras.layers.Conv2D(filters=16, 
            kernel_size=(5, 5),strides=(1,1), padding='same', 
            data_format='channels_first', # nchw, channels_last=nhwc;
            activation='relu')
        self.conv3=tf.keras.layers.Conv2D(filters=120, 
            kernel_size=(5, 5),strides=(1,1), padding='same', 
            data_format='channels_first', # nchw, channels_last=nhwc;
            activation='relu')
        self.max_pool=tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.fc1=tf.keras.layers.Dense(84, input_shape=(16,))
        self.fc2=tf.keras.layers.Dense(10, input_shape=(16,))
    def call(self, inputs):
        
        
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        return self.dense_2(x)

        
def main():
    #old_main()

    

if __name__ == "__main__":
    main()