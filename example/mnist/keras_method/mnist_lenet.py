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
import matplotlib.pyplot as plt
class MyLenet(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyLenet, self).__init__(name='my_model')

        self.num_classes=num_classes
        # Define your layers here.
        self.conv1=tf.keras.layers.Conv2D(filters=6, 
            kernel_size=(5, 5),strides=(1,1), padding='VALID', 
            data_format='channels_first', # nchw, channels_last=nhwc;
            activation='relu')
        self.conv2=tf.keras.layers.Conv2D(filters=16, 
            kernel_size=(5, 5),strides=(1,1), padding='VALID', 
            data_format='channels_first', # nchw, channels_last=nhwc;
            activation='relu')
        self.conv3=tf.keras.layers.Conv2D(filters=120, 
            kernel_size=(5, 5),strides=(1,1), padding='VALID', 
            data_format='channels_first', # nchw, channels_last=nhwc;
            activation='relu')
        self.max_pool=tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.fc1=tf.keras.layers.Dense(84, input_shape=(16,), activation='relu')
        self.fc2=tf.keras.layers.Dense(10, input_shape=(16,), activation='relu')

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.max_pool(self.conv1(inputs))
        x = self.max_pool(self.conv2(x))
        x = self.conv3(x)
        x = self.fc1(inputs)
        return self.fc2(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

def main():
    #old_main()
    model=MyLenet()

    data = np.random.random((1000, 1, 32, 32))
    labels = np.random.random((1000, 1,1,10))

    val_data = np.random.random((100, 1, 32, 32))
    val_labels = np.random.random((100, 10))


    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Trains for 5 epochs.
    model.fit(data, labels, batch_size=32, epochs=5)

if __name__ == "__main__":
    main()