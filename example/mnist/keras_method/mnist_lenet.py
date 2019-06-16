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

# keras way: model subclassing.
# Refer:https://blog.csdn.net/u014061630/article/details/81086564#41___a_classheaderlink_hrefml_titlePermalink_to_this_headlinea_220
import matplotlib.pyplot as plt
class MyLenet(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(MyLenet, self).__init__(name='my_model')

        self.num_classes=num_classes
        # Define your layers here.
        self.conv1=tf.keras.layers.Conv2D(filters=6, 
            kernel_size=(5, 5),strides=(1,1), padding='SAME', 
            data_format='channels_last', # channels_first=nchw, channels_last=nhwc;
            activation='relu')
        self.conv2=tf.keras.layers.Conv2D(filters=16, 
            kernel_size=(5, 5),strides=(1,1), padding='VALID', 
            data_format='channels_last', # nchw, channels_last=nhwc;
            activation='relu')
        self.conv3=tf.keras.layers.Conv2D(filters=120, 
            kernel_size=(5, 5),strides=(1,1), padding='VALID', 
            data_format='channels_last', # nchw, channels_last=nhwc;
            activation='relu')
        self.max_pool=tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.flatten=tf.keras.layers.Flatten()
        self.fc1=tf.keras.layers.Dense(84, activation='relu')
        self.fc2=tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.max_pool(self.conv1(inputs))
        x = self.max_pool(self.conv2(x))
        x = self.conv3(x)
        x = self.flatten(x);
        x = self.fc1(x)
        return self.fc2(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

def get_seqential_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=6, 
            kernel_size=(5, 5),strides=(1,1), padding='SAME', 
            data_format='channels_last', # channels_first=nchw, channels_last=nhwc;
            activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=16, 
            kernel_size=(5, 5),strides=(1,1), padding='VALID', 
            data_format='channels_last', # nchw, channels_last=nhwc;
            activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=120, 
            kernel_size=(5, 5),strides=(1,1), padding='VALID', 
            data_format='channels_last', # nchw, channels_last=nhwc;
            activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

def download_mnist():
    mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data, test_data = train_data/255.0, test_data/255.0

    # print(train_data.shape) # (60000,28,28)
    train_data = train_data[..., tf.newaxis]
    test_data = test_data[..., tf.newaxis]
    # print(train_data.shape) # (60000,28,28,1)
    # print("tf.newaxis=", tf.newaxis)

    # NHWC->NCHW, I don't know why only suport NHWC based on CPU.
    # train_data=train_data.transpose(0,3,1,2)
    # test_data=test_data.transpose(0,3,1,2)
    # print("train_data.shape =", train_data.shape) # (60000,1,28,28)

    rslt_train_lable=[]
    for idx, val in enumerate(train_labels):
        lab10=[0]*10
        lab10[val]=1
        rslt_train_lable.append(lab10)

    rslt_test_lable = []
    for idx, val in enumerate(test_labels):
        lab10=[0]*10
        lab10[val]=1
        rslt_test_lable.append(lab10)

    return train_data, np.array(rslt_train_lable), test_data, np.array(rslt_test_lable)

# subclassing can't be serialized, so can't save json
def save_model_way1(model, model_name):
    # Save json config to disk
    model_config=model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_config)
    # Save weights to disk
    model.save_weights(model_name + ".h5")

def load_model_way1(model, model_name):
    json_config=None
    with open(model_name + ".json") as json_file:
        json_config = json_file.read()
    new_model=tf.keras.models.model_from_json(json_config)
    new_model.load_weights(model_name+".h5")
    return new_model

# Subclassing can't use save.
def save_whole_model(model, model_name):
    model.save(model_name+".h5")
def load_whole_model(model_name):
    new_model=tf.keras.load_model(model_name+".h5")
    return new_model;

# Easy to train.
def train_mnist_by_subclassing_1():
    tf.debugging.set_log_device_placement = True
    tf.device("/GPU:0")
    #old_main()
    # model=MyLenet()
    model=get_seqential_model()

    train_data, train_labels, test_data, test_labels = download_mnist()
    print("train_data.shape =", train_data.shape)
    print("train_labels.shape =", train_labels.shape)

    # train_data = np.random.random((1000, 28, 28, 1))
    # train_labels = np.random.random((1000, 10))

    # loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
    # optimizer=tf.keras.optimizer.Adam()
    
    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  # validation_data=(test_data, test_labels))

    # train=True
    train=False
    if train:
        # Trains for 5 epochs.
        model.fit(train_data, train_labels, batch_size=32, epochs=5)
        model.summary()

        save_model_way1(model, "my_model_way1")
        # save_whole_model(model, "my_model_way2")

    else:
        model = load_model_way1(model, "my_model_way1")
        model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        # model = load_whole_model()
        rslt = model.evaluate(test_data, test_labels, batch_size=32)
        print("evaluate result =", rslt)
        # predications=model.predict(train_data[0])
        # print('predications =', predications)

    # Recreate the exact same model, including weights and optimizer.
    #model = keras.models.load_model('my_model.h5')

from tensorflow.python.client import device_lib
def train_mnist_by_subclassing_2():
    print(device_lib.list_local_devices())
    exit(0)
    #old_main()
    model=MyLenet()

    train_data, train_labels, test_data, test_labels = download_mnist()
    print("train_data.shape =", train_data.shape)
    print("train_labels.shape =", train_labels.shape)

    # loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
    # optimizer=tf.keras.optimizer.Adam()

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Trains for 5 epochs.
    model.fit(train_data, train_labels, batch_size=32, epochs=5)

    # Save entire model to a HDF5 file
    model.save_weights('my_model.h5')

    # Recreate the exact same model, including weights and optimizer.
    #model = keras.models.load_model('my_model.h5')

if __name__ == "__main__":
    train_mnist_by_subclassing_1()
    # train_mnist_by_subclassing_2()