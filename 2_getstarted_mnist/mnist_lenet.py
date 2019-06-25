import  tensorflow as tf
import  numpy as np

import cv2 as cv

# TensorBoard for visulizing learning status.
# Step 1: Set callback for model.fit(), save log to log_path              <br>
# Step 2: Train your model, and then middle result save to log_path       <br>
# Step 3: Run tensorboard --logdir="full_path of log_path", show as follow    <br>
#     $ tensorboard --logdir=[full_path of log_path]
#     TensorBoard 1.13.1 at http://hddl-xpwork:6006 (Press CTRL+C to quit)
# Step 4: In your explore, input: http://hddl-xpwork:6006, if not show, modify to: localhost:6006 <br>
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# Seqential Model
def get_sequential_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=6, 
            kernel_size=(5, 5),strides=(1,1), padding='SAME', 
            data_format='channels_last', # channels_first=nchw, channels_last=nhwc;
            input_shape=(28,28,1), # have inference to save model.
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

# Subclassing Model
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
            input_shape=(28,28,1), # have inference to save model.
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

def download_mnist():
    mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    # Docker display image tip: cannot connect to X server 
    # cv.imshow("img", train_data[0])  # show first image
    # cv.waitKey(0)

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

    return (train_data, np.array(rslt_train_lable)), (test_data, np.array(rslt_test_lable))

# subclassing can't be serialized, so can't save json
def save_model_json_weight(model, model_name):
    # Save json config to disk
    model_config=model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_config)
    # Save weights to disk
    model.save_weights(model_name + ".h5")

def load_model_json_weight(model, model_name):
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
    new_model=tf.keras.models.load_model(model_name+".h5")
    return new_model;

# save keras model to pd
# =====================================================================================
from tensorflow.keras import backend as K
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
            output_names, freeze_var_names)
        return frozen_graph
def save_pd(model, svpath, svfn):
    frozen_graph = freeze_session(K.get_session(),
        output_names=[out.op.name for out in model.outputs])    
    tf.train.write_graph(frozen_graph, svpath, svfn, as_text=False)

import os
def test_pd_model():
    print("Test pd model")
    svpath = "model_pb"
    svfn = "lenet.pd"
    model_name = os.path.join(svpath, svfn)

    from tensorflow.python.platform import gfile

    f = gfile.FastGFile(model_name, 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    print("Print all layers, where we can find input and output layers")
    print("=======================================")
    with tf.Session() as sess:
        # sess.graph.as_default()
        tf.import_graph_def(graph_def)
        op = sess.graph.get_operations()
        for idx, m in enumerate(op):
            print("layer", idx, "=", m.values())
    print("=======================================")

    with tf.Session() as sess:
        sess.graph.as_default()
        # Import a serialized TensorFlow `GraphDef` protocol buffer
        # and place into the current default `Graph`.
        tf.import_graph_def(graph_def)

        # print(graph_def)

        (train_data, train_labels), (test_data, test_labels) = download_mnist()
        fn="test1.png"
        cv2.imwrite(fn, train_data[0]*255)
        img = cv2.imread(fn, 0)
        rsz = cv2.resize(img, (28,28));
        rsz = rsz.reshape(28,28,1)/255.0 # predic input dim = 4
        rsz = rsz.reshape(1,28,28,1)

        # I don't know how to get real name of input and output
        softmax_tensor = sess.graph.get_tensor_by_name('import/dense_1/Softmax:0')
        # predictions = sess.run(softmax_tensor, {'import/conv2d_1_input:0': x_test[:20]})
        #https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
        predictions = sess.run(softmax_tensor, {'import/input_1:0': rsz})
        
        print(predictions)

        print("real label =", np.argmax(train_labels[0]))
        print("pridect label =", np.argmax(predictions))

# Train way1: sequential
def train_sequential_model():
    print("start test classify mnist")

    model = get_sequential_model()

    (train_data, train_labels), (test_data, test_labels) = download_mnist()
    # train_data = np.random.random((6000, 28, 28, 1))
    # train_labels = np.random.random((6000, 10))
    # test_data = np.random.random((1000, 28, 28, 1))
    # test_labels = np.random.random((1000, 10))

    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1, save_best_only=True)

    # Train
    model.fit(train_data, train_labels, batch_size=32, epochs=5,
        validation_data=(test_data, test_labels),
        callbacks=[TensorBoard(log_dir="./lenet_tensorboard")])

    # model.summary()

    # Predict
    rslt=model.evaluate(test_data, test_labels)
    print('*1****rslt =', rslt)

    print("Verify save json weights: ")
    print("=========================")
    save_model_json_weight(model, "my_sequential_model")
    new_model = load_model_json_weight(model, "my_sequential_model")
    new_model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    rslt=new_model.evaluate(test_data, test_labels)
    print('*2****rslt =', rslt)

    print("Verify save whole model: ")
    print("=========================")
    save_whole_model(model, "my_whole_model")
    new_model2 = load_whole_model("my_whole_model")
    new_model2.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    rslt=new_model2.evaluate(test_data, test_labels)
    print('*3****rslt =', rslt)

# Easy to train.
def train_mnist_by_subclassing():
    # tf.debugging.set_log_device_placement = True
    # tf.device("/GPU:0")
    model=MyLenet()

    (train_data, train_labels), (test_data, test_labels) = download_mnist()
    # train_data = np.random.random((6000, 28, 28, 1))
    # train_labels = np.random.random((6000, 10))
    # test_data = np.random.random((1000, 28, 28, 1))
    # test_labels = np.random.random((1000, 10))

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  # validation_data=(test_data, test_labels))

    # subclass tip error:'Currently `save` requires model to be a graph network
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1, save_best_only=True)

    # Trains for 5 epochs.
    # Maybe batch_size=1024, CPU, result in non-convergence.
    model.fit(train_data, train_labels, batch_size=32, epochs=5,
        validation_data=(test_data, test_labels),
        callbacks=[TensorBoard(log_dir="./lenet_tensorboard")])

    model.summary()
    rslt = model.evaluate(test_data, test_labels, batch_size=32)
    print('*1****rslt =', rslt)

    print("Subclassing only save model like this")
    print("=====================================")
    model_name = "my_subclassing_model.h5"  # support tf and h5 format
    # model.save_weights(model_name, save_format='tf')
    model.save_weights(model_name)
    new_model=MyLenet()
    new_model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    new_model.fit(train_data[:1], train_labels[:1])
    new_model.load_weights(model_name)
    rslt = new_model.evaluate(test_data, test_labels, batch_size=32)
    print('*2****rslt =', rslt)
    print("Train finish")

    print("save_pd")
    print("====================================")
    save_pd(model, "model_pb", "lenet.pd")


import cv2
def test_image():
    new_model=MyLenet()
    new_model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    (train_data, train_labels), (test_data, test_labels) = download_mnist()
    new_model.fit(train_data[:1], train_labels[:1])

    model_name = "my_subclassing_model.h5" 
    new_model.load_weights(model_name)

    test_num = 3
    for i in range(test_num):
        fn="test"+str(i)+".png"
        print("test:", i, ",", fn)

        print("real label =", np.argmax(train_labels[i]), end=' ')
        cv2.imwrite(fn, train_data[i]*255)
        img = cv2.imread(fn, 0)
        rsz = cv2.resize(img, (28,28));
        rsz = rsz.reshape(28,28,1)/255.0 # predic input dim = 4
        rsz = rsz.reshape(1,28,28,1)

        # print(rsz.shape)
        rslt = new_model.predict(rsz)
        print("predict label =", np.argmax(rslt))
    
from tensorflow.python.client import device_lib
if __name__ == "__main__":
    print("Print devices infor")
    print(device_lib.list_local_devices())
    print("========================================")

    # train_sequential_model()
    # train_mnist_by_subclassing()
    # test_image()
    
    test_pd_model()
    print("Eixt main")