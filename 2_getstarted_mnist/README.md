# mnist common classify
#### keras method

    Refer: mnist_lenet.py
    If you like to visulizing leaning, please install keras before training.
    $ pip install keras

	$ python mnist_lenet.py

**TensorBoard visulizing**

Step 1: Set callback for model.fit(), save log to log_path              <br>
Step 2: Train your model, and then middle result save to log_path       <br>
Step 3: Run tensorboard --logdir="full_path of log_path", show as follow    <br>
    
    $ tensorboard --logdir=[full_path of log_path]
    TensorBoard 1.13.1 at http://hddl-xpwork:6006 (Press CTRL+C to quit)

Step 4: In your explore, input: http://hddl-xpwork:6006, if not show, modify to: localhost:6006 <br>

**Callback Save middle model**
Call back save middle model, need 2 requirements: <br>
    1. set input shape  <br>
    2. set validate data when training. <br>

#### Slim method classify
    
    https://github.com/tensorflow/models/tree/master/research/slim
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    
    $ cd mnist/slim_method
    $ python3 classify_mnist.py 

# Image classify

    refer:models/tutorials/image/imagenet/classify_image.py
    $ cd ./classify_image_inceptionv3/
    $ python classify_image.py
    
# TestData
    Refer: ./mnist_data
    
