# Dir Structure Introduce
#### mnist_lenet.py

Train mnist lenet by tensorflow.keras method. <br>

| Validation items                                  | comments |
| :-------------------------------------------------| --------:|
| keras sequence method                             | verified |
| keras subclass method                             | verified |
| SLIM slim_method                                  | not      |
| save_weights + json                               | verified |
| model.save(model_name+".h5")                      | verified |
| hdf5 model convert to save_pd                     | verified |
| model.evaluate()                                  | verified |
| model.predict()                                   | verified |
| model.predict() opencv mat                        | verified |
| tensorboard+tensorboard visualize train result    | verified |
| tensorboard+save middle model(checkpoint)         | verified |
| INTEL OpenVINO convert pd model to IR, inference  | verified,[test_cpp] |

#### mnist_fashion.py

Train mnist fashin by tensorflow.keras method.

#### test_cpp

OpenVINO support INTEL CPU GPU..., we can infer model by INTEL OpenVINO. <br>
This is a simple demo, convert tensorflow pd model to IR, and then inference OpenCV Mat. <br>

# Get Started
    
    Train lenet and test model(modify code)
    $ python3 mnist_lenet.py

# Some Note
#### TensorBoard visulizing Usage.

Step 1: Set callback for model.fit(), save log to log_path              <br>
Step 2: Train your model, and then middle result save to log_path       <br>
Step 3: Run tensorboard --logdir="full_path of log_path", show as follow    <br>
    
    $ tensorboard --logdir=[full_path of log_path]
    TensorBoard 1.13.1 at http://username:6006 (Press CTRL+C to quit)

Step 4: In your explore, input: http://username:6006, if not show, modify to: localhost:6006 <br>

**Callback Save middle model**
Call back save middle model, need 2 requirements: <br>
    1. set input shape  <br>
    2. set validate data when training. <br>

#### Slim method classify(Not implemented)
    
    https://github.com/tensorflow/models/tree/master/research/slim
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    $ cd mnist/slim_method
    $ python3 classify_mnist.py 

    
