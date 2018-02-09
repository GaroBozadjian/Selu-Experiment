# Selu-Experiment

## Abstract
In this experiment I used the **Selu activaton function** which is a new activation function used in Self-Normalizing Neural Network(SNN).SNN is a way that uses normalization inside the activation function rather than extranal techniques (batch norm). I used (1,2,4,8,16,32,64) dense layers, used Relu and Sigmoid activation functions with their dropout, same goes for the Selu to print and visualize the loss and the accuracy.

## Introduction
Self-Normalizing Neural Network (SNN) is part of Feed Forward Neural Network(FNN), FNN is one of the first Artificial Neural Networks (ANN), in which the information moves from the input layer(nodes), goes to the hidden layers if there are any, then to the output layer which gives us the results, there will be no loops in this process, only one direction which is forward.
As stated above  SNN  does the normalization inside the activation function which is Selu,the other activation functions don't do that, e.g., Relu or Sigmoid  the output of it has to be normalized.

## What is SELU?
Selu activation function:
![alt text] (/home/garobozadjian/Selu-Experement-/1_Q_lez8e2mP7MdSZf-O5bKw.png"Selu")

Selu has its constant variables λ and α which don't change, that means they are not hyperparameters and we can't backpropagate them and it looks like ELU but with a slight difference.

The value of α = 1.6732632423543772848170429916717 and λ = 1.0507009873554804934193349852946 

Before using the Selu we have to initialize the weights with 0 mean and sqrt(1 / neurons_below) std.dev

```python
tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=np.sqrt(
1 / n_input))
```

## Explaining the Code

I used Python language for this experiment, I used Tensorflow framework and MNIST dataset, so those are the libraries that I imported for this experiment. You can find the code with name SNN.ipynb which you can implement it in Jupyter Notebook.

```python
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

from __future__ import absolute_import, division, print_function
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
```
The parameters that we are going to use:

 learning rate for Gradient Descent Optimizer,training_epochs for forward pass and backward pass of all the training examples, batch_size for the number of training examples in forward/backward pass, display_step for displaying the epochs one by one, drop_out for drop out rate for Relu ans Sigmoid, and n_classes for MNIST total classes (0-9 digits).

 For the graph input we have x which is the number of nodes in one layer, y is the number of the classes of MNIST dataset, dropoutrate and is_training.
  ```python
  learning_rate = 0.05
training_epochs = 20 
batch_size = 100
display_step = 1  
drop_out=0.5  
n_classes = 10 

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, n_classes])
dropoutRate = tf.placeholder(tf.float32)
is_training= tf.placeholder(tf.bool)
  ```

## SELU Activation Function
After we imported the libraries and defined the parameters it's time to create the SELU activation function.
```python
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
```
## SELU Dropout
Then we are going to do another fuction that will give us the dropout for SELU activation function. 

```python
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))
```

We are going to scale the input to 0 mean, unit variance, then do a tensorboard to read the summarized data:

```python
scaler = StandardScaler().fit(mnist.train.images)

logs_path = '~/tmp'
```
## Neural Network Models
We will have two neural network models, one is for Relu and Sigmoid activation functions with there dropout and the other will be for the Selu activation function with its dropout.
For this models we have different parameters which are X is for the input, layers for the number of the hidden layers, layer_nodes is for number of nodes in all the layers,n_classes is for MNIST total classes (0-9 digits), rate is for the dropout rate that we specified at the parameters, and is_traing is for the training rate.

### Relu and Sigmoid network model:
when we are using Relu with or without dropout we change the standard deviation to stddev= np.sqrt(2/784) but when using Sigmoid we leave it as it is stddev= np.sqrt(1/784)
```python
def nn_model(x, layers, layer_nodes, n_classes,rate, is_training):
    layers_list = []
    input_layer = {'weights':tf.Variable(tf.random_normal([784, layer_nodes],stddev=np.sqrt(1/784))),
                      'biases':tf.Variable(tf.random_normal([layer_nodes],stddev=0))}

    output_layer = {'weights':tf.Variable(tf.random_normal([layer_nodes, n_classes],stddev=np.sqrt(1/layer_nodes))),
                    'biases':tf.Variable(tf.random_normal([n_classes],stddev=0))}
    if layers-1 > 0:
        l = tf.add(tf.matmul(x,input_layer['weights']), input_layer['biases'])
        l = tf.nn.sigmoid(l) # we can change this to Relue
        l = tf.nn.dropout(l,drop_out) # we can use dropout or not

        
        for i in range(layers-1):
            
            hidden_layer = {'weights':tf.Variable(tf.random_normal([layer_nodes, layer_nodes],stddev=np.sqrt(1/layer_nodes))),
                  'biases':tf.Variable(tf.random_normal([layer_nodes],stddev=0))}
            
            l = tf.add(tf.matmul(l,hidden_layer['weights']), hidden_layer['biases'])
            l = tf.nn.sigmoid(l)
            l = tf.nn.dropout(l,drop_out)
            #l = dropout_selu(l,rate, training=is_training)
        
        l = tf.matmul(l,output_layer['weights']) + output_layer['biases']
        return l
    return None
```
### Selue network model:
```python
def nn_model(x, layers, layer_nodes, n_classes,rate, is_training):
    layers_list = []
    input_layer = {'weights':tf.Variable(tf.random_normal([784, layer_nodes],stddev=np.sqrt(1/784))),
                      'biases':tf.Variable(tf.random_normal([layer_nodes],stddev=0))}

    output_layer = {'weights':tf.Variable(tf.random_normal([layer_nodes, n_classes],stddev=np.sqrt(1/layer_nodes))),
                    'biases':tf.Variable(tf.random_normal([n_classes],stddev=0))}
    if layers-1 > 0:
        l = tf.add(tf.matmul(x,input_layer['weights']), input_layer['biases'])
        l = selu(l)
        #l = dropout_selu(l,rate, training=is_training)
        
        for i in range(layers-1):
            
            hidden_layer = {'weights':tf.Variable(tf.random_normal([layer_nodes, layer_nodes],stddev=np.sqrt(1/layer_nodes))),
                  'biases':tf.Variable(tf.random_normal([layer_nodes],stddev=0))}
            
            l = tf.add(tf.matmul(l,hidden_layer['weights']), hidden_layer['biases'])
            l = selu(l)
            #l = dropout_selu(l,rate, training=is_training)
        
        l = tf.matmul(l,output_layer['weights']) + output_layer['biases']
        return l
    return None

```

After creating the network model we have to use it to find the accuracy. To do that we have to define the loss and optimizer first then test the model after that we calculate the accuracy.

```python
# Construct model
pred = nn_model(x, 17, 784, n_classes,rate=dropoutRate, is_training= is_training)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

 # Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
         
# Initializing the variables
init = tf.global_variables_initializer()
```
We will summrize the loss and the accuracy and merge them together

```python
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
```

Now it is time to show the results of our work. we do that by launching the graph that  will show as the summary of the loss and the accuracy,we will train the model with the number of epcohs, then going to loop over all the batches x and y, compute the average loss, and print the result by the number of epcohs we have.

```python
# Launch the graph
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = scaler.transform(batch_x)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y, dropoutRate: 0.05, is_training:True})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
            
            accTrain, costTrain, summary = sess.run([accuracy, cost, merged_summary_op], 
                                                        feed_dict={x: batch_x, y: batch_y, 
                                                                   dropoutRate: 0.0, is_training:False})
            summary_writer.add_summary(summary, epoch)
            
            print("Train-Accuracy:", accTrain,"Train-Loss:", costTrain)

            batch_x_test, batch_y_test = mnist.test.next_batch(512)
            batch_x_test = scaler.transform(batch_x_test)

            accTest, costVal = sess.run([accuracy, cost], feed_dict={x: batch_x_test, y: batch_y_test, 
                                                                   dropoutRate: 0.0, is_training:False})

            print("Validation-Accuracy:", accTest,"Val-Loss:", costVal,"\n")
```

### The final result will look like this:

```
Epoch: 0001 cost= nan
Train-Accuracy: 0.09 Train-Loss: nan
Validation-Accuracy: 0.0917969 Val-Loss: nan 

Epoch: 0002 cost= nan
Train-Accuracy: 0.14 Train-Loss: nan
Validation-Accuracy: 0.0859375 Val-Loss: nan 

Epoch: 0003 cost= nan
Train-Accuracy: 0.06 Train-Loss: nan
Validation-Accuracy: 0.0898438 Val-Loss: nan 

Epoch: 0004 cost= nan
Train-Accuracy: 0.17 Train-Loss: nan
Validation-Accuracy: 0.111328 Val-Loss: nan 

Epoch: 0005 cost= nan
Train-Accuracy: 0.06 Train-Loss: nan
Validation-Accuracy: 0.09375 Val-Loss: nan 

Epoch: 0006 cost= nan
Train-Accuracy: 0.12 Train-Loss: nan
Validation-Accuracy: 0.09375 Val-Loss: nan 

Epoch: 0007 cost= nan
Train-Accuracy: 0.06 Train-Loss: nan
Validation-Accuracy: 0.119141 Val-Loss: nan 

Epoch: 0008 cost= nan
Train-Accuracy: 0.2 Train-Loss: nan
Validation-Accuracy: 0.0761719 Val-Loss: nan 

Epoch: 0009 cost= nan
Train-Accuracy: 0.08 Train-Loss: nan
Validation-Accuracy: 0.0761719 Val-Loss: nan 

Epoch: 0010 cost= nan
Train-Accuracy: 0.15 Train-Loss: nan
Validation-Accuracy: 0.0976562 Val-Loss: nan 
```

## visualization

After we get the results that we need from the code above, we generate another code to read the results that we got and convert it in to line graph that we can see the difference between every activation function accuracy and loss. You can find the code with name visualize.ipynb you can implement it in Jupyter Notebook.

### The libraries used:

```python
import numpy as np
import matplotlib
%matplotlib notebook
from matplotlib import pyplot as plt
```

### Parameters:
```python
network_types = ['Relu', 'Relu+dropout','Selu', 'Selu+dropout', 'Sigmoid','Sigmoid+dropout']
network_depths = [1,2,4,8,16,32,64]
```
### Getting the Data from the results:
In this part we are going to take the important data that we need from the results that we got from the previous code. As my results is in .TXT file i have to take the parts which is important, to do that we have to open the file and read it line by line to take the accuracy and the loss as shown in the code below:

```python
train_acc = {}
val_acc = {}
for nt in network_types:
    train_acc[nt] = {}
    val_acc[nt] = {}
    for nd in network_depths:
        train_acc[nt][nd] = []
        val_acc[nt][nd] = []
        
        if nt in ['Relu', 'Selu+dropout']:
            nd_str = "{} layer{} ".format(nd, 's' if nd>1 else '')
        else:
            nd_str = "layer{} {}".format('s' if nd>1 else '', nd)
        filename = 'result/{}({}, 784 neu).txt'.format(nt, nd_str)
        
        with open(filename) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i % 4 == 0 or i % 6 == 3:
                    continue
                pieces = line.split(' ')
                if i % 4 == 1:
                    train_acc[nt][nd].append(float(pieces[1]))
                if i % 4 == 2:
                    val_acc[nt][nd].append(float(pieces[1]))
```

### Colors for each layer:

```python
colors = ['red', 'green', 'blue', 'orange', 'black', 'yellow', 'purple']
```

### Visualize the results:
Here we are visualizing the accuracy for every activation function. The x axis is the number of epochs and the y axis is the accuracy. For the lines, the straight line is for the validation accuracy and the seperate lines is for the training accuracy.

```python
fig, ax = plt.subplots(3,2,figsize=(9,10))
for i, nt in enumerate(network_types):
    current_ax = ax[i//2][i%2]
    current_ax.title.set_text(nt)
    legend = []
    for j, nd in enumerate(network_depths):
        legend.append("{} layers (train)".format(nd))
        legend.append("{} layers (val)".format(nd))
        current_ax.plot(train_acc[nt][nd], linestyle='dotted', color=colors[j])
        current_ax.plot(val_acc[nt][nd], color=colors[j])
        #current_ax.set_ylim(0.9, 1)
        
    current_ax.legend(legend, fontsize=6)
    current_ax.xaxis.label.set_text('Epochs')
    current_ax.xaxis.label.set_fontsize(6)
    current_ax.yaxis.label.set_text('Accuracy')
    current_ax.yaxis.label.set_fontsize(6)
```
## Explain the Visualization:

We are going to start explaining the activation functions without dropout first after that with it dropout.

### Activation functions without dropout:
Frist with Relu activation function, as it is shown in the graph the accuracy of training and validation sets are almost near 1.0 from the layers 1 to 16, for the 32 layers it started from 0.6 and ended near 1.0, but for 64 layers it didn't pass 0.2 because it stops learning.

Selu activation function, from the layers 1 to 4 for both validation and training accuracy near to one, but for layers 8 to 64 they didn't pass 0.2 accuracies because it stops learning.

Sigmoid activation function, for layers 1 and 2 the accuracy for both validation and training start from 0.8 and end near 1.0, layers 4 it started with low accuracy but ended almost to 0.8, but the rest didn't even pass 0.2 accuracies.

### Activation functions with dropout:

Relu with dropout, the layers 1 to 4 had accuracy between 0.8 and 1.0, for the 8 layers it started with low accuracy between 0.2 and 0.4 then increased to almost 1.0, 16 and 32 layers started low between 0.2 to 0.4 and ended near 0.6 of accuracy for both validation and training, for the 64 layers for the training accuracy it started below 0.2 and ended between 0.4 to 0.6 but for the validation accuracy started above 0.2 and ended between 0.4 to 0.6.

Selu with dropout, the 1 to 32 layers had an accuracy almost near 1.0 for both training and validation, but for the 64 layers it starts low with 0.4 accuracies and ended slightly above 0.2.

Sigmoid with dropout, the 1 layer accuracy for both validation and training started little bit below 0.8 and ended between 0.8 to 1.0, 2 layers validation accuracy started below 0.2 but improved and went up to 0.9 for the training started between 0.2 to 0.4 and ended at 0.8, but for the rest they where all below 0.2 of accuracy.

