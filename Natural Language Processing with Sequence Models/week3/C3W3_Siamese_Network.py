#!/usr/bin/env python
# coding: utf-8

# # Creating a Siamese model: Ungraded Lecture Notebook
# 
# In this notebook you will learn how to create a siamese model in TensorFlow.

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow import math
import numpy

# Setting random seeds
numpy.random.seed(10)


# ## Siamese Model

# To create a `Siamese` model you will first need to create a LSTM model. For this you can stack layers using the`Sequential` model. To retrieve the output of both branches of the Siamese model, you can concatenate results using the `Concatenate` layer. You should be familiar with the following layers (notice each layer can be clicked to go to the docs):
#    - [`Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) groups a linear stack of layers into a [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
#    - [`Embedding`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) Maps positive integers into vectors of fixed size. It will have shape (vocabulary length X dimension of output vectors). The dimension of output vectors (called `model_dimension`in the code) is the number of elements in the word embedding. 
#    - [`LSTM`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) The Long Short-Term Memory (LSTM) layer. The number of units should be specified and should match the number of elements in the word embedding. 
#    - [`GlobalAveragePooling1D`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D) Computes global average pooling, which essentially takes the mean across a desired axis. GlobalAveragePooling1D uses one tensor axis to form groups of values and replaces each group with the mean value of that group. 
#    - [`Lambda`](https://trax-ml.readthedocs.io/en/latest/trax.layers.html#trax.layers.base.Fn)  Layer with no weights that applies the function f, which should be specified using a lambda syntax. You will use this layer to apply normalization with the function
#         - `tfmath.l2_normalize(x)`
#         
# - [`Concatenate`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate) Layer that concatenates a list of inputs. This layer will concatenate the normalized outputs of each LSTM into a single output for the model.
# - [`Input`](https://www.tensorflow.org/api_docs/python/tf/keras/Input): it is used to instantiate a Keras tensor.. Remember to set correctly the dimension and type of the input, which are batches of questions. 
# 
# 
# Putting everything together the Siamese model will look like this:

# In[2]:


vocab_size = 500
model_dimension = 128

# Define the LSTM model
LSTM = Sequential()
LSTM.add(layers.Embedding(input_dim=vocab_size, output_dim=model_dimension))
LSTM.add(layers.LSTM(units=model_dimension, return_sequences = True))
LSTM.add(layers.AveragePooling1D())
LSTM.add(layers.Lambda(lambda x: math.l2_normalize(x)))

input1 = layers.Input((None,))
input2 = layers.Input((None,))

# Concatenate two LSTMs together
conc = layers.Concatenate(axis=1)((LSTM(input1), LSTM(input2)))
    

# Use the Parallel combinator to create a Siamese model out of the LSTM 
Siamese = Model(inputs=(input1, input2), outputs=conc)

# Print the summary of the model
Siamese.summary()


# Next is a helper function that prints information for every layer:

# In[3]:


def show_layers(model, layer_prefix):
    print(f"Total layers: {len(model.layers)}\n")
    for i in range(len(model.layers)):
        print('========')
        print(f'{layer_prefix}_{i}: {model.layers[i]}\n')

print('Siamese model:\n')
show_layers(Siamese, 'Parallel.sublayers')

print('Detail of LSTM models:\n')
show_layers(LSTM, 'Serial.sublayers')


# Try changing the parameters defined before the Siamese model and see how it changes!
# 
# You will actually train this model in this week's assignment. For now you should be more familiarized with creating Siamese models using TensorFlow. **Keep it up!**

# In[ ]:




