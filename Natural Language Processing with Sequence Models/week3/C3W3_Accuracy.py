#!/usr/bin/env python
# coding: utf-8

# # Evaluate a Siamese model: Ungraded Lecture Notebook

# In[ ]:


import numpy as np
import tensorflow as tf
import tensorflow.math as math
import tensorflow.linalg as linalg


# ## Inspecting the necessary elements

# In this lecture notebook you will learn how to evaluate a Siamese model using the accuracy metric. Because there are many steps before evaluating a Siamese network (as you will see in this week's assignment) the necessary elements and variables are replicated here using real data from the assignment:
# 
#    - `q1`: vector with dimension `(batch_size X max_length)` containing first questions to compare in the test set.
#    - `q2`: vector with dimension `(batch_size X max_length)` containing second questions to compare in the test set.
#    
#    **Notice that for each pair of vectors within a batch $([q1_1, q1_2, q1_3, ...]$, $[q2_1, q2_2,q2_3, ...])$  $q1_i$ is associated to $q2_k$.**
#         
#         
#    - `y_test`: 1 if  $q1_i$ and $q2_k$ are duplicates, 0 otherwise.
#    
#    - `v1`: output vector from the model's prediction associated with the first questions.
#    - `v2`: output vector from the model's prediction associated with the second questions.

# You can inspect each one of these variables by running the following cells:

# In[ ]:


q1 = np.load('./data/q1.npy')
print(f'q1 has shape: {q1.shape} \n\nAnd it looks like this: \n\n {q1}\n\n')


# Notice those 1s on the right-hand side?  
# 
# Hope you remember that the value of `1` was used for padding. 

# In[ ]:


q2 = np.load('./data/q2.npy')
print(f'q2 has shape: {q2.shape} \n\nAnd looks like this: \n\n {q2}\n\n')


# In[ ]:


y_test = np.load('./data/y_test.npy')
print(f'y_test has shape: {y_test.shape} \n\nAnd looks like this: \n\n {y_test}\n\n')


# In[ ]:


v1 = np.load('./data/v1.npy')
print(f'v1 has shape: {v1.shape} \n\nAnd looks like this: \n\n {v1}\n\n')
v2 = np.load('./data/v2.npy')
print(f'v2 has shape: {v2.shape} \n\nAnd looks like this: \n\n {v2}\n\n')


# ## Calculating the accuracy
# 
# You will calculate the accuracy by iterating over the test set and checking if the model predicts right or wrong.
# 
# The first step is to set the accuracy to zero:

# In[ ]:


accuracy = 0


# You will also need the `batch size` and the `threshold` that determines if two questions are the same or not. 
# 
# **Note :A higher threshold means that only very similar questions will be considered as the same question.**

# In[ ]:


batch_size = 512 # Note: The max it can be is y_test.shape[0] i.e all the samples in test data
threshold = 0.7 # You can play around with threshold and then see the change in accuracy.


# In the assignment you will iterate over multiple batches of data but since this is a simplified version only one batch is provided. 
# 
# **Note: Be careful with the indices when slicing the test data in the assignment!**
# 
# The process is pretty straightforward:
#    - Iterate over each one of the elements in the batch
#    - Compute the cosine similarity between the predictions
#        - For computing the cosine similarity, the two output vectors should have been normalized using L2 normalization meaning their magnitude will be 1. This has been taken care of by the Siamese network you will build in the assignment. Hence the cosine similarity here is just dot product between two vectors. You can check by implementing the usual cosine similarity formula and check if this holds or not.
#    - Determine if this value is greater than the threshold (If it is, consider the two questions as the same and return 1 else 0)
#    - Compare against the actual target and if the prediction matches, add 1 to the accuracy (increment the correct prediction counter)
#    - Divide the accuracy by the number of processed elements

# In[ ]:


for j in range(batch_size):        # Iterate over each element in the batch
    d = math.reduce_sum(v1[j]*v2[j])# Compute the cosine similarity between the predictions as l2 normalized, ||v1[j]||==||v2[j]||==1 so only dot product is needed
    res = d > threshold            # Determine if this value is greater than the threshold (if it is consider the two questions as the same)
    accuracy += tf.cast(y_test[j] == res, tf.int32) # Compare against the actual target and if the prediction matches, add 1 to the accuracy

accuracy = accuracy / batch_size   # Divide the accuracy by the number of processed elements


# In[ ]:


print(f'The accuracy of the model is: {accuracy}')


# This implementation with the `for` loop is great for understanding exactly what is going on when you are computing the accuracy, however you can implement this in a more efficient way using vectorization. You will do this in the assignment.
# 
# Feel free to add a new cell and try it out with this small dataset, and check that you are getting the same results. You can do this with exactly the same functions you used in the cell above.

# **Congratulations on finishing this lecture notebook!** 
# 
# Now you should have a clearer understanding of how to evaluate your Siamese language models using the accuracy metric. 
# 
# **Keep it up!**
